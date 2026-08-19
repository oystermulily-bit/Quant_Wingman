from __future__ import annotations

import numpy as np

from .config import ResearchConfig
from .schemas import CandidateAssessment, LGBMIncrementResult, RetentionDecision, ValidationResult


class RetentionPolicy:
    """Hard gates followed by a fixed, cross-round comparable score."""

    def __init__(self, config: ResearchConfig):
        self.config = config

    def decide(self, validation: ValidationResult, lgbm: LGBMIncrementResult) -> RetentionDecision:
        reasons: list[str] = []
        if not validation.valid:
            reasons.append(validation.failure_reason or "公式无效")
        if validation.coverage < self.config.min_coverage:
            reasons.append("数据覆盖率未达标")
        if abs(validation.rank_ic) < self.config.min_abs_rank_ic:
            reasons.append("RankIC绝对值未达标")
        if abs(validation.icir) < self.config.min_abs_icir:
            reasons.append("ICIR绝对值未达标")
        if abs(validation.rank_icir) < self.config.min_abs_rank_icir:
            reasons.append("RankICIR绝对值未达标")
        if self.config.require_bootstrap_significance:
            crosses_zero = validation.bootstrap_ci_low <= 0 <= validation.bootstrap_ci_high
            if crosses_zero:
                reasons.append("RankIC Bootstrap置信区间包含零")
        if validation.direction_ratio < self.config.min_direction_ratio:
            reasons.append("RankIC方向跨时期不稳定")
        fold_direction = [row.rank_ic for row in validation.fold_results]
        if fold_direction:
            sign = 1 if validation.rank_ic >= 0 else -1
            stable = float(np.mean(np.asarray(fold_direction) * sign > 0))
            if stable < self.config.min_positive_fold_ratio:
                reasons.append("多数Walk-Forward折RankIC方向不一致")
        if lgbm.delta <= self.config.min_lgbm_delta:
            reasons.append("固定LightGBM无正向样本外增量")
        if lgbm.positive_fold_ratio < self.config.min_positive_fold_ratio:
            reasons.append("固定LightGBM增量未在多数折为正")
        if validation.max_correlation >= self.config.max_correlation and lgbm.delta < self.config.high_corr_min_lgbm_delta:
            reasons.append("与SOTA高度相关且边际提升不足")
        if validation.net_sharpe < self.config.min_net_sharpe:
            reasons.append("扣费组合Sharpe未达标")
        if (
            validation.group_returns
            and validation.group_monotonicity < self.config.min_group_monotonicity
        ):
            reasons.append("分组收益单调性未达标")
        if (
            validation.segment_direction_ratio
            < self.config.min_segment_direction_ratio
        ):
            reasons.append("年份/市场状态/品种稳定性未达标")
        if validation.portfolio_delta < -1e-12:
            reasons.append("加入因子后扣费组合表现恶化")
        return RetentionDecision(accepted=not reasons, reasons=reasons)

    def rank(self, assessments: list[CandidateAssessment]) -> None:
        eligible = [row for row in assessments if row.decision.accepted]
        if not eligible:
            return

        def fixed(value: float, scale: float, *, absolute: bool = False) -> float:
            if absolute:
                value = abs(value)
            return float(np.clip(value / max(scale, 1e-12), 0.0, 1.0))

        for row in eligible:
            complexity = min(1.0, row.operator_count / 16.0)
            score = (
                0.25 * fixed(row.validation.rank_icir, self.config.score_rank_icir_scale, absolute=True)
                + 0.20 * fixed(row.validation.icir, self.config.score_icir_scale, absolute=True)
                + 0.25 * fixed(row.lgbm.delta, self.config.score_lgbm_delta_scale)
                + 0.20 * fixed(row.validation.net_sharpe, self.config.score_net_sharpe_scale)
                + 0.10 * fixed(row.validation.direction_ratio, 1.0)
                - 0.10 * row.validation.max_correlation
                - 0.05 * min(1.0, row.validation.turnover)
                - 0.03 * complexity
            )
            row.decision.ranking_score = float(score)
