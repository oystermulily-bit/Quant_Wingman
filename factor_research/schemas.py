from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FoldMetrics:
    fold: int
    ic: float
    rank_ic: float
    net_sharpe: float
    turnover: float


@dataclass
class ValidationResult:
    valid: bool
    coverage: float
    ic: float
    icir: float
    rank_ic: float
    rank_icir: float
    direction_ratio: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    net_return: float
    annual_return: float
    net_sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    turnover: float
    long_exposure: float
    short_exposure: float
    max_correlation: float
    portfolio_delta: float = 0.0
    fold_results: list[FoldMetrics] = field(default_factory=list)
    failure_reason: str | None = None
    invalid_fraction: float = 0.0
    yearly_rank_ic: dict[str, float] = field(default_factory=dict)
    regime_rank_ic: dict[str, float] = field(default_factory=dict)
    symbol_rank_ic: dict[str, float] = field(default_factory=dict)
    group_returns: list[float] = field(default_factory=list)
    group_monotonicity: float = 0.0
    segment_direction_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LGBMIncrementResult:
    baseline_metric: float
    challenge_metric: float
    delta: float
    positive_fold_ratio: float
    fold_deltas: list[float]
    oof_samples: int
    metric_name: str = "rank_ic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetentionDecision:
    accepted: bool
    reasons: list[str]
    ranking_score: float = -1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAssessment:
    factor_id: str
    formula_tokens: list[int]
    formula_decoded: str
    hypothesis: str
    rationale: str
    validation: ValidationResult
    lgbm: LGBMIncrementResult
    decision: RetentionDecision
    operator_count: int

    def public_summary(self) -> dict[str, Any]:
        """Aggregate-only payload safe for the remote Analysis Unit."""
        return {
            "factor_id": self.factor_id,
            "formula_tokens": self.formula_tokens,
            "formula_decoded": self.formula_decoded,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "metrics": self.validation.to_dict(),
            "lgbm_increment": self.lgbm.to_dict(),
            "decision": self.decision.to_dict(),
            "operator_count": self.operator_count,
        }


@dataclass
class AnalysisReport:
    diagnosis: str
    accepted_hypotheses: list[str]
    rejected_hypotheses: list[str]
    next_hypotheses: list[str]
    formula_mutations: list[dict[str, Any]]
    confidence: float
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
