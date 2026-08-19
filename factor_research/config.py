from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    """Versioned, fixed settings used by every candidate in a research run."""

    enabled: bool = field(
        default_factory=lambda: os.environ.get("W1NGMAN_FACTOR_RESEARCH", "1") != "0"
    )
    store_root: Path = Path("factor_research_store")
    validation_protocol: str = "wf_purge20_embargo5_oof_v2"
    feature_implementation_version: str = "auto"
    random_seed: int = 20260812
    purge_gap: int = 20
    embargo_bars: int = 5
    transaction_cost: float = 0.0003
    min_coverage: float = 0.95
    min_abs_rank_ic: float = 0.005
    min_abs_icir: float = 0.05
    min_abs_rank_icir: float = 0.05
    min_direction_ratio: float = 0.60
    min_segment_direction_ratio: float = 0.60
    min_group_monotonicity: float = 0.50
    require_bootstrap_significance: bool = True
    max_correlation: float = 0.80
    high_corr_min_lgbm_delta: float = 0.003
    min_lgbm_delta: float = 0.0
    min_positive_fold_ratio: float = 0.75
    min_net_sharpe: float = -0.25
    max_accept_per_round: int = 3
    bootstrap_samples: int = 200
    bootstrap_block_size: int = 20
    lgbm_num_boost_round: int = 60
    lgbm_num_leaves: int = 15
    lgbm_learning_rate: float = 0.04
    lgbm_min_child_samples: int = 40
    lgbm_feature_fraction: float = 1.0
    lgbm_bagging_fraction: float = 1.0
    lgbm_num_threads: int = 1
    score_rank_icir_scale: float = 1.0
    score_icir_scale: float = 1.0
    score_lgbm_delta_scale: float = 0.03
    score_net_sharpe_scale: float = 2.0
    analysis_enabled: bool = field(
        default_factory=lambda: os.environ.get("W1NGMAN_ANALYSIS_UNIT", "1") != "0"
    )
    analysis_timeout_seconds: float = 60.0

    def for_symbol(self, symbol: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)
        return self.store_root / (safe or "multi_symbol")
