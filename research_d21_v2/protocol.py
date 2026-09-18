"""Frozen D21-v2 configuration. Does not mutate D21-v1 Stage3Config."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from research_stage3.protocol import Stage3Config


STAGE3R_PROTOCOL_VERSION = "w1ngman_stage3r_d21_v2_h2_5d_v1"
STAGE4R_PROTOCOL_VERSION = "w1ngman_stage4r_d21_v2_h2_5d_v1"
HYPOTHESIS_ID = "H2_5D_SLOW_RESIDUAL_BUFFER_V1"
SIGNAL_VERSION = "SLOW_RESIDUAL_ENSEMBLE_V1"
PORTFOLIO_VERSION = "BUFFERED_TOP20_5D_V1"
SIMPLE_SIGNAL_VERSION = "SIMPLE_ENSEMBLE_V1"
NAIVE_PORTFOLIO_VERSION = "NAIVE_TOP20_5D_V1"
LOO_UNIVERSE = "PIT_CSI300_SAME_INDUSTRY_MEMBERS"


@dataclass(frozen=True)
class D21V2Config:
    hypothesis_id: str = HYPOTHESIS_ID
    signal_version: str = SIGNAL_VERSION
    portfolio_version: str = PORTFOLIO_VERSION
    stage3r_protocol: str = STAGE3R_PROTOCOL_VERSION
    stage4r_protocol: str = STAGE4R_PROTOCOL_VERSION
    loo_universe: str = LOO_UNIVERSE
    target_horizon: int = 5
    development_adaptive: bool = True
    holdout_read: bool = False
    execution_lag_bars: int = 1
    n_folds: int = 5
    purge_bars: int = 20
    embargo_bars: int = 5
    holdout_fraction: float = 0.10
    holdout_min_dates: int = 480
    holdout_min_years: int = 2
    initial_train_fraction: float = 0.40
    top_n: int = 20
    exit_rank: int = 40
    target_weight: float = 0.05
    liquidity_median_amount_20d: float = 20_000_000.0
    buy_cost_rate: float = 0.00076
    sell_cost_rate: float = 0.00126
    annual_trading_days: int = 239
    bootstrap_block_days: int = 20
    bootstrap_samples: int = 2000
    random_seed: int = 20260812
    min_industry_others: int = 5
    idio_vol_window: int = 60
    idio_vol_min_obs: int = 54
    res_mom_long: int = 120
    res_mom_skip: int = 20
    res_mom_min_obs: int = 90
    trend_window: int = 60
    trend_skip: int = 5
    trend_min_obs: int = 50
    min_oof_coverage: float = 0.95
    min_net_sharpe: float = 0.0
    min_net_annualized: float = 0.0
    min_sharpe_improvement: float = 0.15
    min_annualized_improvement: float = 0.02
    family_alpha: float = 0.05
    min_phase_agree: int = 3
    min_fold_agree: int = 3
    min_signal_agree: int = 2
    cost_multipliers: tuple[float, ...] = (0.0, 1.0, 1.5)

    def __post_init__(self) -> None:
        if self.exit_rank != 2 * self.top_n:
            raise ValueError("exit_rank must stay frozen at 2×K")
        if self.target_horizon != 5:
            raise ValueError("D21-v2 studies horizon 5 only")
        if self.holdout_read:
            raise ValueError("D21-v2 must not read Holdout")
        if self.top_n * self.target_weight > 1.0 + 1e-12:
            raise ValueError("reference target weights exceed 100%")
        if self.res_mom_long - self.res_mom_skip <= 0:
            raise ValueError("RES_MOM window must skip the recent days")
        if self.trend_window - self.trend_skip <= 0:
            raise ValueError("RES_TREND_EFF window must skip the recent days")

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def as_stage3_config(self) -> Stage3Config:
        """Costs, splits, and naive Top20 knobs reused from D21-v1."""
        return Stage3Config(
            horizons=(self.target_horizon,),
            execution_lag_bars=self.execution_lag_bars,
            n_folds=self.n_folds,
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
            holdout_fraction=self.holdout_fraction,
            holdout_min_dates=self.holdout_min_dates,
            holdout_min_years=self.holdout_min_years,
            initial_train_fraction=self.initial_train_fraction,
            top_n=self.top_n,
            target_weight=self.target_weight,
            liquidity_median_amount_20d=self.liquidity_median_amount_20d,
            buy_cost_rate=self.buy_cost_rate,
            sell_cost_rate=self.sell_cost_rate,
            annual_trading_days=self.annual_trading_days,
            bootstrap_block_days=self.bootstrap_block_days,
            bootstrap_samples=self.bootstrap_samples,
            random_seed=self.random_seed,
        )


EXPERIMENTS: tuple[dict[str, str], ...] = (
    {
        "experiment_id": "A",
        "signal_version": SIMPLE_SIGNAL_VERSION,
        "portfolio_version": NAIVE_PORTFOLIO_VERSION,
        "score_column": "simple_ensemble",
        "valid_column": "valid_signal",
        "execution": "naive",
    },
    {
        "experiment_id": "B",
        "signal_version": SIGNAL_VERSION,
        "portfolio_version": NAIVE_PORTFOLIO_VERSION,
        "score_column": "slow_residual_ensemble",
        "valid_column": "valid_slow_signal",
        "execution": "naive",
    },
    {
        "experiment_id": "C",
        "signal_version": SIMPLE_SIGNAL_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "score_column": "simple_ensemble",
        "valid_column": "valid_signal",
        "execution": "buffered",
    },
    {
        "experiment_id": "D",
        "signal_version": SIGNAL_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "score_column": "slow_residual_ensemble",
        "valid_column": "valid_slow_signal",
        "execution": "buffered",
    },
)

GATE_CANDIDATE = "D"
RELATIVE_BASELINE = "A"
HOLM_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("B_minus_A", "B", "A"),
    ("C_minus_A", "C", "A"),
    ("D_minus_C", "D", "C"),
    ("D_minus_A", "D", "A"),
)
NEW_SIGNAL_COLUMNS: tuple[str, ...] = (
    "IDIO_LOW_VOL_60",
    "RES_MOM_120_20",
    "RES_TREND_EFF_60_5",
)
