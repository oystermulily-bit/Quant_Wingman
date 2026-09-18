"""Point-in-time CSI 300 candidate features that require panel context.

These features intentionally live outside ``model_core.features``.  The legacy
registry can only see OHLCV tensors, whereas the candidates below require
historical index membership, industry membership, index weights and directional
tradability.  They are persisted for Development research but are not added to
``simple_ensemble`` and do not open the Stage 5 gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CANDIDATE_FEATURE_VERSION = "hs300_context_candidates_v1"


@dataclass(frozen=True)
class CandidateFeatureSpec:
    name: str
    column: str
    expression: str
    description: str


CANDIDATE_FEATURE_SPECS = (
    CandidateFeatureSpec(
        name="MARKET_RESIDUAL_MOMENTUM_20",
        column="market_residual_momentum_20",
        expression="MOM20(i,t) - mean_LOO_market(MOM20(j,t))",
        description="20-day stock momentum minus the equal-weight momentum of all other quoted members.",
    ),
    CandidateFeatureSpec(
        name="INDUSTRY_LOO_RESIDUAL_MOMENTUM_20",
        column="industry_loo_residual_momentum_20",
        expression="MOM20(i,t) - mean_LOO_industry(MOM20(j,t))",
        description="20-day stock momentum minus the equal-weight momentum of its other industry members.",
    ),
    CandidateFeatureSpec(
        name="STOCK_RESIDUAL_VOLATILITY_20",
        column="stock_residual_volatility_20",
        expression="std_pop_20(r(i,s) - mean_LOO_industry(r(j,s)))",
        description="20-observation population volatility of the stock's industry-LOO daily residual.",
    ),
    CandidateFeatureSpec(
        name="INDUSTRY_BREADTH_LOO",
        column="industry_breadth_loo",
        expression="mean_LOO_industry(1[r(j,t) > 0])",
        description="Fraction of other valid industry members with a positive daily return.",
    ),
    CandidateFeatureSpec(
        name="INDUSTRY_DISPERSION_1D_LOO",
        column="industry_dispersion_1d_loo",
        expression="std_pop_LOO_industry(r(j,t))",
        description="Population standard deviation of other valid industry members' daily returns.",
    ),
    CandidateFeatureSpec(
        name="TRADABILITY_CROWDING_LOO",
        column="tradability_crowding_loo",
        expression="mean_LOO_industry(1[not(has_quote & can_buy_open & can_sell_open)])",
        description="Fraction of other industry members that are not fully two-way tradable at that day's open.",
    ),
    CandidateFeatureSpec(
        name="INDEX_WEIGHT_CHANGE_PCT",
        column="index_weight_change_pct",
        expression="weight_pct(i,t) - weight_pct(i,t-1); prior=0 only on a verified entry day",
        description="Daily CSI 300 weight change in percentage points, with conservative spell-boundary handling.",
    ),
    CandidateFeatureSpec(
        name="MEMBERSHIP_AGE_DAYS",
        column="membership_age_days",
        expression="calendar_days(date(t) - entry_effective_date(i,current_spell))",
        description="Calendar days elapsed since the current CSI 300 membership spell began.",
    ),
)

CANDIDATE_FEATURE_COLUMNS = tuple(spec.column for spec in CANDIDATE_FEATURE_SPECS)


def _loo_moments(
    values: pd.Series,
    groupers: list[pd.Series],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return LOO count, mean and population std without treating NaN as zero."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    valid = numeric.notna()
    zero_filled = numeric.fillna(0.0)
    grouped_values = zero_filled.groupby(groupers, sort=False, observed=True)
    grouped_valid = valid.astype(np.int64).groupby(groupers, sort=False, observed=True)
    total = grouped_values.transform("sum")
    total_sq = zero_filled.pow(2).groupby(
        groupers, sort=False, observed=True
    ).transform("sum")
    count = grouped_valid.transform("sum").astype(float)

    self_valid = valid.astype(float)
    loo_count = count - self_valid
    loo_sum = total - zero_filled
    loo_sum_sq = total_sq - zero_filled.pow(2)
    denom = loo_count.where(loo_count.gt(0))
    loo_mean = loo_sum / denom

    # A cross-sectional dispersion needs at least two peers.  Floating-point
    # cancellation can produce a tiny negative variance, so clip only after
    # the algebra has been evaluated.
    variance = (loo_sum_sq / denom) - loo_mean.pow(2)
    variance = variance.clip(lower=0.0)
    loo_std = np.sqrt(variance).where(loo_count.ge(2))
    return loo_count, loo_mean, loo_std


def _loo_fraction(
    indicator: pd.Series,
    valid: pd.Series,
    groupers: list[pd.Series],
) -> pd.Series:
    """LOO mean of a Boolean indicator over valid peers only."""
    indicator_float = indicator.astype(float).where(valid, 0.0)
    valid_int = valid.astype(np.int64)
    total = indicator_float.groupby(groupers, sort=False, observed=True).transform("sum")
    count = valid_int.groupby(groupers, sort=False, observed=True).transform("sum")
    loo_count = count - valid_int
    loo_total = total - indicator_float
    return loo_total / loo_count.where(loo_count.gt(0))


def add_candidate_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the eight registered candidates to a date/code-sorted member panel."""
    required = {
        "date",
        "code",
        "industry_code",
        "has_quote",
        "can_buy_open",
        "can_sell_open",
        "weight_pct",
        "entry_effective_date",
        "stock_return_1d",
        "MOM_20",
        "stock_residual_1d",
        "industry_breadth_loo",
    }
    if missing := required - set(panel):
        raise ValueError(f"candidate feature panel missing {sorted(missing)}")

    out = panel.sort_values(["date", "code"], kind="stable").copy()
    date_group = [out["date"]]
    industry_group = [out["date"], out["industry_code"]]

    _, out["market_momentum_20_loo"], _ = _loo_moments(out["MOM_20"], date_group)
    _, out["industry_momentum_20_loo"], _ = _loo_moments(
        out["MOM_20"], industry_group
    )
    out["market_residual_momentum_20"] = (
        out["MOM_20"] - out["market_momentum_20_loo"]
    )
    out["industry_loo_residual_momentum_20"] = (
        out["MOM_20"] - out["industry_momentum_20_loo"]
    )

    _, _, out["industry_dispersion_1d_loo"] = _loo_moments(
        out["stock_return_1d"], industry_group
    )

    residual_groups = ["code", "entry_effective_date", "industry_code"]
    out["stock_residual_volatility_20"] = (
        out.groupby(residual_groups, sort=False, observed=True, dropna=False)[
            "stock_residual_1d"
        ]
        .rolling(window=20, min_periods=20)
        .std(ddof=0)
        .reset_index(level=residual_groups, drop=True)
        .reindex(out.index)
    )

    quote_known = out["has_quote"].notna()
    status_known = (
        quote_known
        & out["can_buy_open"].notna()
        & out["can_sell_open"].notna()
    )
    fully_tradable = (
        out["has_quote"].fillna(False).astype(bool)
        & out["can_buy_open"].fillna(False).astype(bool)
        & out["can_sell_open"].fillna(False).astype(bool)
    )
    out["tradability_crowding_loo"] = _loo_fraction(
        ~fully_tradable, status_known, industry_group
    )

    out["weight_pct"] = pd.to_numeric(out["weight_pct"], errors="coerce")
    entry = pd.to_datetime(out["entry_effective_date"], errors="coerce").dt.normalize()
    previous_weight = out.groupby("code", sort=False, observed=True)["weight_pct"].shift(1)
    previous_entry = entry.groupby(out["code"], sort=False, observed=True).shift(1)
    verified_entry_day = entry.notna() & out["date"].eq(entry)
    verified_new_spell = previous_entry.notna() & entry.notna() & entry.ne(previous_entry)
    previous_weight = previous_weight.mask(verified_entry_day | verified_new_spell, 0.0)
    out["index_weight_change_pct"] = out["weight_pct"] - previous_weight

    age = (out["date"] - entry).dt.days.astype(float)
    out["membership_age_days"] = age.where(entry.notna() & age.ge(0))
    out["candidate_feature_version"] = CANDIDATE_FEATURE_VERSION
    return out
