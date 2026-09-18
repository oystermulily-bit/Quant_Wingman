"""Industry-LOO residual signals for SLOW_RESIDUAL_ENSEMBLE_V1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .protocol import D21V2Config, SIGNAL_VERSION


def _winsor_rank(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low = valid.quantile(0.01)
    high = valid.quantile(0.99)
    clipped = values.clip(lower=low, upper=high)
    return clipped.rank(method="average", pct=True)


def _safe_log_return(current: pd.Series, previous: pd.Series) -> pd.Series:
    valid = current.gt(0) & previous.gt(0)
    out = pd.Series(np.nan, index=current.index, dtype=float)
    out.loc[valid] = np.log(current.loc[valid] / previous.loc[valid])
    return out


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).sum()


def _rolling_count(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).count()


class SlowResidualEnsembleBuilder:
    """Build the frozen D21-v2 three-signal equal-weight ensemble."""

    required_panel = {"date", "code", "industry_code"}
    required_bars = {"date", "code", "close_tr", "has_quote", "amount"}

    def __init__(self, config: D21V2Config | None = None) -> None:
        self.config = config or D21V2Config()

    def build(
        self,
        member_panel: pd.DataFrame,
        daily_bars: pd.DataFrame,
        trading_dates: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        missing = self.required_panel - set(member_panel.columns)
        if missing:
            raise ValueError(f"member_panel missing {sorted(missing)}")
        missing_bars = self.required_bars - set(daily_bars.columns)
        if missing_bars:
            raise ValueError(f"daily_bars missing {sorted(missing_bars)}")
        if member_panel.duplicated(["date", "code"]).any():
            raise ValueError("member_panel contains duplicate date+code")
        if daily_bars.duplicated(["date", "code"]).any():
            raise ValueError("daily_bars contains duplicate date+code")

        members = member_panel[["date", "code", "industry_code"]].copy()
        members["date"] = pd.to_datetime(members["date"]).dt.normalize()
        members["code"] = members["code"].astype(str)
        members["is_member"] = True

        bars = daily_bars[["date", "code", "close_tr", "amount", "has_quote"]].copy()
        bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
        bars["code"] = bars["code"].astype(str)
        bars["close_tr"] = pd.to_numeric(bars["close_tr"], errors="coerce")
        bars["amount"] = pd.to_numeric(bars["amount"], errors="coerce")
        valid_quote = bars["has_quote"].fillna(False).astype(bool)
        bars.loc[~valid_quote, ["close_tr", "amount"]] = np.nan

        if trading_dates is None:
            dates = pd.DatetimeIndex(members["date"].drop_duplicates()).sort_values()
        else:
            dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize()
            dates = dates.drop_duplicates().sort_values()
        codes = pd.Index(members["code"].drop_duplicates()).sort_values()
        grid = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])

        aligned = (
            bars.set_index(["date", "code"])
            .reindex(grid)
            .join(members.set_index(["date", "code"]), how="left")
            .reset_index()
        )
        aligned["is_member"] = aligned["is_member"].fillna(False).astype(bool)
        # PIT industry only; never forward-fill current industry onto missing dates.
        aligned = aligned.sort_values(["code", "date"], kind="stable")
        grouped = aligned.groupby("code", sort=False, observed=True)
        prev_close = grouped["close_tr"].shift(1)
        aligned["r_1d"] = _safe_log_return(aligned["close_tr"], prev_close)

        valid = aligned.loc[
            aligned["is_member"]
            & aligned["industry_code"].notna()
            & aligned["r_1d"].notna(),
            ["date", "code", "industry_code", "r_1d"],
        ]
        stats = (
            valid.groupby(["date", "industry_code"], observed=True)["r_1d"]
            .agg(industry_sum="sum", industry_count="count")
            .reset_index()
        )
        aligned = aligned.merge(stats, on=["date", "industry_code"], how="left")
        others = aligned["industry_count"] - 1
        eligible = (
            aligned["is_member"]
            & aligned["r_1d"].notna()
            & others.ge(self.config.min_industry_others)
        )
        aligned["epsilon"] = np.nan
        aligned.loc[eligible, "epsilon"] = aligned.loc[eligible, "r_1d"] - (
            aligned.loc[eligible, "industry_sum"] - aligned.loc[eligible, "r_1d"]
        ) / (aligned.loc[eligible, "industry_count"] - 1)

        cfg = self.config
        grouped = aligned.groupby("code", sort=False, observed=True)
        eps = grouped["epsilon"]
        aligned["IDIO_LOW_VOL_60"] = -eps.transform(
            lambda series: series.rolling(
                cfg.idio_vol_window, min_periods=cfg.idio_vol_min_obs
            ).std(ddof=0)
        )
        long_sum = eps.transform(lambda series: _rolling_sum(series, cfg.res_mom_long))
        skip_sum = eps.transform(lambda series: _rolling_sum(series, cfg.res_mom_skip))
        long_count = eps.transform(lambda series: _rolling_count(series, cfg.res_mom_long))
        skip_count = eps.transform(lambda series: _rolling_count(series, cfg.res_mom_skip))
        n_valid = long_count - skip_count
        skip_sum = skip_sum.where(skip_count.gt(0), 0.0)
        aligned["RES_MOM_120_20"] = np.where(
            n_valid.ge(cfg.res_mom_min_obs),
            (100.0 / n_valid) * (long_sum - skip_sum),
            np.nan,
        )

        abs_eps = aligned["epsilon"].abs()
        abs_group = aligned.assign(_abs=abs_eps).groupby("code", sort=False, observed=True)["_abs"]
        trend_sum = eps.transform(lambda series: _rolling_sum(series, cfg.trend_window))
        trend_skip_sum = eps.transform(lambda series: _rolling_sum(series, cfg.trend_skip))
        trend_abs = abs_group.transform(lambda series: _rolling_sum(series, cfg.trend_window))
        trend_abs_skip = abs_group.transform(
            lambda series: _rolling_sum(series, cfg.trend_skip)
        )
        trend_count = eps.transform(lambda series: _rolling_count(series, cfg.trend_window))
        trend_skip_count = eps.transform(lambda series: _rolling_count(series, cfg.trend_skip))
        trend_n = trend_count - trend_skip_count
        trend_skip_sum = trend_skip_sum.where(trend_skip_count.gt(0), 0.0)
        trend_abs_skip = trend_abs_skip.where(trend_skip_count.gt(0), 0.0)
        numerator = trend_sum - trend_skip_sum
        denominator = (trend_abs - trend_abs_skip) + 1e-12
        aligned["RES_TREND_EFF_60_5"] = np.where(
            trend_n.ge(cfg.trend_min_obs),
            numerator / denominator,
            np.nan,
        )
        aligned["median_amount_20"] = grouped["amount"].transform(
            lambda series: series.rolling(20, min_periods=20).median()
        )

        panel = members[["date", "code", "industry_code"]].merge(
            aligned[
                [
                    "date",
                    "code",
                    "epsilon",
                    "IDIO_LOW_VOL_60",
                    "RES_MOM_120_20",
                    "RES_TREND_EFF_60_5",
                    "median_amount_20",
                    "r_1d",
                ]
            ],
            on=["date", "code"],
            how="left",
        )
        for signal in ("IDIO_LOW_VOL_60", "RES_MOM_120_20", "RES_TREND_EFF_60_5"):
            panel[f"{signal}_rank"] = panel.groupby("date", sort=False, observed=True)[signal].transform(
                _winsor_rank
            )
        rank_columns = [
            "IDIO_LOW_VOL_60_rank",
            "RES_MOM_120_20_rank",
            "RES_TREND_EFF_60_5_rank",
        ]
        panel["slow_residual_ensemble"] = panel[rank_columns].mean(axis=1, skipna=False)
        panel["valid_slow_signal"] = panel[rank_columns].notna().all(axis=1)
        panel["signal_version"] = SIGNAL_VERSION
        return panel.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
