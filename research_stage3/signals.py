"""Causal simple signals and market/industry/stock context for stage 3."""
from __future__ import annotations

import numpy as np
import pandas as pd


SIGNAL_VERSION = "simple_ensemble_v1"


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


class SimpleSignalBuilder:
    """Build MOM_20, REV_5 and LOW_VOL_20 without future observations."""

    required_panel = {"date", "code", "industry_code", "has_quote"}
    required_bars = {"date", "code", "close_tr", "amount", "has_quote"}

    def build(self, member_panel: pd.DataFrame, daily_bars: pd.DataFrame) -> pd.DataFrame:
        if missing := self.required_panel - set(member_panel):
            raise ValueError(f"member_panel missing {sorted(missing)}")
        if missing := self.required_bars - set(daily_bars):
            raise ValueError(f"daily_bars missing {sorted(missing)}")
        if member_panel.duplicated(["date", "code"]).any():
            raise ValueError("member_panel contains duplicate date+code")
        if daily_bars.duplicated(["date", "code"]).any():
            raise ValueError("daily_bars contains duplicate date+code")

        bars = daily_bars[["date", "code", "close_tr", "amount", "has_quote"]].copy()
        bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
        bars["close_tr"] = pd.to_numeric(bars["close_tr"], errors="coerce")
        bars["amount"] = pd.to_numeric(bars["amount"], errors="coerce")
        valid_quote = bars["has_quote"].fillna(False).astype(bool)
        bars.loc[~valid_quote, ["close_tr", "amount"]] = np.nan
        bars = bars.sort_values(["code", "date"], kind="stable")

        group = bars.groupby("code", sort=False, observed=True)
        close_1 = group["close_tr"].shift(1)
        close_5 = group["close_tr"].shift(5)
        close_20 = group["close_tr"].shift(20)
        bars["stock_return_1d"] = _safe_log_return(bars["close_tr"], close_1)
        bars["MOM_20"] = _safe_log_return(bars["close_tr"], close_20)
        bars["REV_5"] = -_safe_log_return(bars["close_tr"], close_5)
        bars["LOW_VOL_20"] = -(
            bars.groupby("code", sort=False, observed=True)["stock_return_1d"]
            .rolling(window=20, min_periods=20)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
        bars["median_amount_20"] = (
            bars.groupby("code", sort=False, observed=True)["amount"]
            .rolling(window=20, min_periods=20)
            .median()
            .reset_index(level=0, drop=True)
        )

        panel = member_panel.copy()
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        keep = [
            "date", "code", "stock_return_1d", "MOM_20", "REV_5", "LOW_VOL_20",
            "median_amount_20",
        ]
        panel = panel.merge(bars[keep], on=["date", "code"], how="left", validate="one_to_one")

        by_date = panel.groupby("date", sort=False, observed=True)
        panel["market_return_1d"] = by_date["stock_return_1d"].transform("mean")
        panel["market_breadth"] = by_date["stock_return_1d"].transform(
            lambda values: values.gt(0).where(values.notna()).mean()
        )
        market_daily = (
            panel[["date", "market_return_1d", "market_breadth"]]
            .drop_duplicates("date")
            .sort_values("date")
        )
        market_daily["market_volatility_20"] = market_daily["market_return_1d"].rolling(
            20, min_periods=20
        ).std(ddof=0)
        panel = panel.merge(
            market_daily[["date", "market_volatility_20"]],
            on="date",
            how="left",
            validate="many_to_one",
        )

        industry_group = panel.groupby(
            ["date", "industry_code"], sort=False, observed=True
        )["stock_return_1d"]
        industry_sum = industry_group.transform("sum")
        industry_count = industry_group.transform("count")
        positive = panel["stock_return_1d"].gt(0).where(panel["stock_return_1d"].notna())
        positive_sum = positive.groupby(
            [panel["date"], panel["industry_code"]], observed=True
        ).transform("sum")
        panel["industry_member_count_loo"] = (industry_count - 1).clip(lower=0)
        panel["industry_return_1d_loo"] = (
            (industry_sum - panel["stock_return_1d"]) / (industry_count - 1)
        ).where(industry_count > 1)
        panel["industry_breadth_loo"] = (
            (positive_sum - positive.astype(float)) / (industry_count - 1)
        ).where(industry_count > 1)
        panel["industry_relative_market_1d"] = (
            panel["industry_return_1d_loo"] - panel["market_return_1d"]
        )
        panel["stock_residual_1d"] = (
            panel["stock_return_1d"] - panel["industry_return_1d_loo"]
        )

        for signal in ("MOM_20", "REV_5", "LOW_VOL_20"):
            panel[f"{signal}_rank"] = panel.groupby(
                "date", sort=False, observed=True
            )[signal].transform(_winsor_rank)
        rank_columns = ["MOM_20_rank", "REV_5_rank", "LOW_VOL_20_rank"]
        panel["simple_ensemble"] = panel[rank_columns].mean(axis=1, skipna=False)
        panel["valid_signal"] = panel[rank_columns].notna().all(axis=1)
        panel["signal_version"] = SIGNAL_VERSION
        return panel.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
