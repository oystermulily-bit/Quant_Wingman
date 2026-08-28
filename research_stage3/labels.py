"""Causal 1/3/5-day open-to-open label registry for stage 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LABEL_VERSION = "open_tr_t1_to_tH1_v1"


@dataclass(frozen=True)
class LabelRegistry:
    horizons: tuple[int, ...] = (1, 3, 5)
    execution_lag_bars: int = 1

    def build(
        self,
        member_panel: pd.DataFrame,
        daily_bars: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
        trading_status: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        required_panel = {"date", "code", "industry_code"}
        required_bars = {"date", "code", "open_tr", "has_quote"}
        if missing := required_panel - set(member_panel):
            raise ValueError(f"member_panel missing {sorted(missing)}")
        if missing := required_bars - set(daily_bars):
            raise ValueError(f"daily_bars missing {sorted(missing)}")
        dates = pd.DatetimeIndex(trading_dates).normalize().drop_duplicates().sort_values()
        date_position = {date: idx for idx, date in enumerate(dates)}

        signals = member_panel[["date", "code", "industry_code"]].copy()
        signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
        bars = daily_bars[["date", "code", "open_tr", "has_quote"]].copy()
        bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
        if bars.duplicated(["date", "code"]).any():
            raise ValueError("daily_bars contains duplicate date+code")
        bars["open_tr"] = pd.to_numeric(bars["open_tr"], errors="coerce")
        bars.loc[~bars["has_quote"].fillna(False).astype(bool), "open_tr"] = np.nan
        open_lookup = bars.set_index(["date", "code"])["open_tr"]

        status_lookup: pd.DataFrame | None = None
        if trading_status is not None:
            status_lookup = trading_status.copy()
            status_lookup["date"] = pd.to_datetime(status_lookup["date"]).dt.normalize()
            status_lookup = status_lookup.set_index(["date", "code"])

        outputs: list[pd.DataFrame] = []
        for horizon in self.horizons:
            block = signals.copy()
            positions = block["date"].map(date_position)
            entry_pos = positions + self.execution_lag_bars
            exit_pos = entry_pos + horizon
            valid_date = positions.notna() & (exit_pos < len(dates))
            block["entry_date"] = pd.NaT
            block["exit_date"] = pd.NaT
            if valid_date.any():
                block.loc[valid_date, "entry_date"] = dates.take(
                    entry_pos.loc[valid_date].astype(int)
                ).to_numpy()
                block.loc[valid_date, "exit_date"] = dates.take(
                    exit_pos.loc[valid_date].astype(int)
                ).to_numpy()
            entry_index = pd.MultiIndex.from_frame(block[["entry_date", "code"]])
            exit_index = pd.MultiIndex.from_frame(block[["exit_date", "code"]])
            entry_open = open_lookup.reindex(entry_index).to_numpy(dtype=float)
            exit_open = open_lookup.reindex(exit_index).to_numpy(dtype=float)
            entry_price_valid = (
                valid_date.to_numpy() & np.isfinite(entry_open) & (entry_open > 0)
            )
            exit_price_valid = (
                valid_date.to_numpy() & np.isfinite(exit_open) & (exit_open > 0)
            )
            valid_price = entry_price_valid & exit_price_valid
            gross = np.full(len(block), np.nan, dtype=float)
            gross[valid_price] = np.log(exit_open[valid_price] / entry_open[valid_price])
            block["horizon"] = int(horizon)
            block["entry_open_tr"] = entry_open
            block["exit_open_tr"] = exit_open
            block["gross_log_return"] = gross
            block["entry_blocked"] = ~entry_price_valid
            block["exit_blocked"] = ~exit_price_valid
            if status_lookup is not None:
                entry_status = status_lookup.reindex(entry_index)
                exit_status = status_lookup.reindex(exit_index)
                block["entry_blocked"] |= ~entry_status[
                    "can_buy_open"
                ].fillna(False).to_numpy(bool)
                block["exit_blocked"] |= ~exit_status[
                    "can_sell_open"
                ].fillna(False).to_numpy(bool)
            block["executable_log_return"] = gross
            block.loc[
                block["entry_blocked"] | block["exit_blocked"],
                "executable_log_return",
            ] = np.nan

            market_mean = block.groupby("date", observed=True)["gross_log_return"].transform("mean")
            group = block.groupby(["date", "industry_code"], observed=True)["gross_log_return"]
            industry_sum = group.transform("sum")
            industry_count = group.transform("count")
            denom = (industry_count - 1).replace(0, np.nan)
            industry_loo = (industry_sum - block["gross_log_return"]) / denom
            block["ABSOLUTE"] = block["gross_log_return"]
            block["MARKET_EXCESS"] = block["gross_log_return"] - market_mean
            block["INDUSTRY_EXCESS"] = block["gross_log_return"] - industry_loo
            outputs.append(block)

        wide = pd.concat(outputs, ignore_index=True)
        long = wide.melt(
            id_vars=[
                "date", "code", "industry_code", "horizon", "entry_date", "exit_date",
                "entry_open_tr", "exit_open_tr", "gross_log_return",
                "executable_log_return", "entry_blocked", "exit_blocked",
            ],
            value_vars=["ABSOLUTE", "MARKET_EXCESS", "INDUSTRY_EXCESS"],
            var_name="label_type",
            value_name="value",
        )
        long = long.rename(columns={"date": "signal_date"})
        long["label_version"] = LABEL_VERSION
        long["label_start_time"] = pd.to_datetime(long["entry_date"]).dt.tz_localize(
            "Asia/Shanghai", nonexistent="NaT", ambiguous="NaT"
        ) + pd.Timedelta(hours=9, minutes=30)
        long["label_end_time"] = pd.to_datetime(long["exit_date"]).dt.tz_localize(
            "Asia/Shanghai", nonexistent="NaT", ambiguous="NaT"
        ) + pd.Timedelta(hours=9, minutes=30)
        return long.sort_values(
            ["signal_date", "code", "horizon", "label_type"], kind="stable"
        ).reset_index(drop=True)
