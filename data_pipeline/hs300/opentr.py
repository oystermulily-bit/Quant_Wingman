"""Causal OpenTR/CloseTR from unadjusted OHLC and exchange PRECLOSE.

First valid bar: CloseTR = 1, OpenTR = Open / PreClose.
Later bars: TR[t] = CloseTR[t-1] * price[t] / PreClose[t].
Missing or non-positive PRECLOSE/OHLC breaks the chain; the next valid bar
restarts as a new first bar. Values are never forward/back filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADJUSTMENT_TYPE = "PRECLOSE_CHAIN_V1"


def _valid_inputs(frame: pd.DataFrame) -> pd.Series:
    required = ["open", "high", "low", "close", "preclose"]
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    positive = numeric.gt(0).all(axis=1)
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    return positive & finite


def build_preclose_chain(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "open", "high", "low", "close", "preclose"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing {sorted(missing)}")
    if bars.empty:
        empty = bars.copy()
        for column in ("open_tr", "high_tr", "low_tr", "close_tr"):
            empty[column] = pd.Series(dtype="float64")
        empty["chain_valid"] = pd.Series(dtype="bool")
        empty["adjustment_type"] = pd.Series(dtype="string")
        return empty

    ordered = bars.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce").dt.normalize()
    ordered = ordered.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    valid = _valid_inputs(ordered)
    opens = pd.to_numeric(ordered["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(ordered["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(ordered["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(ordered["close"], errors="coerce").to_numpy(dtype=float)
    preclose = pd.to_numeric(ordered["preclose"], errors="coerce").to_numpy(dtype=float)
    codes = ordered["code"].astype(str).to_numpy()

    open_tr = np.full(len(ordered), np.nan)
    high_tr = np.full(len(ordered), np.nan)
    low_tr = np.full(len(ordered), np.nan)
    close_tr = np.full(len(ordered), np.nan)
    chain_valid = np.zeros(len(ordered), dtype=bool)
    prev_close_tr = np.nan
    prev_code = None

    for index in range(len(ordered)):
        code = codes[index]
        if code != prev_code:
            prev_close_tr = np.nan
            prev_code = code
        if not bool(valid.iloc[index]):
            prev_close_tr = np.nan
            continue
        if not np.isfinite(prev_close_tr):
            close_tr[index] = 1.0
            scale = 1.0 / preclose[index]
            open_tr[index] = opens[index] * scale
            high_tr[index] = highs[index] * scale
            low_tr[index] = lows[index] * scale
        else:
            scale = prev_close_tr / preclose[index]
            open_tr[index] = opens[index] * scale
            high_tr[index] = highs[index] * scale
            low_tr[index] = lows[index] * scale
            close_tr[index] = closes[index] * scale
        chain_valid[index] = np.isfinite(open_tr[index]) and open_tr[index] > 0
        if not chain_valid[index]:
            open_tr[index] = high_tr[index] = low_tr[index] = close_tr[index] = np.nan
            prev_close_tr = np.nan
            continue
        prev_close_tr = close_tr[index]

    ordered["open_tr"] = open_tr
    ordered["high_tr"] = high_tr
    ordered["low_tr"] = low_tr
    ordered["close_tr"] = close_tr
    ordered["chain_valid"] = chain_valid
    ordered["adjustment_type"] = np.where(chain_valid, ADJUSTMENT_TYPE, pd.NA)
    return ordered


def prefix_matches(full: pd.DataFrame, prefix_dates: pd.Series, code: str) -> bool:
    subset = full[full["code"].astype(str) == str(code)].copy()
    dates = pd.DatetimeIndex(pd.to_datetime(prefix_dates)).normalize()
    cut = dates.max()
    full_prefix = subset[subset["date"] <= cut].reset_index(drop=True)
    rebuilt = build_preclose_chain(full_prefix.drop(
        columns=["open_tr", "high_tr", "low_tr", "close_tr", "chain_valid", "adjustment_type"],
        errors="ignore",
    ))
    columns = ["open_tr", "close_tr", "chain_valid"]
    left = full_prefix[columns].reset_index(drop=True)
    right = rebuilt[columns].reset_index(drop=True)
    return left.equals(right)
