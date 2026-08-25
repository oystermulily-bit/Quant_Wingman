"""Derived timestamp policy for CSI 300 PIT tables."""

from __future__ import annotations

import pandas as pd

TIMEZONE = "Asia/Shanghai"
AVAILABLE_AT_HOUR = 21
TICK_EPS = 0.01
_OPEN_ENDED = {0, 18991231, 19000101, 20991231, 20999999, 99991231, 99999999}


def to_trade_date(values) -> pd.Series:
    """Parse vendor dates. Open-ended sentinels and non-dates become NaT.

    8-digit YYYYMMDD values are parsed per element. A single 0/NaN in the
    series no longer forces the whole column through nanosecond epoch parsing.
    """
    series = pd.Series(values, copy=False)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if series.empty:
        return out

    text = series.astype("string").str.strip()
    missing = text.isna() | text.isin(["", "<NA>", "None", "NaT", "nan", "NaN"])
    numeric = pd.to_numeric(text, errors="coerce")
    as_int = numeric.dropna().astype("int64")
    open_ended = as_int.index[as_int.isin(_OPEN_ENDED) | (as_int < 19000101)]
    eight = as_int.astype(str).str.fullmatch(r"\d{8}").fillna(False)
    eight_idx = as_int.index[eight.to_numpy()]
    eight_idx = eight_idx.difference(open_ended)
    if len(eight_idx):
        parsed = pd.to_datetime(
            as_int.loc[eight_idx].astype(str), format="%Y%m%d", errors="coerce"
        )
        out.loc[eight_idx] = parsed

    remaining = series.index.difference(out.dropna().index).difference(open_ended)
    remaining = remaining.difference(series.index[missing.fillna(False)])
    if len(remaining):
        parsed = pd.to_datetime(series.loc[remaining], errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
        out.loc[remaining] = parsed.dt.normalize()
    return pd.to_datetime(out).dt.normalize()


def available_at(dates) -> pd.Series:
    base = to_trade_date(dates)
    return base.dt.tz_localize(TIMEZONE) + pd.Timedelta(hours=AVAILABLE_AT_HOUR)


def session_open(dates) -> pd.Series:
    base = to_trade_date(dates)
    return base.dt.tz_localize(TIMEZONE) + pd.Timedelta(hours=9, minutes=30)


def session_close(dates) -> pd.Series:
    base = to_trade_date(dates)
    return base.dt.tz_localize(TIMEZONE) + pd.Timedelta(hours=15)
