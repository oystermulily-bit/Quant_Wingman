"""Directional open tradability from status + unadjusted quotes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .time_policy import TICK_EPS, available_at, to_trade_date


def _flag(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(False, index=series.index if series is not None else None)
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.upper()
    return (
        numeric.fillna(0).ne(0)
        | text.isin(["1", "Y", "TRUE", "T", "YES"])
    )


def build_trading_status(
    status: pd.DataFrame,
    bars: pd.DataFrame | None = None,
    *,
    tick_eps: float = TICK_EPS,
) -> pd.DataFrame:
    frame = status.copy()
    code_col = "MARKET_CODE" if "MARKET_CODE" in frame.columns else "code"
    date_col = "TRADE_DATE" if "TRADE_DATE" in frame.columns else "date"
    frame["code"] = frame[code_col].astype(str).str.upper()
    frame["date"] = to_trade_date(frame[date_col])
    frame["preclose"] = pd.to_numeric(frame.get("PRECLOSE"), errors="coerce")
    frame["limit_up_price"] = pd.to_numeric(frame.get("HIGH_LIMITED"), errors="coerce")
    frame["limit_down_price"] = pd.to_numeric(frame.get("LOW_LIMITED"), errors="coerce")
    frame["is_suspended"] = _flag(frame["IS_SUSP_SEC"]) if "IS_SUSP_SEC" in frame.columns else False
    frame["is_st"] = _flag(frame["IS_ST_SEC"]) if "IS_ST_SEC" in frame.columns else False
    frame["is_xr"] = _flag(frame["IS_XR_SEC"]) if "IS_XR_SEC" in frame.columns else False
    frame["is_wd"] = _flag(frame["IS_WD_SEC"]) if "IS_WD_SEC" in frame.columns else False

    if bars is not None and not bars.empty:
        quote = bars[["date", "code", "open", "volume", "amount"]].copy()
        quote["date"] = to_trade_date(quote["date"])
        quote["code"] = quote["code"].astype(str).str.upper()
        frame = frame.merge(quote, on=["date", "code"], how="left")
    else:
        frame["open"] = np.nan
        frame["volume"] = np.nan
        frame["amount"] = np.nan

    open_px = pd.to_numeric(frame["open"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    has_quote = open_px.gt(0) & volume.gt(0) & amount.gt(0)
    not_suspended = ~frame["is_suspended"].fillna(False).astype(bool)
    buyable = (
        has_quote
        & not_suspended
        & frame["limit_up_price"].notna()
        & (open_px < (frame["limit_up_price"] - tick_eps))
    )
    sellable = (
        has_quote
        & not_suspended
        & frame["limit_down_price"].notna()
        & (open_px > (frame["limit_down_price"] + tick_eps))
    )
    buy_reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    sell_reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    buy_reason = buy_reason.mask(~has_quote, "MISSING_QUOTE")
    sell_reason = sell_reason.mask(~has_quote, "MISSING_QUOTE")
    buy_reason = buy_reason.mask(~not_suspended, "SUSPENDED")
    sell_reason = sell_reason.mask(~not_suspended, "SUSPENDED")
    buy_reason = buy_reason.mask(
        has_quote & not_suspended & ~buyable, "LIMIT_UP_OPEN"
    )
    sell_reason = sell_reason.mask(
        has_quote & not_suspended & ~sellable, "LIMIT_DOWN_OPEN"
    )

    out = pd.DataFrame(
        {
            "date": frame["date"],
            "code": frame["code"],
            "is_suspended": frame["is_suspended"].astype(bool),
            "is_st": frame["is_st"].astype(bool),
            "is_xr": frame["is_xr"].astype(bool),
            "is_wd": frame["is_wd"].astype(bool),
            "limit_up_price": frame["limit_up_price"],
            "limit_down_price": frame["limit_down_price"],
            "preclose": frame["preclose"],
            "can_buy_open": buyable.fillna(False).astype(bool),
            "can_sell_open": sellable.fillna(False).astype(bool),
            "buy_block_reason": buy_reason,
            "sell_block_reason": sell_reason,
            "available_at": available_at(frame["date"]),
            "available_at_source": "DERIVED_POLICY",
        }
    )
    if out.duplicated(["date", "code"]).any():
        raise ValueError("trading_status has duplicate date+code")
    return out.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
