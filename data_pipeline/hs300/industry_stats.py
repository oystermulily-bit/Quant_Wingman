"""In-universe Shenwan L1 statistics with leave-one-out returns.

Statistics at date t use only member quotes with available_at on t.
A stock never contributes to its own industry mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def industry_loo_context(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "code", "industry_code", "close_tr", "has_quote"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing {sorted(missing)}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["close_tr"] = pd.to_numeric(frame["close_tr"], errors="coerce")
    frame.loc[~frame["has_quote"].fillna(False).astype(bool), "close_tr"] = np.nan
    frame = frame.sort_values(["code", "date"], kind="stable")
    frame["return_1d"] = frame.groupby("code", sort=False)["close_tr"].transform(
        lambda series: np.log(series / series.shift(1))
    )
    valid = frame.dropna(subset=["industry_code", "return_1d"])
    grouped = valid.groupby(["date", "industry_code"], observed=True)["return_1d"]
    total = grouped.transform("sum")
    count = grouped.transform("count")
    valid = valid.copy()
    valid["loo_member_count"] = count - 1
    valid["industry_return_loo"] = np.where(
        valid["loo_member_count"] > 0,
        (total - valid["return_1d"]) / valid["loo_member_count"],
        np.nan,
    )
    context = valid.groupby(["date", "industry_code"], observed=True).agg(
        member_count=("code", "nunique"),
        return_1d=("return_1d", "mean"),
        breadth=("return_1d", lambda s: float((s > 0).mean()) if len(s) else np.nan),
    ).reset_index()
    stock = valid[
        ["date", "code", "industry_code", "return_1d", "loo_member_count", "industry_return_loo"]
    ].copy()
    return context, stock


def prefix_loo_invariant(panel: pd.DataFrame, cut: pd.Timestamp) -> bool:
    cut = pd.Timestamp(cut).normalize()
    full_stock = industry_loo_context(panel)[1]
    prefix_panel = panel[pd.to_datetime(panel["date"]).dt.normalize() <= cut].copy()
    prefix_stock = industry_loo_context(prefix_panel)[1]
    left = (
        full_stock[full_stock["date"] <= cut][["date", "code", "industry_return_loo"]]
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )
    right = (
        prefix_stock[["date", "code", "industry_return_loo"]]
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )
    return left.equals(right)
