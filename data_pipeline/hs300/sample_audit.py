"""Yearly sample audit. Does not compute strategy metrics or read Holdout PnL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLE_SEED = 20260825
DAYS_PER_YEAR = 5
NAMES_PER_DAY = 10


def _load(root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / "standardized" / f"{name}.parquet")


def run_sample_audit(root: Path, *, seed: int = SAMPLE_SEED) -> dict[str, Any]:
    calendar = _load(root, "trading_calendar")
    members = _load(root, "universe_membership")
    bars = _load(root, "daily_bars")
    status = _load(root, "trading_status")
    industries = _load(root, "industry_membership")
    master = _load(root, "security_master")
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
    members["date"] = pd.to_datetime(members["date"]).dt.normalize()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    status["date"] = pd.to_datetime(status["date"]).dt.normalize()
    trading = calendar.loc[
        calendar["is_trading_day"].fillna(False) & calendar["is_complete_session"].fillna(False),
        "date",
    ]
    trading = pd.DatetimeIndex(trading).sort_values()
    rng = np.random.default_rng(seed)

    member_sets = members.groupby("date")["code"].agg(lambda s: frozenset(s.astype(str)))
    reconstitution = []
    previous = None
    for date, codes in member_sets.items():
        if previous is not None and codes != previous:
            reconstitution.append(pd.Timestamp(date))
        previous = codes

    forced_pool = {
        "reconstitution": reconstitution,
        "xr": pd.to_datetime(status.loc[status["is_xr"].fillna(False), "date"]).tolist(),
        "wd": pd.to_datetime(status.loc[status["is_wd"].fillna(False), "date"]).tolist(),
        "suspended": pd.to_datetime(status.loc[status["is_suspended"].fillna(False), "date"]).tolist(),
        "limit_up": pd.to_datetime(
            status.loc[status["buy_block_reason"].astype("string").eq("LIMIT_UP_OPEN"), "date"]
        ).tolist(),
        "industry_change": pd.to_datetime(industries["valid_from"]).dropna().tolist(),
        "delist": pd.to_datetime(master["delist_date"]).dropna().tolist(),
    }

    samples: list[dict[str, Any]] = []
    coverage = {key: 0 for key in forced_pool}
    years = sorted({int(date.year) for date in trading})
    for year in years:
        year_days = trading[trading.year == year]
        if year_days.empty:
            continue
        forced_dates: list[pd.Timestamp] = []
        for key, values in forced_pool.items():
            in_year = [pd.Timestamp(value).normalize() for value in values if pd.Timestamp(value).year == year]
            in_year = [date for date in in_year if date in year_days]
            if in_year:
                forced_dates.append(in_year[0])
                coverage[key] += 1
        remaining = [date for date in year_days if date not in forced_dates]
        need = max(0, DAYS_PER_YEAR - len(forced_dates))
        if remaining and need:
            pick_idx = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
            forced_dates.extend(remaining[i] for i in np.atleast_1d(pick_idx))
        chosen_days = sorted(set(forced_dates))[:DAYS_PER_YEAR]
        for date in chosen_days:
            day_members = members.loc[members["date"] == date, "code"].astype(str).tolist()
            if not day_members:
                continue
            n_pick = min(NAMES_PER_DAY, len(day_members))
            picked = [day_members[i] for i in rng.choice(len(day_members), size=n_pick, replace=False)]
            for code in picked:
                bar = bars[(bars["date"] == date) & (bars["code"].astype(str) == code)]
                st = status[(status["date"] == date) & (status["code"].astype(str) == code)]
                samples.append(
                    {
                        "date": date.date().isoformat(),
                        "code": code,
                        "has_quote": bool(bar["has_quote"].iloc[0]) if len(bar) else False,
                        "quote_missing_reason": (
                            None if bar.empty else (None if pd.isna(bar["quote_missing_reason"].iloc[0]) else str(bar["quote_missing_reason"].iloc[0]))
                        ),
                        "is_xr": bool(st["is_xr"].iloc[0]) if len(st) else None,
                        "is_wd": bool(st["is_wd"].iloc[0]) if len(st) else None,
                        "is_suspended": bool(st["is_suspended"].iloc[0]) if len(st) else None,
                        "buy_block_reason": (
                            None if st.empty or pd.isna(st["buy_block_reason"].iloc[0]) else str(st["buy_block_reason"].iloc[0])
                        ),
                        "can_buy_open": bool(st["can_buy_open"].iloc[0]) if len(st) else None,
                        "can_sell_open": bool(st["can_sell_open"].iloc[0]) if len(st) else None,
                    }
                )

    report = {
        "seed": seed,
        "days_per_year": DAYS_PER_YEAR,
        "names_per_day": NAMES_PER_DAY,
        "years": years,
        "forced_event_years_covered": coverage,
        "sample_rows": len(samples),
        "samples": samples,
        "holdout_metrics_read": False,
    }
    out = root / "validation" / "sample_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
