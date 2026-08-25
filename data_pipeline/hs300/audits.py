"""Temporal leakage and prefix-invariance audits. Never scores Holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .opentr import build_preclose_chain, prefix_matches
from .seal import holdout_index


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def audit_holdout_isolation(root: Path, holdout_start: pd.Timestamp) -> dict:
    issues: list[str] = []
    for name in ("daily_bars", "universe_membership", "trading_status"):
        path = root / "standardized" / f"{name}.parquet"
        frame = pd.read_parquet(path, columns=["date"])
        leaked = int(holdout_index(frame["date"], holdout_start).sum())
        if leaked:
            issues.append(f"{name}:{leaked}")
    labels = root / "labels" / "development_labels.parquet"
    if labels.is_file():
        frame = pd.read_parquet(labels, columns=["signal_date"])
        leaked = int(holdout_index(frame["signal_date"], holdout_start).sum())
        if leaked:
            issues.append(f"development_labels:{leaked}")
    lock = json.loads((root / "sealed" / "holdout.lock.json").read_text(encoding="utf-8"))
    if lock.get("readable_prices") or lock.get("readable_metrics"):
        issues.append("holdout_lock_readable")
    report = {
        "passed": not issues,
        "holdout_start": pd.Timestamp(holdout_start).date().isoformat(),
        "issues": issues,
    }
    write_json(root / "validation" / "temporal_leakage_report.json", report)
    return report


def audit_industry_known_at(panel: pd.DataFrame) -> dict:
    if panel is None or panel.empty or "industry_known_at" not in panel.columns:
        return {"passed": True, "future_industry_rows": 0}
    known = pd.to_datetime(panel["industry_known_at"], utc=True, errors="coerce")
    if getattr(known.dt, "tz", None) is not None:
        known_date = known.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    else:
        known_date = known.dt.normalize()
    dates = pd.to_datetime(panel["date"]).dt.normalize()
    mapped = panel["industry_code"].notna()
    future = mapped & known_date.notna() & (known_date > dates)
    report = {
        "passed": not bool(future.any()),
        "future_industry_rows": int(future.sum()),
    }
    return report


def audit_prefix_invariance(daily_bars: pd.DataFrame, *, symbols: int = 12) -> dict:
    quoted = daily_bars[daily_bars["has_quote"].fillna(False).astype(bool)].copy()
    needed = {"date", "code", "open_raw", "high_raw", "low_raw", "close_raw", "preclose"}
    if quoted.empty or not needed.issubset(quoted.columns):
        report = {"passed": False, "reason": "no quoted bars"}
        return report
    bars = quoted.rename(
        columns={
            "open_raw": "open",
            "high_raw": "high",
            "low_raw": "low",
            "close_raw": "close",
        }
    )
    chain = build_preclose_chain(
        bars[["date", "code", "open", "high", "low", "close", "preclose"]]
    )
    checked = 0
    failed = []
    for code, group in chain.sort_values("date").groupby("code"):
        if len(group) < 16:
            continue
        cut = group["date"].iloc[len(group) // 2]
        ok = prefix_matches(group, group.loc[group["date"] <= cut, "date"], str(code))
        checked += 1
        if not ok:
            failed.append(str(code))
        if checked >= symbols:
            break
    report = {
        "passed": checked > 0 and not failed,
        "symbols_checked": checked,
        "failed_symbols": failed,
    }
    return report


def write_prefix_report(root: Path, daily_bars: pd.DataFrame) -> dict:
    report = audit_prefix_invariance(daily_bars)
    write_json(root / "validation" / "prefix_invariance_report.json", report)
    return report
