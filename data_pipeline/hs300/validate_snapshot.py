"""Snapshot validation reports. Never computes Holdout strategy metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .opentr import build_preclose_chain, prefix_matches


def _min_valid_date(frame: pd.DataFrame, value_col: str) -> str | None:
    if frame.empty or value_col not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[value_col], errors="coerce")
    keep = frame.loc[numeric.gt(0), "date"] if "date" in frame.columns else pd.Series(dtype="datetime64[ns]")
    if keep.empty:
        return None
    return pd.to_datetime(keep).min().date().isoformat()


def build_validation_report(
    *,
    calendar: pd.DataFrame,
    members: pd.DataFrame,
    exceptions: pd.DataFrame,
    daily_bars: pd.DataFrame,
    trading_status: pd.DataFrame,
    industries: pd.DataFrame,
) -> dict:
    trading_days = calendar.loc[
        calendar["is_trading_day"].fillna(False) & calendar["is_complete_session"].fillna(False)
    ]
    counts = members.groupby("date")["code"].nunique()
    quoted = daily_bars[daily_bars["has_quote"].fillna(False).astype(bool)]
    status_ready = trading_status[pd.to_numeric(trading_status.get("preclose"), errors="coerce").gt(0)]
    report = {
        "research_status": "DATA_NOT_FORMALLY_VALIDATED",
        "trading_days": int(len(trading_days)),
        "first_trading_day": trading_days["date"].min().date().isoformat() if len(trading_days) else None,
        "last_trading_day": trading_days["date"].max().date().isoformat() if len(trading_days) else None,
        "membership_exception_days": int(len(exceptions)),
        "days_not_exactly_300": int((counts != 300).sum()) if not counts.empty else None,
        "kline_start": _min_valid_date(quoted.rename(columns={"open_raw": "open"}), "open"),
        "status_preclose_start": _min_valid_date(status_ready, "preclose"),
        "industry_interval_rows": int(len(industries)),
        "quoted_rows": int(len(quoted)),
        "member_rows": int(len(members)),
        "hard_checks": {},
    }
    report["hard_checks"]["daily_member_count_300"] = report["days_not_exactly_300"] == 0
    report["hard_checks"]["no_membership_exceptions"] = report["membership_exception_days"] == 0
    report["opentr_start"] = max(
        value
        for value in (report["kline_start"], report["status_preclose_start"])
        if value is not None
    ) if report["kline_start"] and report["status_preclose_start"] else report["kline_start"]
    sample = quoted.sort_values(["code", "date"]).groupby("code").head(40)
    prefix_ok = True
    checked = 0
    if not sample.empty and {"open_raw", "preclose"}.issubset(sample.columns):
        bars = sample.rename(
            columns={
                "open_raw": "open",
                "high_raw": "high",
                "low_raw": "low",
                "close_raw": "close",
            }
        )
        needed = {"date", "code", "open", "high", "low", "close", "preclose"}
        if needed.issubset(bars.columns):
            chained = build_preclose_chain(bars[list(needed)])
            for code, group in chained.groupby("code"):
                if len(group) < 6:
                    continue
                cut = group["date"].iloc[len(group) // 2]
                prefix_ok = prefix_ok and prefix_matches(group, group.loc[group["date"] <= cut, "date"], code)
                checked += 1
                if checked >= 8:
                    break
    report["hard_checks"]["opentr_prefix_invariant_sample"] = prefix_ok if checked else None
    report["opentr_prefix_symbols_checked"] = checked
    return report


def write_validation_artifacts(root: Path, report: dict) -> None:
    folder = root / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "<html><head><meta charset='utf-8'><title>CSI 300 snapshot validation</title></head><body>",
        "<h1>CSI 300 snapshot validation</h1>",
        f"<p>status: {report.get('research_status')}</p>",
        "<ul>",
    ]
    for key, value in report.items():
        if key == "hard_checks":
            continue
        lines.append(f"<li>{key}: {value}</li>")
    lines.append("</ul><h2>hard checks</h2><ul>")
    for key, value in (report.get("hard_checks") or {}).items():
        lines.append(f"<li>{key}: {value}</li>")
    lines.append("</ul></body></html>")
    (folder / "report.html").write_text("\n".join(lines), encoding="utf-8")
