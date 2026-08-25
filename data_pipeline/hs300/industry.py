"""Shenwan level-1 membership intervals for historical CSI 300 names."""

from __future__ import annotations

import pandas as pd

from .time_policy import available_at, to_trade_date


def shenwan_level1_codes(
    industry_base: pd.DataFrame, *, expected_count: int | None = 31
) -> pd.DataFrame:
    frame = industry_base.copy()
    if "LEVEL_TYPE" in frame.columns:
        frame = frame[pd.to_numeric(frame["LEVEL_TYPE"], errors="coerce") == 1]
    if "INDEX_CODE" in frame.columns:
        frame = frame[frame["INDEX_CODE"].astype(str).str.upper().str.endswith(".SI")]
    if frame.empty:
        raise ValueError("no Shenwan level-1 industry codes")
    if expected_count is not None and len(frame) != expected_count:
        raise ValueError(f"expected {expected_count} Shenwan L1 codes, got {len(frame)}")
    return frame.reset_index(drop=True)


def _keep_primary_conflict(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conflict = frame.duplicated(["code", "valid_from"], keep=False)
    if not conflict.any():
        return frame, frame.iloc[0:0].copy()
    ranked = frame.loc[conflict].copy()
    ranked["_open"] = ranked["valid_to"].isna()
    ranked = ranked.sort_values(
        ["code", "valid_from", "_open", "valid_to", "industry_code"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    keep_idx = ranked.groupby(["code", "valid_from"], sort=False).head(1).index
    kept = pd.concat(
        [frame.loc[~conflict], frame.loc[keep_idx]],
        ignore_index=False,
    )
    dropped = frame.loc[conflict & ~frame.index.isin(keep_idx)].copy()
    dropped["reason"] = "CONFLICTING_L1_DROPPED_SECONDARY"
    kept = kept.drop(columns=["_open"], errors="ignore")
    dropped = dropped.drop(columns=["_open"], errors="ignore")
    return kept, dropped


def _close_gaps_until_next(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the last known L1 until the day before the next INDATE.

    This does not invent a classification before the first INDATE. It only
    removes vendor holes between consecutive intervals.
    """
    if frame.empty:
        return frame
    parts: list[pd.DataFrame] = []
    for _, hist in frame.groupby("code", sort=False):
        hist = hist.sort_values("valid_from", kind="stable").copy()
        nxt = hist["valid_from"].shift(-1)
        has_next = nxt.notna()
        if has_next.any():
            hist.loc[has_next, "valid_to"] = nxt[has_next] - pd.Timedelta(days=1)
        parts.append(hist)
    return pd.concat(parts, ignore_index=True)


def build_industry_membership(
    constituents: pd.DataFrame,
    industry_base: pd.DataFrame,
    member_codes: list[str],
    *,
    expected_l1_count: int | None = 31,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    level1 = shenwan_level1_codes(industry_base, expected_count=expected_l1_count)
    allowed = set(level1["INDEX_CODE"].astype(str).str.upper())
    names = {}
    if "LEVEL1_NAME" in level1.columns:
        names = dict(
            zip(
                level1["INDEX_CODE"].astype(str).str.upper(),
                level1["LEVEL1_NAME"].astype(str),
            )
        )
    frame = constituents.copy()
    index_col = "INDEX_CODE" if "INDEX_CODE" in frame.columns else "industry_code"
    code_col = "CON_CODE" if "CON_CODE" in frame.columns else "code"
    start_col = "INDATE" if "INDATE" in frame.columns else "valid_from"
    end_col = "OUTDATE" if "OUTDATE" in frame.columns else "valid_to"
    frame["industry_code"] = frame[index_col].astype(str).str.upper()
    frame["code"] = frame[code_col].astype(str).str.upper()
    frame = frame[frame["industry_code"].isin(allowed)]
    wanted = {str(code).upper() for code in member_codes}
    frame = frame[frame["code"].isin(wanted)]
    frame["valid_from"] = to_trade_date(frame[start_col])
    frame["valid_to"] = to_trade_date(frame[end_col])
    frame["industry_name"] = frame["industry_code"].map(names)
    frame["industry_system"] = "SHENWAN"
    frame["level"] = 1
    frame["effective_at"] = frame["valid_from"].dt.tz_localize("Asia/Shanghai")
    frame["known_at"] = available_at(frame["valid_from"])
    frame["known_at_source"] = "DERIVED_POLICY"
    columns = [
        "code",
        "industry_system",
        "level",
        "industry_code",
        "industry_name",
        "valid_from",
        "valid_to",
        "effective_at",
        "known_at",
        "known_at_source",
    ]
    out = frame[columns].drop_duplicates(["code", "valid_from", "industry_code"])
    out, dropped = _keep_primary_conflict(out)
    out = _close_gaps_until_next(out)
    out = out.sort_values(["code", "valid_from"], kind="stable").reset_index(drop=True)
    missing = sorted(wanted - set(out["code"].astype(str)))
    missing_frame = pd.DataFrame(
        {
            "code": missing,
            "reason": "NO_L1_HISTORY",
        }
    )
    exceptions = pd.concat([dropped, missing_frame], ignore_index=True)
    if not dropped.empty and "reason" not in dropped.columns:
        exceptions["reason"] = exceptions["reason"].fillna("CONFLICTING_L1_DROPPED_SECONDARY")
    return out, exceptions.reset_index(drop=True)
