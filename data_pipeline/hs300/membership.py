"""Daily CSI 300 membership from frozen index weights."""

from __future__ import annotations

import pandas as pd

from .config import INDEX_CODE
from .time_policy import available_at, to_trade_date

KNOWN_AT_SOURCE = "MISSING_NATIVE_INDEX_ANNOUNCEMENT"
MEMBERSHIP_SOURCE = "DAILY_INDEX_WEIGHT"
WEIGHT_SUM_ATOL = 0.25
EXPECTED_MEMBERS = 300


def _detect_unit(daily_sum: pd.Series) -> str:
    median = float(daily_sum.median())
    if 99.0 <= median <= 101.0:
        return "PERCENT"
    if 0.99 <= median <= 1.01:
        return "FRACTION"
    raise ValueError(
        f"index weight daily sum median={median} is neither percent nor fraction"
    )


def build_universe_membership(
    weights: pd.DataFrame,
    *,
    index_code: str = INDEX_CODE,
    expected_members: int = EXPECTED_MEMBERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"CON_CODE", "TRADE_DATE", "WEIGHT"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"index_weight missing {sorted(missing)}")
    frame = weights.copy()
    frame["date"] = to_trade_date(frame["TRADE_DATE"])
    frame["code"] = frame["CON_CODE"].astype(str).str.upper()
    frame["index_code"] = (
        frame["INDEX_CODE"].astype(str).str.upper()
        if "INDEX_CODE" in frame.columns
        else index_code
    )
    frame["weight_raw"] = pd.to_numeric(frame["WEIGHT"], errors="coerce")
    frame = frame.dropna(subset=["date", "code", "weight_raw"])
    frame = frame[frame["index_code"] == index_code.upper()]
    if frame.duplicated(["date", "index_code", "code"]).any():
        raise ValueError("index_weight has duplicate date+code")

    daily_sum = frame.groupby("date", observed=True)["weight_raw"].sum(min_count=1)
    unit = _detect_unit(daily_sum)
    frame["weight_pct"] = frame["weight_raw"] * (100.0 if unit == "FRACTION" else 1.0)

    members = frame[["date", "index_code", "code", "weight_pct"]].copy()
    members["is_member"] = True
    members["effective_at"] = members["date"].dt.tz_localize("Asia/Shanghai")
    members["known_at"] = pd.NaT
    members["known_at_source"] = KNOWN_AT_SOURCE
    members["membership_source"] = MEMBERSHIP_SOURCE
    members["available_at"] = available_at(members["date"])
    members["source_request_id"] = (
        members["date"].dt.strftime("%Y%m%d") + ":" + members["code"]
    )

    ordered = members.sort_values(["code", "date"], kind="stable")
    gap = ordered.groupby("code", sort=False)["date"].diff().dt.days.fillna(0)
    new_spell = gap.gt(10)
    spell_id = new_spell.groupby(ordered["code"]).cumsum()
    ordered["spell_id"] = spell_id.to_numpy()
    # Daily training rows may keep the already-observed spell start. The
    # realized last membership date is only known after the stock leaves, so
    # it stays in the audit table and is never copied onto prior days.
    snapshot_end = ordered["date"].max() if len(ordered) else pd.NaT
    spells = (
        ordered.groupby(["code", "spell_id"], sort=False)["date"]
        .agg(entry_effective_date="min", last_member_date="max")
        .reset_index()
    )
    spells["index_code"] = index_code
    still_open = spells["last_member_date"].eq(snapshot_end) if len(spells) else False
    spells["realized_exit_date"] = spells["last_member_date"]
    spells.loc[still_open, "realized_exit_date"] = pd.NaT
    spells["censored_at_snapshot_end"] = still_open
    spells = spells.drop(columns=["last_member_date"])
    members = ordered.merge(
        spells[["code", "spell_id", "entry_effective_date"]],
        on=["code", "spell_id"],
        how="left",
    )
    members = members.drop(columns=["spell_id"])
    spells = spells.sort_values(["code", "spell_id"], kind="stable").reset_index(drop=True)

    counts = members.groupby("date", observed=True)["code"].nunique()
    sums = members.groupby("date", observed=True)["weight_pct"].sum(min_count=1)
    exception_rows = []
    for date, count in counts.items():
        weight_sum = float(sums.loc[date]) if date in sums.index else float("nan")
        reasons = []
        if int(count) != expected_members:
            reasons.append(f"member_count={int(count)}")
        if not (abs(weight_sum - 100.0) <= WEIGHT_SUM_ATOL):
            reasons.append(f"weight_sum={weight_sum:.6f}")
        if reasons:
            exception_rows.append(
                {
                    "date": date,
                    "index_code": index_code,
                    "member_count": int(count),
                    "weight_sum": weight_sum,
                    "weight_unit_detected": unit,
                    "reason": ";".join(reasons),
                }
            )
    exceptions = pd.DataFrame(exception_rows)
    members = members.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    if "exit_effective_date" in members.columns:
        members = members.drop(columns=["exit_effective_date"])
    return members, exceptions, spells
