"""Assemble Wingman-ready frozen tables from CSI 300 raw parquet parts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipeline.hs300_panel import (
    HS300PanelDataManager,
    SNAPSHOT_MANIFEST_VERSION,
    SnapshotValidationError,
)
from research_stage3.labels import LabelRegistry
from research_stage3.protocol import DateSplitProtocol

from .config import (
    INDEX_CODE,
    RAW_SNAPSHOT_ROOT,
    RESEARCH_START,
    REQUESTED_START,
    SCHEMA_VERSION,
    SNAPSHOT_ID,
)
from .corporate_actions import build_corporate_actions
from .industry import build_industry_membership, shenwan_level1_codes
from .membership import build_universe_membership
from .opentr import build_preclose_chain
from .raw_store import RawStore, sha256_file
from .time_policy import available_at, session_close, session_open, to_trade_date
from .tradability import build_trading_status
from .validate_snapshot import build_validation_report, write_validation_artifacts
from .artifacts import update_coverage_gap, write_checksums, write_holdout_dates, write_schema
from .audits import audit_holdout_isolation, audit_industry_known_at, write_prefix_report
from .features_wingman import compute_wingman_feature_tensors, save_feature_tensors
from .industry_stats import industry_loo_context
from .lineage import attach_lineage
from .seal import assert_holdout_sealed, split_development_and_holdout, write_sealed_holdout


def _concat_parts(store: RawStore, method: str) -> pd.DataFrame:
    folder = store.raw_root / method
    if not folder.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in sorted(folder.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _row_hash(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    payload = frame[columns].astype("string").fillna("").agg("|".join, axis=1)
    return payload.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def build_calendar(
    raw_calendar: pd.DataFrame,
    end_date: pd.Timestamp,
    *,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    dates = to_trade_date(raw_calendar["date"] if "date" in raw_calendar.columns else raw_calendar.iloc[:, 0])
    start = pd.Timestamp(start if start is not None else RESEARCH_START)
    keep = (dates >= start) & (dates <= end_date)
    dates = dates.loc[keep].drop_duplicates().sort_values()
    out = pd.DataFrame({"date": dates})
    out["market"] = "SH"
    out["is_trading_day"] = True
    out["session_open"] = session_open(out["date"])
    out["session_close"] = session_close(out["date"])
    out["is_complete_session"] = out["date"] <= end_date
    out["available_at"] = available_at(out["date"])
    return out.reset_index(drop=True)


def prepare_kline_with_preclose(
    kline: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    bars = kline.copy()
    rename = {
        "MARKET_CODE": "code",
        "code": "code",
        "kline_time": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
    }
    bars = bars.rename(columns={key: value for key, value in rename.items() if key in bars.columns})
    if "code" not in bars.columns and "MARKET_CODE" in kline.columns:
        bars["code"] = kline["MARKET_CODE"]
    bars["code"] = bars["code"].astype(str).str.upper()
    bars["date"] = to_trade_date(bars["date"] if "date" in bars.columns else bars.get("kline_time"))
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in bars.columns:
            bars[column] = pd.to_numeric(bars[column], errors="coerce")

    status_small = status.copy()
    if not status_small.empty:
        status_small["code"] = status_small.get("MARKET_CODE", status_small.get("code")).astype(str).str.upper()
        status_small["date"] = to_trade_date(status_small.get("TRADE_DATE", status_small.get("date")))
        status_small["preclose"] = pd.to_numeric(status_small.get("PRECLOSE"), errors="coerce")
        status_small = status_small[["date", "code", "preclose"]].drop_duplicates(["date", "code"])
        bars = bars.merge(status_small, on=["date", "code"], how="left")
    else:
        bars["preclose"] = pd.NA
    return bars


def assemble_quoted_bars(
    kline: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    """Chain OpenTR on the vendor kline. Not sliced to index membership."""
    bars = prepare_kline_with_preclose(kline, status)
    chain = build_preclose_chain(
        bars[["date", "code", "open", "high", "low", "close", "preclose"]]
    )
    bars = bars.merge(
        chain[
            [
                "date",
                "code",
                "open_tr",
                "high_tr",
                "low_tr",
                "close_tr",
                "chain_valid",
                "adjustment_type",
            ]
        ],
        on=["date", "code"],
        how="left",
    )
    quoted = (
        bars["open"].gt(0)
        & bars["high"].gt(0)
        & bars["low"].gt(0)
        & bars["close"].gt(0)
        & bars["volume"].ge(0)
        & bars["amount"].ge(0)
        & bars["chain_valid"].fillna(False)
    )
    bars["has_quote"] = quoted
    bars["quote_missing_reason"] = pd.Series(pd.NA, index=bars.index, dtype="string")
    bars.loc[~quoted, "quote_missing_reason"] = "MEMBER_WITH_MISSING_SOURCE_QUOTE"
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "preclose",
        "open_tr",
        "high_tr",
        "low_tr",
        "close_tr",
    ]
    bars.loc[~quoted, numeric_cols] = pd.NA
    return bars


def _finalize_daily_bar_frame(panel_bars: pd.DataFrame) -> pd.DataFrame:
    panel_bars = panel_bars.copy()
    panel_bars["available_at"] = available_at(panel_bars["date"])
    panel_bars["available_at_source"] = "DERIVED_POLICY"
    panel_bars["revision_id"] = "raw_v1"
    hashed = panel_bars.reindex(
        columns=[
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "preclose",
        ]
    )
    panel_bars["source_row_hash"] = _row_hash(hashed.fillna(""), list(hashed.columns))
    out = pd.DataFrame(
        {
            "date": panel_bars["date"],
            "code": panel_bars["code"],
            "open_raw": panel_bars["open"],
            "high_raw": panel_bars["high"],
            "low_raw": panel_bars["low"],
            "close_raw": panel_bars["close"],
            "volume": panel_bars["volume"],
            "amount": panel_bars["amount"],
            "preclose": panel_bars["preclose"],
            "open_tr": panel_bars["open_tr"],
            "high_tr": panel_bars["high_tr"],
            "low_tr": panel_bars["low_tr"],
            "close_tr": panel_bars["close_tr"],
            "adjustment_type": panel_bars["adjustment_type"],
            "available_at": panel_bars["available_at"],
            "available_at_source": panel_bars["available_at_source"],
            "has_quote": panel_bars["has_quote"].astype(bool),
            "quote_missing_reason": panel_bars["quote_missing_reason"],
            "revision_id": panel_bars["revision_id"],
            "source_row_hash": panel_bars["source_row_hash"],
        }
    )
    if out.duplicated(["date", "code"]).any():
        raise ValueError("daily_bars duplicate date+code")
    return out.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def slice_member_panel(quoted_bars: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    member_keys = members.loc[members["is_member"], ["date", "code"]].drop_duplicates()
    panel_bars = member_keys.merge(quoted_bars, on=["date", "code"], how="left")
    missing = panel_bars["has_quote"].isna()
    panel_bars.loc[missing, "has_quote"] = False
    panel_bars.loc[missing, "quote_missing_reason"] = panel_bars.loc[
        missing, "quote_missing_reason"
    ].fillna("UNAVAILABLE_FROM_VENDOR")
    return _finalize_daily_bar_frame(panel_bars)


def slice_execution_ledger(quoted_bars: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Keep quotes for every date of any stock that was ever an index member.

    The research panel stays membership-sliced. This ledger is what the
    backtest uses to flatten positions after a stock leaves CSI 300.
    """
    ever = set(members["code"].astype(str).str.upper().unique())
    bars = quoted_bars.copy()
    bars["code"] = bars["code"].astype(str).str.upper()
    panel_bars = bars[bars["code"].isin(ever)].copy()
    if panel_bars.empty:
        return _finalize_daily_bar_frame(panel_bars)
    missing = panel_bars["has_quote"].isna()
    panel_bars.loc[missing, "has_quote"] = False
    panel_bars.loc[missing, "quote_missing_reason"] = panel_bars.loc[
        missing, "quote_missing_reason"
    ].fillna("UNAVAILABLE_FROM_VENDOR")
    return _finalize_daily_bar_frame(panel_bars)


def build_daily_bars(
    kline: pd.DataFrame,
    status: pd.DataFrame,
    members: pd.DataFrame,
) -> pd.DataFrame:
    return slice_member_panel(assemble_quoted_bars(kline, status), members)


def _manifest_file(rel: str, path: Path) -> dict:
    import pyarrow.parquet as pq

    rows = int(pq.ParquetFile(path).metadata.num_rows)
    return {
        "path": rel,
        "sha256": sha256_file(path),
        "row_count": rows,
        "rows": rows,
    }


def _apply_lineage(frame: pd.DataFrame, *, data_version: str, source: str, fetched_at: str) -> pd.DataFrame:
    return attach_lineage(
        frame,
        schema_version=SCHEMA_VERSION,
        snapshot_id=SNAPSHOT_ID,
        data_version=data_version,
        source=source,
        fetched_at=fetched_at,
    )


def freeze_panel_snapshot(root: Path, *, raw_root: Path | None = None) -> dict:
    root = Path(root)
    store = RawStore(root, raw_root=raw_root)
    gap_path = root / "coverage_gap.json"
    if not gap_path.is_file():
        gap_path = RAW_SNAPSHOT_ROOT / "coverage_gap.json"
    raw_manifest = json.loads(gap_path.read_text(encoding="utf-8"))
    end_date = pd.to_datetime(str(raw_manifest["end_date"])).normalize()
    research_start = pd.Timestamp(RESEARCH_START)

    print(f"freeze {SNAPSHOT_ID}: loading raw parts from {store.raw_root}", flush=True)
    calendar = build_calendar(
        _concat_parts(store, "trading_calendar"), end_date, start=research_start
    )
    weights = _concat_parts(store, "index_weight")
    members, exceptions, membership_spells = build_universe_membership(weights)
    members = members[members["date"] >= research_start].reset_index(drop=True)
    kline = _concat_parts(store, "daily_kline")
    status_raw = _concat_parts(store, "stock_status")
    print("freeze: chaining OpenTR on full kline, then slicing to research start", flush=True)
    quoted_bars = assemble_quoted_bars(kline, status_raw)
    daily_bars = slice_member_panel(quoted_bars, members)
    execution_bars = slice_execution_ledger(quoted_bars, members)
    daily_bars = daily_bars[daily_bars["date"] >= research_start].reset_index(drop=True)
    execution_bars = execution_bars[execution_bars["date"] >= research_start].reset_index(
        drop=True
    )
    trading_status = build_trading_status(
        status_raw,
        quoted_bars[["date", "code", "open", "volume", "amount"]],
    )
    ever_codes = set(members["code"].astype(str).str.upper().unique())
    execution_status = trading_status[
        trading_status["code"].astype(str).str.upper().isin(ever_codes)
    ].copy()
    execution_status = execution_status[
        execution_status["date"] >= research_start
    ].reset_index(drop=True)
    member_status = members[["date", "code"]].merge(
        trading_status, on=["date", "code"], how="left"
    )
    missing_status = member_status["can_buy_open"].isna()
    if missing_status.any():
        member_status.loc[missing_status, "can_buy_open"] = False
        member_status.loc[missing_status, "can_sell_open"] = False
        member_status.loc[missing_status, "is_suspended"] = False
        member_status.loc[missing_status, "is_st"] = False
        member_status.loc[missing_status, "is_xr"] = False
        member_status.loc[missing_status, "is_wd"] = False
        member_status.loc[missing_status, "buy_block_reason"] = "STATUS_UNAVAILABLE_FROM_VENDOR"
        member_status.loc[missing_status, "sell_block_reason"] = "STATUS_UNAVAILABLE_FROM_VENDOR"
        member_status.loc[missing_status, "available_at"] = available_at(
            member_status.loc[missing_status, "date"]
        )
        member_status.loc[missing_status, "available_at_source"] = "DERIVED_POLICY"
    industry_base = _concat_parts(store, "industry_base")
    industry_const = _concat_parts(store, "industry_constituent")
    member_codes = sorted(members["code"].astype(str).unique().tolist())
    industries, industry_exceptions = build_industry_membership(industry_const, industry_base, member_codes)
    actions = build_corporate_actions(
        _concat_parts(store, "dividends"),
        _concat_parts(store, "right_issues"),
        _concat_parts(store, "equity_changes"),
    )
    basics = _concat_parts(store, "stock_basic")
    if basics.empty:
        master = pd.DataFrame(
            {
                "code": member_codes,
                "name": pd.NA,
                "exchange": [code.split(".")[-1] for code in member_codes],
                "list_date": pd.NaT,
                "delist_date": pd.NaT,
                "security_type": "A_SHARE",
                "board": pd.NA,
                "valid_from": pd.Timestamp(REQUESTED_START),
                "valid_to": pd.NaT,
                "known_at": pd.NaT,
            }
        )
    else:
        code_col = "MARKET_CODE" if "MARKET_CODE" in basics.columns else "code"
        master = pd.DataFrame(
            {
                "code": basics[code_col].astype(str).str.upper(),
                "name": basics["SECURITY_NAME"] if "SECURITY_NAME" in basics.columns else pd.NA,
                "exchange": basics[code_col].astype(str).str.upper().str.split(".").str[-1],
                "list_date": to_trade_date(basics["LISTDATE"]) if "LISTDATE" in basics.columns else pd.NaT,
                "delist_date": to_trade_date(basics["DELISTDATE"]) if "DELISTDATE" in basics.columns else pd.NaT,
                "security_type": "A_SHARE",
                "board": basics["BOARD"] if "BOARD" in basics.columns else pd.NA,
                "valid_from": to_trade_date(basics["LISTDATE"]) if "LISTDATE" in basics.columns else pd.Timestamp(REQUESTED_START),
                "valid_to": to_trade_date(basics["DELISTDATE"]) if "DELISTDATE" in basics.columns else pd.NaT,
                "known_at": pd.NaT,
            }
        ).drop_duplicates("code")

    if not exceptions.empty and "date" in exceptions.columns:
        exceptions = exceptions.loc[
            pd.to_datetime(exceptions["date"]).dt.normalize() >= research_start
        ].reset_index(drop=True)

    split_dates = calendar.loc[
        calendar["is_trading_day"] & calendar["is_complete_session"], "date"
    ]
    split_plan = DateSplitProtocol().build(split_dates)
    split_payload = split_plan.to_dict()
    holdout_start = pd.Timestamp(split_plan.holdout_start).normalize()
    print(f"freeze: sealing Holdout from {split_payload['holdout_start']} ({split_payload['holdout_date_count']} days)", flush=True)
    members, holdout_members = split_development_and_holdout(members, holdout_start)
    daily_bars, holdout_bars = split_development_and_holdout(daily_bars, holdout_start)
    member_status, holdout_status = split_development_and_holdout(
        member_status, holdout_start
    )
    execution_bars, holdout_execution_bars = split_development_and_holdout(
        execution_bars, holdout_start
    )
    execution_status, holdout_execution_status = split_development_and_holdout(
        execution_status, holdout_start
    )

    built_at = datetime.now(timezone.utc).isoformat()
    data_version = f"{SNAPSHOT_ID}:{end_date.date().isoformat()}"
    meta_files = sorted(path for path in store.raw_root.rglob("*.meta.json") if ".retry" not in path.name)
    provenance_meta = json.loads(meta_files[0].read_text(encoding="utf-8")) if meta_files else {}
    sdk_version = provenance_meta.get("sdk_version") or "1.1.9"
    mcp_sha = provenance_meta.get("mcp_server_sha256")
    source = f"AmazingData SDK {sdk_version}"
    lineage_kw = {
        "data_version": data_version,
        "source": source,
        "fetched_at": built_at,
    }

    members = _apply_lineage(members, **lineage_kw)
    daily_bars = _apply_lineage(daily_bars, **lineage_kw)
    member_status = _apply_lineage(member_status, **lineage_kw)
    calendar = _apply_lineage(calendar, **lineage_kw)
    industries = _apply_lineage(industries, **lineage_kw)
    master = _apply_lineage(master, **lineage_kw)
    actions = _apply_lineage(actions, **lineage_kw)
    industry_exceptions = _apply_lineage(industry_exceptions, **lineage_kw)
    exceptions = _apply_lineage(exceptions, **lineage_kw)
    holdout_members = _apply_lineage(holdout_members, **lineage_kw)
    holdout_bars = _apply_lineage(holdout_bars, **lineage_kw)
    holdout_status = _apply_lineage(holdout_status, **lineage_kw)
    execution_bars = _apply_lineage(execution_bars, **lineage_kw)
    execution_status = _apply_lineage(execution_status, **lineage_kw)
    holdout_execution_bars = _apply_lineage(holdout_execution_bars, **lineage_kw)
    holdout_execution_status = _apply_lineage(holdout_execution_status, **lineage_kw)
    membership_spells = _apply_lineage(membership_spells, **lineage_kw)

    sealed = write_sealed_holdout(
        root,
        {
            "daily_bars": holdout_bars,
            "universe_membership": holdout_members,
            "trading_status": holdout_status,
            "execution_bars": holdout_execution_bars,
            "execution_status": holdout_execution_status,
        },
        holdout_start=split_payload["holdout_start"],
        holdout_date_count=split_payload["holdout_date_count"],
        holdout_date_hash=split_payload["holdout_date_hash"],
        protocol=split_payload["protocol"],
    )

    std = root / "standardized"
    files: dict[str, dict] = {}
    mapping = {
        "trading_calendar": calendar,
        "universe_membership": members,
        "daily_bars": daily_bars,
        "trading_status": member_status,
        "industry_membership": industries,
        "security_master": master,
        "corporate_actions": actions,
        "industry_exceptions": industry_exceptions,
        "membership_exceptions": exceptions,
        "execution_bars": execution_bars,
        "execution_status": execution_status,
    }
    for name, frame in mapping.items():
        path = std / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        files[name] = _manifest_file(f"standardized/{name}.parquet", path)

    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    spell_path = audit_dir / "membership_spell_audit.parquet"
    membership_spells.to_parquet(spell_path, index=False)
    files["membership_spell_audit"] = _manifest_file(
        "audit/membership_spell_audit.parquet", spell_path
    )

    raw_requests = []
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rel = meta_path.with_suffix("").with_suffix(".parquet").relative_to(root)
        raw_requests.append(
            {
                "request_id": f"{meta.get('method')}/{meta.get('part_id')}",
                "snapshot_id": SNAPSHOT_ID,
                "mcp_server_sha256": meta.get("mcp_server_sha256"),
                "sdk_version": meta.get("sdk_version"),
                "method": meta.get("method"),
                "parameters": json.dumps(meta.get("parameters", {}), sort_keys=True),
                "requested_at": meta.get("requested_at"),
                "completed_at": meta.get("completed_at"),
                "response_path": str(rel).replace("\\", "/"),
                "response_sha256": meta.get("response_sha256"),
                "row_count": meta.get("row_count"),
                "success": meta.get("success"),
                "retry_count": meta.get("retry_count"),
                "error": meta.get("error"),
            }
        )
    raw_manifest_frame = _apply_lineage(pd.DataFrame(raw_requests), **lineage_kw)
    raw_path = std / "raw_request_manifest.parquet"
    raw_manifest_frame.to_parquet(raw_path, index=False)
    files["raw_request_manifest"] = _manifest_file(
        "standardized/raw_request_manifest.parquet", raw_path
    )

    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "split_plan.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (splits_dir / "holdout.lock.json").write_text(
        json.dumps(
            {
                "holdout_start": split_payload["holdout_start"],
                "holdout_date_count": split_payload["holdout_date_count"],
                "holdout_date_hash": split_payload["holdout_date_hash"],
                "protocol": split_payload["protocol"],
                "readable_metrics": False,
                "readable_prices": False,
                "prices_location": "sealed/holdout",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    split_info = write_holdout_dates(root, calendar)
    assert_holdout_sealed(root, holdout_start)

    manifest = {
        "manifest_version": SNAPSHOT_MANIFEST_VERSION,
        "snapshot_id": SNAPSHOT_ID,
        "schema_version": SCHEMA_VERSION,
        "data_version": data_version,
        "built_at": built_at,
        "frozen": True,
        "sources": [source],
        "index_code": INDEX_CODE,
        "requested_start": REQUESTED_START.isoformat(),
        "research_start": RESEARCH_START.isoformat(),
        "end_date": int(end_date.strftime("%Y%m%d")),
        "sdk_version": sdk_version,
        "mcp_server_sha256": mcp_sha,
        "files": files,
        "sealed_files": sealed.get("files", {}),
        "membership_exception_count": int(len(exceptions)),
        "research_status": "DATA_NOT_FORMALLY_VALIDATED",
        "shenwan_l1_count": int(len(shenwan_level1_codes(industry_base))) if not industry_base.empty else 0,
        "kline_vendor_start": raw_manifest.get("kline_vendor_start", "2013-01-04"),
        "opentr_start": raw_manifest.get("opentr_start", RESEARCH_START.isoformat()),
    }
    coverage = build_validation_report(
        calendar=calendar,
        members=members,
        exceptions=exceptions,
        daily_bars=daily_bars,
        trading_status=member_status,
        industries=industries,
    )
    coverage["holdout_start"] = split_payload["holdout_start"]
    coverage["holdout_date_count"] = split_payload["holdout_date_count"]
    coverage["holdout_metrics_readable"] = False
    coverage["holdout_prices_readable"] = False
    manifest["validation"] = coverage["hard_checks"]
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("freeze: running formal Data Gate", flush=True)
    manager = HS300PanelDataManager(
        root, required_start=RESEARCH_START.isoformat()
    )
    manager.load(raise_on_gate_failure=False)
    if manager.report is None:
        raise SnapshotValidationError("formal Data Gate did not produce a report")
    gate_payload = manager.report.to_dict()
    (root / "validation").mkdir(parents=True, exist_ok=True)
    (root / "validation" / "panel_gate.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "validation" / "data_gate_report.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "validation" / "coverage_report.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if not manager.report.passed or manager.panel is None:
        (root / "STATUS.txt").write_text(
            "RAW_SNAPSHOT_COMPLETE\nSTANDARDIZED_FROZEN\nDATA_GATE_FAILED\n",
            encoding="utf-8",
        )
        raise SnapshotValidationError(
            json.dumps(gate_payload, ensure_ascii=False, default=str)
        )

    development = manager.development_panel(split_plan.development_dates)
    context, stock_loo = industry_loo_context(development)
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    context.to_parquet(derived / "industry_context.parquet", index=False)
    stock_loo.to_parquet(derived / "industry_loo_stock.parquet", index=False)
    labels = LabelRegistry().build(
        development,
        manager.bars_for_dates(split_plan.development_dates),
        pd.DatetimeIndex(split_plan.development_dates),
        manager.trading_status[
            manager.trading_status["date"].isin(pd.DatetimeIndex(split_plan.development_dates))
        ]
        if manager.trading_status is not None
        else None,
    )
    labels = _apply_lineage(labels, **lineage_kw)
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_dir / "development_labels.parquet", index=False)
    (labels_dir / "README.txt").write_text(
        "Development labels only. Holdout labels are not written.\n",
        encoding="utf-8",
    )
    files["development_labels"] = _manifest_file(
        "labels/development_labels.parquet", labels_dir / "development_labels.parquet"
    )

    leakage = audit_holdout_isolation(root, holdout_start)
    prefix = write_prefix_report(root, daily_bars)
    industry_known = audit_industry_known_at(manager.panel)
    (root / "validation" / "industry_known_at_report.json").write_text(
        json.dumps(industry_known, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if not leakage["passed"]:
        raise SnapshotValidationError(f"holdout isolation failed: {leakage['issues']}")
    if not prefix["passed"]:
        raise SnapshotValidationError(f"prefix invariance failed: {prefix}")
    if not industry_known["passed"]:
        raise SnapshotValidationError(
            f"industry known_at leak: {industry_known['future_industry_rows']} rows"
        )

    from model_core.vocab import FEATURE_NAMES, VOCAB_VERSION
    from .panel_tensors import save_panel_tensors

    print("freeze: computing Wingman 65-d feature tensors", flush=True)
    tensors = manager.to_tensors()
    features, feature_valid, feature_names = compute_wingman_feature_tensors(tensors)
    if tuple(feature_names) != tuple(FEATURE_NAMES):
        raise ValueError("feature names drifted from the live Wingman vocab")
    if int(features.shape[1]) != len(FEATURE_NAMES) or int(features.shape[1]) == 0:
        raise ValueError(f"expected {len(FEATURE_NAMES)} Wingman features, got {features.shape}")
    feature_meta = save_feature_tensors(
        root / "panel" / "hs300_features.npz",
        features=features,
        valid=feature_valid,
        names=feature_names,
        vocab_version=VOCAB_VERSION,
    )
    packed = replace(tensors, features=features)
    tensor_meta = save_panel_tensors(packed, root / "panel" / "hs300_tensors.npz")
    (root / "panel" / "tensor_manifest.json").write_text(
        json.dumps(
            tensor_meta
            | {
                "expected_members": 300,
                "features": feature_meta,
                "vocab_version": VOCAB_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    research_status = "DATA_READY_FOR_DEVELOPMENT"
    coverage["research_status"] = research_status
    coverage["panel_gate"] = manager.report.status
    write_validation_artifacts(root, coverage)
    (root / "validation" / "data_gate_report.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "validation" / "coverage_report.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest.update(
        {
            "research_status": research_status,
            "research_start": RESEARCH_START.isoformat(),
            "quoted_start": coverage.get("kline_start"),
            "status_start": coverage.get("status_preclose_start"),
            "opentr_start": coverage.get("opentr_start") or RESEARCH_START.isoformat(),
            "holdout_start": split_info["holdout_start"],
            "holdout_date_count": split_info["holdout_date_count"],
            "holdout_date_hash": split_info["holdout_date_hash"],
            "tensor": tensor_meta,
            "features": feature_meta,
            "files": files,
        }
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    update_coverage_gap(
        root,
        {
            "end_date": raw_manifest.get("end_date"),
            "snapshot_id": SNAPSHOT_ID,
            "research_start": RESEARCH_START.isoformat(),
            "requested_start": REQUESTED_START.isoformat(),
            "kline_vendor_start": manifest["kline_vendor_start"],
            "opentr_start": manifest["opentr_start"],
            "quoted_start": coverage.get("kline_start"),
            "holdout_start": split_info["holdout_start"],
            "holdout_date_count": split_info["holdout_date_count"],
            "holdout_prices_sealed": True,
            "industry_coverage": manager.report.metadata.get("industry_coverage"),
            "panel_gate": manager.report.status,
        },
    )
    write_schema(root)
    (root / "STATUS.txt").write_text(
        "RAW_SNAPSHOT_COMPLETE\n"
        "STANDARDIZED_FROZEN\n"
        "DATA_GATE_PASSED\n"
        f"RESEARCH_START={RESEARCH_START.isoformat()}\n",
        encoding="utf-8",
    )
    write_checksums(root)
    return manifest
