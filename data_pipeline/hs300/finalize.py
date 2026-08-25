"""Finalize a frozen CSI 300 snapshot: TR OHLC, tensors, checksums, sample audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import (
    update_coverage_gap,
    update_manifest_coverage,
    write_checksums,
    write_holdout_dates,
    write_schema,
)
from .opentr import build_preclose_chain
from .raw_store import RawStore, sha256_file
from .sample_audit import run_sample_audit
from .standardize import _concat_parts, prepare_kline_with_preclose
from data_pipeline.hs300_panel import HS300PanelDataManager


def patch_high_low_tr(root: Path) -> dict:
    bars_path = root / "standardized" / "daily_bars.parquet"
    existing = pd.read_parquet(bars_path)
    if {"high_tr", "low_tr"}.issubset(existing.columns):
        quoted = existing["has_quote"].fillna(False).astype(bool)
        high_ok = pd.to_numeric(existing.loc[quoted, "high_tr"], errors="coerce").gt(0)
        if quoted.any() and bool(high_ok.any()):
            return {
                "skipped": True,
                "quoted_rows": int(quoted.sum()),
                "comparable_open_tr": None,
                "open_tr_mismatch": 0,
                "high_tr_quoted": int(high_ok.sum()),
                "daily_bars_sha256": sha256_file(bars_path),
                "daily_bars_rows": int(len(existing)),
            }
    store = RawStore(root)
    kline = _concat_parts(store, "daily_kline")
    status = _concat_parts(store, "stock_status")
    prepared = prepare_kline_with_preclose(kline, status)
    chain = build_preclose_chain(
        prepared[["date", "code", "open", "high", "low", "close", "preclose"]]
    )
    bars_path = root / "standardized" / "daily_bars.parquet"
    bars = pd.read_parquet(bars_path)
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    bars["code"] = bars["code"].astype(str).str.upper()
    recomputed = chain.rename(
        columns={
            "open_tr": "open_tr_recomputed",
            "high_tr": "high_tr_recomputed",
            "low_tr": "low_tr_recomputed",
            "close_tr": "close_tr_recomputed",
        }
    )[
        [
            "date",
            "code",
            "open_tr_recomputed",
            "high_tr_recomputed",
            "low_tr_recomputed",
            "close_tr_recomputed",
        ]
    ].copy()
    recomputed["code"] = recomputed["code"].astype(str).str.upper()
    merged = bars.merge(recomputed, on=["date", "code"], how="left")
    quoted = merged["has_quote"].fillna(False).astype(bool)
    stored_open = pd.to_numeric(merged["open_tr"], errors="coerce")
    new_open = pd.to_numeric(merged["open_tr_recomputed"], errors="coerce")
    comparable = quoted & stored_open.notna() & new_open.notna()
    mismatch = comparable & ~np.isclose(
        stored_open.to_numpy(dtype=float),
        new_open.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-8,
        equal_nan=False,
    )
    report = {
        "quoted_rows": int(quoted.sum()),
        "comparable_open_tr": int(comparable.sum()),
        "open_tr_mismatch": int(mismatch.sum()),
        "high_tr_quoted": int((quoted & pd.to_numeric(merged["high_tr_recomputed"], errors="coerce").gt(0)).sum()),
    }
    check_path = root / "validation" / "opentr_recompute_check.json"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if mismatch.any():
        raise ValueError(f"open_tr recompute mismatch on {int(mismatch.sum())} rows")
    merged["high_tr"] = np.where(
        quoted,
        pd.to_numeric(merged["high_tr_recomputed"], errors="coerce"),
        np.nan,
    )
    merged["low_tr"] = np.where(
        quoted,
        pd.to_numeric(merged["low_tr_recomputed"], errors="coerce"),
        np.nan,
    )
    drop = [
        "open_tr_recomputed",
        "high_tr_recomputed",
        "low_tr_recomputed",
        "close_tr_recomputed",
    ]
    out = merged.drop(columns=[column for column in drop if column in merged.columns])
    out.to_parquet(bars_path, index=False)
    report["daily_bars_sha256"] = sha256_file(bars_path)
    report["daily_bars_rows"] = int(len(out))
    return report


def finalize_snapshot(root: Path) -> dict:
    root = Path(root)
    tr_report = patch_high_low_tr(root)
    files_update = {
        "daily_bars": {
            "path": "standardized/daily_bars.parquet",
            "sha256": tr_report["daily_bars_sha256"],
            "rows": tr_report["daily_bars_rows"],
        }
    }
    exception_path = root / "standardized" / "membership_exceptions.parquet"
    if exception_path.is_file():
        files_update["membership_exceptions"] = {
            "path": "standardized/membership_exceptions.parquet",
            "sha256": sha256_file(exception_path),
            "rows": int(len(pd.read_parquet(exception_path))),
        }
    update_manifest_coverage(root, {}, files_update=files_update)

    gate = json.loads((root / "validation" / "data_gate_report.json").read_text(encoding="utf-8"))
    coverage = update_coverage_gap(
        root,
        {
            "kline_vendor_start": "2013-01-04",
            "status_preclose_start": gate.get("status_preclose_start"),
            "opentr_start": gate.get("opentr_start"),
            "quoted_start": gate.get("kline_start"),
            "holdout_start": gate.get("holdout_start"),
            "holdout_date_count": gate.get("holdout_date_count"),
        },
    )
    calendar = pd.read_parquet(root / "standardized" / "trading_calendar.parquet")
    split_info = write_holdout_dates(root, calendar)
    sample = run_sample_audit(root)
    write_schema(root)

    manager = HS300PanelDataManager(root, required_start="2010-01-01")
    manager.load(raise_on_gate_failure=True)
    tensor_meta = manager.save_tensors()
    labels = manager.load_labels(development_only=True)

    meta_files = sorted((root / "raw").rglob("*.meta.json"))
    provenance = {}
    if meta_files:
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        provenance = {
            "sdk_version": meta.get("sdk_version") or "1.1.9",
            "mcp_server_sha256": meta.get("mcp_server_sha256"),
        }
    update_manifest_coverage(
        root,
        {
            "kline_vendor_start": coverage.get("kline_vendor_start"),
            "status_start": coverage.get("status_preclose_start"),
            "opentr_start": coverage.get("opentr_start"),
            "quoted_start": coverage.get("quoted_start"),
            **provenance,
        },
        files_update=files_update,
    )
    checksum_path = write_checksums(root)
    summary = {
        "open_tr_mismatch": tr_report["open_tr_mismatch"],
        "high_tr_quoted": tr_report["high_tr_quoted"],
        "panel_gate": manager.report.status if manager.report else None,
        "tensor": tensor_meta,
        "development_label_rows": int(len(labels)),
        "sample_rows": sample["sample_rows"],
        "holdout": split_info,
        "checksums": str(checksum_path).replace("\\", "/"),
    }
    (root / "validation" / "finalize_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # checksums were written before finalize_report; refresh once.
    write_checksums(root)
    return summary
