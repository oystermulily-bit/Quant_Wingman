"""Snapshot sidecar artifacts: checksums, schema, coverage, holdout dates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .raw_store import sha256_file
from research_stage3.protocol import DateSplitProtocol


SCHEMA_DOCUMENT = {
    "schema_version": "hs300_pit_v2",
    "index_code": "000300.SH",
    "timezone": "Asia/Shanghai",
    "tables": {
        "trading_calendar": {"primary_key": ["date"]},
        "universe_membership": {"primary_key": ["date", "index_code", "code"]},
        "daily_bars": {
            "primary_key": ["date", "code", "revision_id"],
            "price_policy": "unadjusted OHLCV; OpenTR/HighTR/LowTR/CloseTR from PRECLOSE chain",
        },
        "trading_status": {"primary_key": ["date", "code"]},
        "industry_membership": {"primary_key": ["code", "industry_system", "level", "valid_from"]},
        "security_master": {"primary_key": ["code", "valid_from"]},
        "corporate_actions": {"primary_key": ["action_id"]},
        "raw_request_manifest": {"primary_key": ["request_id"]},
    },
    "tensors": {
        "dates": {"shape": ["T"], "dtype": "datetime64[D]"},
        "symbols": {"shape": ["N"], "note": "historical CSI 300 CON_CODE union, sorted"},
        "raw_ohlcv": {"shape": ["N", 5, "T"], "layout": "open,high,low,close,volume", "missing": "NaN"},
        "amount": {"shape": ["N", "T"], "missing": "NaN"},
        "causal_prices": {"shape": ["N", 4, "T"], "layout": "open_tr,high_tr,low_tr,close_tr", "missing": "NaN"},
        "membership_mask": {"shape": ["N", "T"], "daily_true_count": 300},
        "quote_valid_mask": {"shape": ["N", "T"]},
        "buyable_mask": {"shape": ["N", "T"]},
        "sellable_mask": {"shape": ["N", "T"]},
        "industry_id": {"shape": ["N", "T"], "unmapped": -1},
        "features": {
            "shape": ["N", 65, "T"],
            "note": "Wingman FEATURE_NAMES order; stored NaN; adapter may zero-fill at read",
            "file": "panel/hs300_features.npz",
        },
    },
    "labels": {
        "directory": "labels/",
        "development_only": True,
        "holdout_labels": "not written",
    },
}


def write_schema(root: Path) -> Path:
    path = root / "schema.json"
    path.write_text(json.dumps(SCHEMA_DOCUMENT, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_holdout_dates(root: Path, calendar: pd.DataFrame) -> dict[str, Any]:
    trading = calendar.loc[
        calendar["is_trading_day"].fillna(False).astype(bool)
        & calendar["is_complete_session"].fillna(False).astype(bool),
        "date",
    ]
    plan = DateSplitProtocol().build(trading)
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    all_dates = pd.DatetimeIndex(pd.to_datetime(trading)).normalize().drop_duplicates().sort_values()
    holdout_dates = all_dates[-plan.holdout_date_count :]
    serial = "\n".join(date.date().isoformat() for date in holdout_dates)
    digest = hashlib.sha256(serial.encode("ascii")).hexdigest()
    if digest != plan.holdout_date_hash:
        raise ValueError("holdout date hash mismatch while writing holdout_dates.parquet")
    lock_path = splits_dir / "holdout.lock.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("holdout_date_hash") not in (None, digest):
            raise ValueError("existing HOLDOUT.lock hash does not match calendar")
        if lock.get("readable_metrics") is True:
            raise PermissionError("holdout metrics were marked readable; refusing to rewrite dates")

    frame = pd.DataFrame({"date": holdout_dates, "role": "HOLDOUT"})
    out_path = splits_dir / "holdout_dates.parquet"
    frame.to_parquet(out_path, index=False)

    roles: list[dict[str, Any]] = []
    for date in plan.development_dates:
        roles.append({"split_protocol": plan.protocol, "fold_id": pd.NA, "role": "DEVELOPMENT", "date": date})
    for fold in plan.folds:
        for role_name, values in (
            ("TRAIN", fold.train_dates),
            ("PURGE", fold.purge_dates),
            ("OOF_VALIDATION", fold.validation_dates),
            ("EMBARGO", fold.embargo_dates),
        ):
            for date in values:
                roles.append(
                    {
                        "split_protocol": plan.protocol,
                        "fold_id": fold.fold_id,
                        "role": role_name,
                        "date": date,
                    }
                )
    for date in holdout_dates:
        roles.append(
            {
                "split_protocol": plan.protocol,
                "fold_id": pd.NA,
                "role": "HOLDOUT",
                "date": date,
            }
        )
    pd.DataFrame(roles).to_parquet(splits_dir / "date_roles.parquet", index=False)
    return {
        "holdout_start": plan.holdout_start.date().isoformat(),
        "holdout_date_count": int(plan.holdout_date_count),
        "holdout_date_hash": digest,
        "development_date_count": len(plan.development_dates),
    }


def update_coverage_gap(root: Path, extra: dict[str, Any]) -> dict[str, Any]:
    path = root / "coverage_gap.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def update_manifest_coverage(root: Path, extra: dict[str, Any], *, files_update: dict[str, dict] | None = None) -> dict[str, Any]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(extra)
    if files_update:
        files = manifest.setdefault("files", {})
        files.update(files_update)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def iter_checksum_paths(root: Path) -> Iterable[Path]:
    named = [
        root / "manifest.json",
        root / "schema.json",
        root / "coverage_gap.json",
        root / "STATUS.txt",
    ]
    for path in named:
        if path.is_file():
            yield path
    for folder in (
        root / "standardized",
        root / "splits",
        root / "labels",
        root / "derived",
        root / "validation",
        root / "panel",
        root / "sealed",
    ):
        if not folder.exists():
            continue
        yield from sorted(path for path in folder.rglob("*") if path.is_file())


def write_checksums(root: Path) -> Path:
    path = root / "checksums.sha256"
    lines: list[str] = []
    for file_path in iter_checksum_paths(root):
        if file_path.name == "checksums.sha256":
            continue
        rel = file_path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(file_path)}  {rel}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
