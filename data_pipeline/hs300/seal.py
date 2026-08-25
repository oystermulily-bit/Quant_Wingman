"""Physically isolate Holdout prices from the Development standardized layer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .raw_store import sha256_file


SEAL_DIR = "sealed/holdout"
HOLD_LOCK = "sealed/holdout.lock.json"


def holdout_index(dates: pd.Series, holdout_start: pd.Timestamp) -> pd.Series:
    parsed = pd.to_datetime(dates).dt.normalize()
    return parsed >= pd.Timestamp(holdout_start).normalize()


def split_development_and_holdout(
    frame: pd.DataFrame,
    holdout_start: pd.Timestamp,
    *,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if date_col not in frame.columns:
        return frame.copy(), frame.iloc[0:0].copy()
    mask = holdout_index(frame[date_col], holdout_start)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def write_sealed_holdout(
    root: Path,
    tables: dict[str, pd.DataFrame],
    *,
    holdout_start: str,
    holdout_date_count: int,
    holdout_date_hash: str,
    protocol: str,
) -> dict:
    folder = root / "sealed" / "holdout"
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for name, frame in tables.items():
        path = folder / f"{name}.parquet"
        if path.exists():
            try:
                path.chmod(0o644)
            except OSError:
                pass
        frame.to_parquet(path, index=False)
        files[name] = {
            "path": f"sealed/holdout/{name}.parquet",
            "sha256": sha256_file(path),
            "row_count": int(len(frame)),
            "rows": int(len(frame)),
        }
    lock = {
        "holdout_start": holdout_start,
        "holdout_date_count": holdout_date_count,
        "holdout_date_hash": holdout_date_hash,
        "protocol": protocol,
        "readable_metrics": False,
        "readable_prices": False,
        "unseal_phrase": "UNSEAL_HOLDOUT_PRICES",
        "files": files,
    }
    (root / "sealed" / "holdout.lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "README.txt").write_text(
        "Holdout raw prices are sealed. Standardized Development tables must not "
        "contain these dates. Do not compute strategy metrics from this folder.\n"
        "Read requires confirm='UNSEAL_HOLDOUT_PRICES' and is for data-quality "
        "audits only — never score Holdout.\n",
        encoding="utf-8",
    )
    protect_holdout_files(root)
    return lock


def protect_holdout_files(root: Path) -> None:
    """Mark sealed Holdout parquet files read-only after they are written."""
    import os
    import stat

    folder = root / "sealed" / "holdout"
    if not folder.is_dir():
        return
    for path in folder.glob("*.parquet"):
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP)


def assert_holdout_sealed(root: Path, holdout_start: pd.Timestamp) -> None:
    lock_path = root / "sealed" / "holdout.lock.json"
    if not lock_path.is_file():
        raise PermissionError("missing sealed/holdout.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("readable_prices") or lock.get("readable_metrics"):
        raise PermissionError("holdout lock allows reading prices or metrics")
    for name in ("daily_bars", "universe_membership", "trading_status"):
        path = root / "standardized" / f"{name}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["date"])
        if holdout_index(frame["date"], holdout_start).any():
            raise PermissionError(f"{name} still contains Holdout dates")


def load_sealed_holdout_prices(root: Path, *, confirm: str) -> dict[str, pd.DataFrame]:
    from .config import SEAL_CONFIRM_PHRASE

    if confirm != SEAL_CONFIRM_PHRASE:
        raise PermissionError("Holdout prices are sealed")
    lock = json.loads((root / "sealed" / "holdout.lock.json").read_text(encoding="utf-8"))
    if lock.get("readable_prices"):
        raise PermissionError("refuse to load: lock marked readable_prices")
    # Even with the confirm phrase, metrics remain forbidden. Prices are only
    # returned for data-quality audits that the caller must not score.
    out = {}
    for name, entry in (lock.get("files") or {}).items():
        out[name] = pd.read_parquet(root / entry["path"])
    return out
