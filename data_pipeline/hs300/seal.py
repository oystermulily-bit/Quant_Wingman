"""Isolate Holdout prices from the Development standardized layer.

This is a process convention plus a one-time capability token, read-only
files, and an append-only access log. It is not a separate OS user, HSM, or
encrypted vault. Callers must not describe it as physical isolation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .raw_store import sha256_file


SEAL_DIR = "sealed/holdout"
HOLD_LOCK = "sealed/holdout.lock.json"
CAPABILITY_NAME = "holdout.capability"
ACCESS_LOG_NAME = "holdout.access.log"


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


def _try_chmod_readonly(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP)
    except OSError:
        pass


def _try_chmod_user_write(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IRGRP)
    except OSError:
        pass


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _append_access_log(root: Path, event: str, **fields: object) -> None:
    folder = root / "sealed"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ACCESS_LOG_NAME
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


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
            _try_chmod_user_write(path)
        frame.to_parquet(path, index=False)
        files[name] = {
            "path": f"sealed/holdout/{name}.parquet",
            "sha256": sha256_file(path),
            "row_count": int(len(frame)),
            "rows": int(len(frame)),
        }
    token = secrets.token_urlsafe(32)
    digest = _token_digest(token)
    capability_path = root / "sealed" / CAPABILITY_NAME
    used_path = root / "sealed" / f"{CAPABILITY_NAME}.used"
    if used_path.exists():
        _try_chmod_user_write(used_path)
        used_path.unlink()
    if capability_path.exists():
        _try_chmod_user_write(capability_path)
    capability_path.write_text(token + "\n", encoding="utf-8")
    lock = {
        "holdout_start": holdout_start,
        "holdout_date_count": holdout_date_count,
        "holdout_date_hash": holdout_date_hash,
        "protocol": protocol,
        "readable_metrics": False,
        "readable_prices": False,
        "isolation": "one_shot_capability_readonly_files_access_log",
        "capability_hash": digest,
        "capability_one_shot": True,
        "capability_path": f"sealed/{CAPABILITY_NAME}",
        "access_log": f"sealed/{ACCESS_LOG_NAME}",
        "files": files,
    }
    (root / "sealed" / "holdout.lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "README.txt").write_text(
        "Holdout raw prices are sealed. Standardized Development tables must not "
        "contain these dates. Do not compute strategy metrics from this folder.\n"
        "Read requires the one-shot capability token in sealed/holdout.capability "
        "and is for data-quality audits only — never score Holdout.\n"
        "This seal is a workflow control, not a separate-permission vault.\n",
        encoding="utf-8",
    )
    _append_access_log(root, "sealed", holdout_start=holdout_start)
    protect_holdout_files(root)
    return lock


def protect_holdout_files(root: Path) -> None:
    """Mark sealed Holdout parquet files and the unused capability read-only."""
    folder = root / "sealed" / "holdout"
    if folder.is_dir():
        for path in folder.glob("*.parquet"):
            _try_chmod_readonly(path)
    capability = root / "sealed" / CAPABILITY_NAME
    if capability.is_file():
        _try_chmod_readonly(capability)
    lock_path = root / "sealed" / "holdout.lock.json"
    if lock_path.is_file():
        _try_chmod_readonly(lock_path)


def assert_holdout_sealed(root: Path, holdout_start: pd.Timestamp) -> None:
    lock_path = root / "sealed" / "holdout.lock.json"
    if not lock_path.is_file():
        raise PermissionError("missing sealed/holdout.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("readable_prices") or lock.get("readable_metrics"):
        raise PermissionError("holdout lock allows reading prices or metrics")
    names = (
        "daily_bars",
        "universe_membership",
        "trading_status",
        "execution_bars",
        "execution_status",
    )
    for name in names:
        path = root / "standardized" / f"{name}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["date"])
        if holdout_index(frame["date"], holdout_start).any():
            raise PermissionError(f"{name} still contains Holdout dates")


def load_sealed_holdout_prices(
    root: Path,
    *,
    capability_token: str,
) -> dict[str, pd.DataFrame]:
    token = str(capability_token or "").strip()
    token_fp = _token_digest(token)[:12] if token else ""
    _append_access_log(root, "attempt", token_fingerprint=token_fp)
    if not token:
        _append_access_log(root, "denied", reason="empty_token", token_fingerprint=token_fp)
        raise PermissionError("Holdout prices are sealed")
    lock_path = root / "sealed" / "holdout.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("readable_prices"):
        _append_access_log(root, "denied", reason="lock_readable_prices")
        raise PermissionError("refuse to load: lock marked readable_prices")
    expected = str(lock.get("capability_hash") or "")
    if not expected or not hmac.compare_digest(_token_digest(token), expected):
        _append_access_log(root, "denied", reason="token_mismatch", token_fingerprint=token_fp)
        raise PermissionError("Holdout prices are sealed")
    capability_path = root / "sealed" / CAPABILITY_NAME
    if not capability_path.is_file():
        _append_access_log(root, "denied", reason="capability_consumed")
        raise PermissionError("Holdout capability token already consumed")
    stored = capability_path.read_text(encoding="utf-8").strip()
    if not hmac.compare_digest(stored, token):
        _append_access_log(root, "denied", reason="capability_file_mismatch")
        raise PermissionError("Holdout prices are sealed")
    used_path = root / "sealed" / f"{CAPABILITY_NAME}.used"
    _try_chmod_user_write(capability_path)
    capability_path.replace(used_path)
    _try_chmod_readonly(used_path)
    out = {}
    for name, entry in (lock.get("files") or {}).items():
        out[name] = pd.read_parquet(root / entry["path"])
    _append_access_log(root, "granted", tables=sorted(out), token_fingerprint=token_fp)
    return out
