"""Immutable raw snapshot writer. Successful responses are never overwritten."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SCHEMA_VERSION, SNAPSHOT_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class RawStore:
    def __init__(self, root: Path, *, raw_root: Path | None = None) -> None:
        self.root = Path(root)
        self.raw_root = Path(raw_root) if raw_root is not None else self.root / "raw"
        self.raw_root.mkdir(parents=True, exist_ok=True)

    def part_paths(self, method: str, part_id: str) -> tuple[Path, Path]:
        folder = self.raw_root / method
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{part_id}.parquet", folder / f"{part_id}.meta.json"

    def is_complete(self, method: str, part_id: str) -> bool:
        data_path, meta_path = self.part_paths(method, part_id)
        if not data_path.exists() or not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return bool(meta.get("success")) and meta.get("response_sha256") == sha256_file(data_path)

    def save(
        self,
        *,
        method: str,
        part_id: str,
        frame: pd.DataFrame,
        parameters: dict[str, Any],
        mcp_server_sha256: str,
        sdk_version: str,
        retry_count: int = 0,
    ) -> dict[str, Any]:
        data_path, meta_path = self.part_paths(method, part_id)
        if self.is_complete(method, part_id):
            return json.loads(meta_path.read_text(encoding="utf-8")) | {"skipped": True}

        started = datetime.now(timezone.utc).isoformat()
        tmp = data_path.with_suffix(".parquet.partial")
        frame.to_parquet(tmp, index=False)
        tmp.replace(data_path)
        digest = sha256_file(data_path)
        completed = datetime.now(timezone.utc).isoformat()
        meta = {
            "snapshot_id": SNAPSHOT_ID,
            "schema_version": SCHEMA_VERSION,
            "source": "AmazingData",
            "mcp_server_sha256": mcp_server_sha256,
            "sdk_version": sdk_version,
            "method": method,
            "part_id": part_id,
            "parameters": parameters,
            "requested_at": started,
            "completed_at": completed,
            "response_sha256": digest,
            "row_count": int(len(frame)),
            "success": True,
            "retry_count": retry_count,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def save_failure(
        self,
        *,
        method: str,
        part_id: str,
        parameters: dict[str, Any],
        error: str,
        retry_count: int,
        mcp_server_sha256: str,
        sdk_version: str,
    ) -> None:
        _, meta_path = self.part_paths(method, part_id)
        fail_path = meta_path.with_name(f"{part_id}.retry{retry_count}.meta.json")
        fail_path.write_text(
            json.dumps(
                {
                    "snapshot_id": SNAPSHOT_ID,
                    "source": "AmazingData",
                    "mcp_server_sha256": mcp_server_sha256,
                    "sdk_version": sdk_version,
                    "method": method,
                    "part_id": part_id,
                    "parameters": parameters,
                    "success": False,
                    "retry_count": retry_count,
                    "error": error,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
