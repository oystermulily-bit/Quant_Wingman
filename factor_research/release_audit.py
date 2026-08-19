from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ResearchConfig
from .factor_artifact import data_fingerprint


@dataclass(frozen=True)
class ReleaseAuditOutcome:
    release_id: str
    report_path: str | None
    result: dict[str, Any] | None
    reused: bool
    status: str


class ReleaseAuditStore:
    """Immutable, exactly-once holdout audit registry."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "research.sqlite3"
        self._initialise()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _initialise(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS release_audits(
                    release_id TEXT NOT NULL,
                    data_fingerprint TEXT NOT NULL,
                    sota_snapshot_hash TEXT NOT NULL,
                    model_config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    report_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    PRIMARY KEY(
                        release_id, data_fingerprint,
                        sota_snapshot_hash, model_config_hash
                    )
                );
                CREATE TRIGGER IF NOT EXISTS release_audits_no_delete
                BEFORE DELETE ON release_audits
                BEGIN
                    SELECT RAISE(ABORT, 'release audits are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS release_audits_no_overwrite
                BEFORE UPDATE ON release_audits
                WHEN OLD.status != 'reserved'
                     OR NEW.release_id != OLD.release_id
                     OR NEW.data_fingerprint != OLD.data_fingerprint
                     OR NEW.sota_snapshot_hash != OLD.sota_snapshot_hash
                     OR NEW.model_config_hash != OLD.model_config_hash
                BEGIN
                    SELECT RAISE(ABORT, 'release audits cannot be overwritten');
                END;
                """
            )

    def reserve(
        self,
        *,
        release_id: str,
        data_fingerprint_value: str,
        sota_snapshot_hash: str,
        model_config_hash: str,
        report_path: Path,
    ) -> tuple[bool, sqlite3.Row | None]:
        key = (
            release_id,
            data_fingerprint_value,
            sota_snapshot_hash,
            model_config_hash,
        )
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                """SELECT * FROM release_audits WHERE
                   release_id=? AND data_fingerprint=?
                   AND sota_snapshot_hash=? AND model_config_hash=?""",
                key,
            ).fetchone()
            if existing is not None:
                db.commit()
                return False, existing
            db.execute(
                """INSERT INTO release_audits(
                       release_id,data_fingerprint,sota_snapshot_hash,
                       model_config_hash,status,report_path
                   ) VALUES(?,?,?,?,?,?)""",
                (*key, "reserved", str(report_path.resolve())),
            )
            db.commit()
            return True, None
        finally:
            db.close()

    def finish(
        self,
        *,
        release_id: str,
        data_fingerprint_value: str,
        sota_snapshot_hash: str,
        model_config_hash: str,
        result: dict[str, Any] | None,
        error: str | None = None,
    ) -> None:
        status = "complete" if error is None else "failed"
        with self.connect() as db:
            changed = db.execute(
                """UPDATE release_audits SET status=?, result_json=?, error=?,
                   completed_at=CURRENT_TIMESTAMP
                   WHERE release_id=? AND data_fingerprint=?
                   AND sota_snapshot_hash=? AND model_config_hash=?
                   AND status='reserved'""",
                (
                    status,
                    None if result is None else json.dumps(result, ensure_ascii=False),
                    error,
                    release_id,
                    data_fingerprint_value,
                    sota_snapshot_hash,
                    model_config_hash,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("release audit reservation is missing or immutable")


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(engine) -> tuple[list[list[int]], str, str]:
    rows: list[dict] = []
    research = getattr(engine, "factor_research", None)
    if research is not None:
        rows = research.store.accepted(
            research.data_fingerprint, research.validation_protocol
        )
        rows = sorted(rows, key=lambda row: row["factor_id"])
    formulas = [list(row["formula_tokens"]) for row in rows]
    if not formulas and getattr(engine, "best_formula", None) is not None:
        formulas = [list(engine.best_formula)]
    snapshot = [
        {"factor_id": row.get("factor_id"), "formula_tokens": row["formula_tokens"]}
        for row in rows
    ] or [{"factor_id": getattr(engine, "best_factor_id", None), "formula_tokens": f}
          for f in formulas]
    protocol = (
        research.validation_protocol
        if research is not None
        else str(getattr(engine, "data_protocol", "unknown"))
    )
    return formulas, _digest(snapshot), _digest(
        {
            "generator_backend": getattr(engine, "generator_backend", None),
            "research_protocol": protocol,
            "cost_rate": 0.0003,
            "holdout_fraction": getattr(engine, "holdout_fraction", None),
            "sota_aggregation": "equal_weight_mean_v1",
        }
    )


def audit_engine_release(
    engine,
    full_data_manager,
    spec,
    symbol: str,
    *,
    store_root: Path | None = None,
    report_root: Path = Path("strategies") / "releases",
) -> ReleaseAuditOutcome:
    """Audit one frozen SOTA snapshot once; repeated calls only reuse its row."""
    from model_core.holdout import evaluate_holdout_snapshot

    formulas, sota_hash, model_hash = _snapshot(engine)
    if not formulas:
        raise ValueError("release has no frozen SOTA formula")
    full_fingerprint = data_fingerprint(full_data_manager)
    release_id = "release-" + _digest(
        {
            "symbol": symbol,
            "data": full_fingerprint,
            "sota": sota_hash,
            "model": model_hash,
        }
    )[:16]
    report_path = Path(report_root) / release_id / f"holdout_{symbol}.json"
    root = store_root or ResearchConfig().for_symbol(symbol)
    store = ReleaseAuditStore(root)
    reserved, existing = store.reserve(
        release_id=release_id,
        data_fingerprint_value=full_fingerprint,
        sota_snapshot_hash=sota_hash,
        model_config_hash=model_hash,
        report_path=report_path,
    )
    if not reserved:
        result = json.loads(existing["result_json"]) if existing["result_json"] else None
        return ReleaseAuditOutcome(
            release_id,
            existing["report_path"],
            result,
            True,
            str(existing["status"]),
        )

    try:
        result = evaluate_holdout_snapshot(full_data_manager, formulas, spec)
        result.update(
            release_id=release_id,
            data_fingerprint=full_fingerprint,
            sota_snapshot_hash=sota_hash,
            model_config_hash=model_hash,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            raise FileExistsError(f"immutable release report already exists: {report_path}")
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"symbol": symbol, "formulas": formulas, **result},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(report_path)
        store.finish(
            release_id=release_id,
            data_fingerprint_value=full_fingerprint,
            sota_snapshot_hash=sota_hash,
            model_config_hash=model_hash,
            result=result,
        )
        return ReleaseAuditOutcome(
            release_id, str(report_path.resolve()), result, False, "complete"
        )
    except Exception as exc:
        store.finish(
            release_id=release_id,
            data_fingerprint_value=full_fingerprint,
            sota_snapshot_hash=sota_hash,
            model_config_hash=model_hash,
            result=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
