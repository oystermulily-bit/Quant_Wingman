from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import torch

from .factor_artifact import load_factor_values, save_factor_values


class SOTAStore:
    """SQLite metadata plus NPZ factor values, scoped by research protocol."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.values_dir = self.root / "factor_values"
        self.db_path = self.root / "research.sqlite3"
        self._initialise()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialise(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS factors (
                    factor_id TEXT PRIMARY KEY,
                    canonical_formula TEXT NOT NULL,
                    formula_tokens TEXT NOT NULL,
                    formula_decoded TEXT NOT NULL,
                    formula_version INTEGER NOT NULL,
                    vocab_version TEXT NOT NULL,
                    feature_implementation_version TEXT NOT NULL,
                    data_fingerprint TEXT NOT NULL,
                    validation_protocol TEXT NOT NULL,
                    generator TEXT NOT NULL,
                    llm_model TEXT NOT NULL,
                    hypothesis_id TEXT,
                    created_round INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    max_correlation REAL NOT NULL,
                    ranking_score REAL NOT NULL,
                    artifact_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_factor_scope
                ON factors(data_fingerprint, validation_protocol, status);
                """
            )

    def has(self, factor_id: str) -> bool:
        with self.connect() as db:
            return db.execute(
                "SELECT 1 FROM factors WHERE factor_id=?", (factor_id,)
            ).fetchone() is not None

    def save_result(
        self,
        record: dict[str, Any],
        factor: torch.Tensor | None,
        invalid_mask: torch.Tensor | None = None,
    ) -> None:
        artifact_path: str | None = None
        if factor is not None and record["status"] == "accepted":
            path = self.values_dir / f"{record['factor_id']}.npz"
            save_factor_values(path, factor, invalid_mask)
            artifact_path = str(path.resolve())
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO factors(
                    factor_id, canonical_formula, formula_tokens, formula_decoded,
                    formula_version, vocab_version, feature_implementation_version,
                    data_fingerprint, validation_protocol, generator, llm_model,
                    hypothesis_id, created_round, status, metrics_json,
                    max_correlation, ranking_score, artifact_path
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(factor_id) DO UPDATE SET
                    status=excluded.status,
                    metrics_json=excluded.metrics_json,
                    max_correlation=excluded.max_correlation,
                    ranking_score=excluded.ranking_score,
                    artifact_path=COALESCE(excluded.artifact_path, factors.artifact_path)
                """,
                (
                    record["factor_id"], record["canonical_formula"],
                    json.dumps(record["formula_tokens"]), record["formula_decoded"],
                    record.get("formula_version", 1), record["vocab_version"],
                    record["feature_implementation_version"], record["data_fingerprint"],
                    record["validation_protocol"], record.get("generator", "rd_agent"),
                    record["llm_model"], record.get("hypothesis_id"),
                    record["created_round"], record["status"],
                    json.dumps(record["metrics"], ensure_ascii=False, allow_nan=False),
                    float(record["max_correlation"]), float(record["ranking_score"]),
                    artifact_path,
                ),
            )

    def accepted(self, data_fingerprint: str, validation_protocol: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM factors
                   WHERE data_fingerprint=? AND validation_protocol=? AND status='accepted'
                   ORDER BY ranking_score DESC, created_at ASC""",
                (data_fingerprint, validation_protocol),
            ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["formula_tokens"] = json.loads(result["formula_tokens"])
        result["metrics"] = json.loads(result.pop("metrics_json"))
        return result

    def matrix(self, data_fingerprint: str, validation_protocol: str) -> tuple[torch.Tensor | None, list[dict]]:
        rows = self.accepted(data_fingerprint, validation_protocol)
        values: list[torch.Tensor] = []
        valid_rows: list[dict] = []
        for row in rows:
            path = row.get("artifact_path")
            if not path or not Path(path).exists():
                continue
            values.append(load_factor_values(Path(path)))
            valid_rows.append(row)
        if not values:
            return None, []
        shape = values[0].shape
        if any(value.shape != shape for value in values):
            raise ValueError("SOTA factor artifact shapes are inconsistent")
        return torch.stack(values, dim=1), valid_rows

    def status(self, data_fingerprint: str, validation_protocol: str) -> dict[str, Any]:
        rows = self.accepted(data_fingerprint, validation_protocol)
        return {
            "accepted_count": len(rows),
            "factor_ids": [row["factor_id"] for row in rows],
            "best_ranking_score": rows[0]["ranking_score"] if rows else None,
            "db_path": str(self.db_path.resolve()),
        }
