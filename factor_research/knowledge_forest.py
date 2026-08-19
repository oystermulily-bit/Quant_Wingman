from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import uuid4


class KnowledgeForest:
    """Persistent hypothesis/task/formula/experiment DAG in the research DB."""

    def __init__(self, connect):
        self._connect = connect
        self._initialise()

    def _initialise(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS hypotheses(
                    hypothesis_id TEXT PRIMARY KEY, parent_id TEXT, round_index INTEGER,
                    statement TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS tasks(
                    task_id TEXT PRIMARY KEY, hypothesis_id TEXT, round_index INTEGER,
                    status TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS formulas(
                    factor_id TEXT PRIMARY KEY, hypothesis_id TEXT, round_index INTEGER,
                    formula_decoded TEXT NOT NULL, status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments(
                    experiment_id TEXT PRIMARY KEY, factor_id TEXT, round_index INTEGER,
                    payload_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS validation_results(
                    result_id TEXT PRIMARY KEY, factor_id TEXT, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS factor_relations(
                    source_factor_id TEXT, target_factor_id TEXT, relation TEXT,
                    value REAL, PRIMARY KEY(source_factor_id,target_factor_id,relation)
                );
                CREATE TABLE IF NOT EXISTS failures(
                    failure_id TEXT PRIMARY KEY, factor_id TEXT, round_index INTEGER,
                    reason TEXT NOT NULL, details_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS analysis_reports(
                    report_id TEXT PRIMARY KEY, round_index INTEGER, payload_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def add_hypothesis(self, round_index: int, statement: str, parent_id: str | None = None) -> str:
        statement = statement or "unspecified"
        with self._connect() as db:
            existing = db.execute(
                """SELECT hypothesis_id FROM hypotheses
                   WHERE round_index=? AND statement=? AND status='proposed'
                   ORDER BY created_at LIMIT 1""",
                (round_index, statement),
            ).fetchone()
            if existing:
                return str(existing["hypothesis_id"])
            if parent_id is None:
                proposed_parent = db.execute(
                    """SELECT hypothesis_id FROM hypotheses
                       WHERE round_index=? AND status='proposed'
                       ORDER BY created_at DESC LIMIT 1""",
                    (round_index,),
                ).fetchone()
                if proposed_parent:
                    parent_id = str(proposed_parent["hypothesis_id"])
        hypothesis_id = uuid4().hex
        with self._connect() as db:
            db.execute(
                """INSERT INTO hypotheses(
                       hypothesis_id,parent_id,round_index,statement,status,created_at
                   ) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)""",
                (hypothesis_id, parent_id, round_index, statement, "proposed"),
            )
        return hypothesis_id

    def record_failure(
        self,
        factor_id: str | None,
        round_index: int,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO failures VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)",
                (
                    uuid4().hex,
                    factor_id,
                    round_index,
                    reason,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )

    def record_assessment(self, assessment, round_index: int, hypothesis_id: str) -> None:
        status = "accepted" if assessment.decision.accepted else "rejected"
        payload = assessment.public_summary()
        with self._connect() as db:
            db.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)",
                (
                    uuid4().hex,
                    hypothesis_id,
                    round_index,
                    status,
                    json.dumps(
                        {"factor_id": assessment.factor_id, "kind": "factor_validation"},
                        ensure_ascii=False,
                    ),
                ),
            )
            db.execute(
                "INSERT OR REPLACE INTO formulas VALUES(?,?,?,?,?)",
                (assessment.factor_id, hypothesis_id, round_index, assessment.formula_decoded, status),
            )
            db.execute(
                "INSERT INTO experiments VALUES(?,?,?,?,CURRENT_TIMESTAMP)",
                (uuid4().hex, assessment.factor_id, round_index, json.dumps(payload, ensure_ascii=False)),
            )
            db.execute(
                "INSERT INTO validation_results VALUES(?,?,?)",
                (uuid4().hex, assessment.factor_id, json.dumps(assessment.validation.to_dict(), ensure_ascii=False)),
            )
            for reason in assessment.decision.reasons:
                if not assessment.decision.accepted:
                    db.execute(
                        "INSERT INTO failures VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)",
                        (uuid4().hex, assessment.factor_id, round_index, reason, "{}"),
                    )
            db.execute(
                "UPDATE hypotheses SET status=? WHERE hypothesis_id=?",
                (status, hypothesis_id),
            )

    def record_relations(self, factor_id: str, correlations: list[tuple[str, float]]) -> None:
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO factor_relations VALUES(?,?,?,?)",
                [(factor_id, other, "correlation", float(value)) for other, value in correlations],
            )

    def record_analysis(self, round_index: int, report: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO analysis_reports VALUES(?,?,?,CURRENT_TIMESTAMP)",
                (uuid4().hex, round_index, json.dumps(report, ensure_ascii=False)),
            )
            parent = db.execute(
                """SELECT hypothesis_id FROM hypotheses
                   WHERE round_index=? AND status='accepted'
                   ORDER BY created_at DESC LIMIT 1""",
                (round_index,),
            ).fetchone()
            if parent is None:
                parent = db.execute(
                    """SELECT hypothesis_id FROM hypotheses
                       WHERE round_index=? ORDER BY created_at DESC LIMIT 1""",
                    (round_index,),
                ).fetchone()
            parent_id = str(parent["hypothesis_id"]) if parent else None
            for statement in report.get("next_hypotheses", []):
                db.execute(
                    """INSERT INTO hypotheses(
                           hypothesis_id,parent_id,round_index,statement,status,created_at
                       ) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)""",
                    (uuid4().hex, parent_id, round_index + 1, str(statement), "proposed"),
                )

    def guidance(self, limit: int = 8) -> dict[str, Any]:
        with self._connect() as db:
            next_rows = db.execute(
                "SELECT payload_json FROM analysis_reports ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            failures = db.execute(
                "SELECT reason, COUNT(*) n FROM failures GROUP BY reason ORDER BY n DESC LIMIT ?",
                (limit,),
            ).fetchall()
            proposed = db.execute(
                """SELECT statement FROM hypotheses WHERE status='proposed'
                   ORDER BY round_index DESC, created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        report = json.loads(next_rows[0]) if next_rows else {}
        return {
            "next_hypotheses": (
                [row["statement"] for row in proposed]
                or report.get("next_hypotheses", [])[:limit]
            ),
            "formula_mutations": report.get("formula_mutations", [])[:limit],
            "repeated_failure_reasons": [row["reason"] for row in failures],
        }
