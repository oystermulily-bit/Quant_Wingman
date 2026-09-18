"""Fail-closed admission and persistent budgets for independent S2 research.

This module does not inspect historical D21 outcomes, read research data, touch
Holdout, or promote SOTA. Engineering authorization and execution freeze are
separate. A frozen live plan binds one ledger directory, so changing the run ID
or output directory cannot silently reset its search allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any


PROTOCOL_ID = "JOINT_RESEARCH_S2"
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical_fingerprint(value: Any) -> str:
    """Hash an exact JSON configuration; reject NaN and non-JSON objects."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(name: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, not a boolean")


def _identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve()))


@dataclass(frozen=True)
class FrozenJointPlan:
    protocol_id: str = PROTOCOL_ID
    hypothesis_id: str = ""
    design_approved: bool = False
    execution_frozen: bool = False
    feature_ids: tuple[str, ...] = ()
    # The first digest covers only ordered IDs. The contract digest separately
    # binds formula implementations, direction, availability, and missing rules.
    feature_manifest_hash: str = ""
    feature_contract_hash: str = ""
    data_fingerprint: str = ""
    split_fingerprint: str = ""
    config_fingerprint: str = ""
    implementation_hash: str = ""
    signal_output: str = ""
    freeze_id: str = ""
    risk_policy_hash: str = ""
    concentration_policy_hash: str = ""
    max_outer_scopes: int = 5
    max_trials_per_scope: int = 480
    max_total_trials: int = 2400
    ledger_directory: str = ""
    horizon: int = 5
    trade_universe: str = "PIT_HS300"

    def __post_init__(self) -> None:
        for name in ("design_approved", "execution_frozen"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("max_outer_scopes", "max_trials_per_scope", "max_total_trials", "horizon"):
            _positive_integer(name, getattr(self, name))
        if self.max_outer_scopes < 2:
            raise ValueError("joint research requires at least two outer scopes")
        if self.max_total_trials > self.max_outer_scopes * self.max_trials_per_scope:
            raise ValueError("total budget exceeds scope capacities")
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError("unsupported joint research protocol")
        if self.signal_output not in ("", "model_prediction", "equal_weight"):
            raise ValueError("signal_output must be an empty draft value, model_prediction, or equal_weight")
        if self.horizon != 5 or self.trade_universe != "PIT_HS300":
            raise ValueError("S2 admission is restricted to H5 / PIT_HS300")
        if not isinstance(self.feature_ids, tuple):
            raise ValueError("feature_ids must be an immutable tuple")
        for value in self.feature_ids:
            _identifier("feature ID", value)
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("duplicate feature IDs")
        for name in ("hypothesis_id", "freeze_id", "ledger_directory"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        for name in self._hash_fields():
            value = getattr(self, name)
            if not isinstance(value, str) or (value and not _HASH_RE.fullmatch(value)):
                raise ValueError(f"{name} must be an empty draft value or lowercase SHA256")
        if (self.feature_manifest_hash and
                self.feature_manifest_hash != canonical_fingerprint(self.feature_ids)):
            raise ValueError("feature_manifest_hash does not match ordered feature_ids")

    @staticmethod
    def _hash_fields() -> tuple[str, ...]:
        return ("feature_manifest_hash", "feature_contract_hash", "data_fingerprint", "split_fingerprint",
                "config_fingerprint", "implementation_hash", "risk_policy_hash",
                "concentration_policy_hash")

    def to_dict(self) -> dict:
        # Explicit JSON round trip makes this immediately serializable and
        # exposes feature_ids as an array, not a Python-only tuple.
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value: dict) -> "FrozenJointPlan":
        if not isinstance(value, dict):
            raise ValueError("plan must be an object")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown plan fields: {sorted(unknown)}")
        copied = dict(value)
        if "feature_ids" in copied:
            if not isinstance(copied["feature_ids"], (list, tuple)):
                raise ValueError("feature_ids must be an array")
            copied["feature_ids"] = tuple(copied["feature_ids"])
        return cls(**copied)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    def validate(self, *, mode: str = "development") -> dict:
        if mode not in ("synthetic", "development"):
            raise ValueError("only synthetic or development modes are permitted")
        if mode == "development":
            missing = []
            for name in ("hypothesis_id", "freeze_id", "ledger_directory", "signal_output", *self._hash_fields()):
                if not getattr(self, name).strip():
                    missing.append(name)
            if not self.feature_ids:
                missing.append("feature_ids")
            if not self.design_approved:
                missing.append("design_approved")
            if not self.execution_frozen:
                missing.append("execution_frozen")
            if missing:
                raise ValueError("live research admission denied; missing: " + ", ".join(missing))
            if self.max_outer_scopes != 5:
                raise ValueError("live S2 requires exactly five frozen outer scopes")
            if not Path(self.ledger_directory).is_absolute():
                raise ValueError("live ledger_directory must be an absolute frozen path")
        return {
            "protocol_id": self.protocol_id,
            "plan_fingerprint": self.fingerprint,
            "mode": mode,
            "status": "ENGINEERING_ONLY" if mode == "synthetic" else "JOINT_RESEARCH_ADMITTED",
            "research_evidence": False,
            "evidence_generated": False,
            "development_execution_allowed": mode == "development",
            "research_passed": False,
            "production_allowed": False,
            "holdout_allowed": False,
            "holdout_read": False,
            "sota_allowed": False,
            "sota_promoted": False,
            "stage5_allowed": False,
        }


class RunLedger:
    """Append-only run/scope identities and nonrefundable trial reservations.

    Caller provides a new output directory (or reopens this module's existing
    ledger there). A completed scope records actual trials but retains its full
    reservation against the plan total, including failures and process crashes.
    This conservative accounting also covers an unrecorded interrupted trial.
    It does not reuse historical D20's 480-trial budget: S2 has its own explicit
    per-scope and total caps, frozen as part of the plan.
    """

    filename = "joint_run_ledger.sqlite3"

    def __init__(self, output_directory: str | Path):
        if not isinstance(output_directory, (str, Path)) or not str(output_directory).strip():
            raise ValueError("an explicit output directory is required")
        self.output_directory = Path(output_directory).resolve()
        if self.output_directory == Path.cwd().resolve() or self.output_directory.parent == self.output_directory:
            raise ValueError("ledger requires a dedicated output directory")
        self.path = self.output_directory / self.filename
        if self.output_directory.exists() and not self.path.exists() and any(self.output_directory.iterdir()):
            raise ValueError("refusing to initialize ledger inside a nonempty unrelated directory")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        existing = self.path.exists()
        conn = self._connect()
        try:
            if existing:
                try:
                    version = conn.execute("SELECT value FROM metadata WHERE key = 'schema'").fetchone()
                except sqlite3.DatabaseError as exc:
                    raise ValueError("not an S2 run ledger") from exc
                if version is None or version[0] != "joint_s2_v1":
                    raise ValueError("unsupported run ledger schema")
            else:
                conn.executescript("""
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    INSERT INTO metadata VALUES ('schema', 'joint_s2_v1');
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY,
                        plan_hash TEXT NOT NULL UNIQUE,
                        freeze_id TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT
                    );
                    CREATE UNIQUE INDEX frozen_live_once ON runs(freeze_id)
                        WHERE mode = 'development';
                    CREATE TABLE scopes (
                        run_id TEXT NOT NULL REFERENCES runs(run_id),
                        scope_id TEXT NOT NULL,
                        reserved_trials INTEGER NOT NULL,
                        actual_trials INTEGER,
                        status TEXT NOT NULL,
                        reserved_at TEXT NOT NULL,
                        finished_at TEXT,
                        PRIMARY KEY (run_id, scope_id)
                    );
                """)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _transaction(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        _identifier("run_id", run_id)
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("unknown run_id")
        return row

    @staticmethod
    def _plan(row: sqlite3.Row) -> FrozenJointPlan:
        plan = FrozenJointPlan.from_dict(json.loads(row["plan_json"]))
        if plan.fingerprint != row["plan_hash"]:
            raise ValueError("stored frozen plan fingerprint mismatch")
        return plan

    def begin_run(self, plan: FrozenJointPlan, run_id: str, *, mode: str = "synthetic") -> dict:
        if not isinstance(plan, FrozenJointPlan):
            raise ValueError("plan must be FrozenJointPlan")
        admission = plan.validate(mode=mode)
        _identifier("run_id", run_id)
        if mode == "development" and _path_key(plan.ledger_directory) != _path_key(self.output_directory):
            raise ValueError("ledger directory differs from the frozen plan; no budget reset allowed")
        with self._transaction() as conn:
            try:
                conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                             (run_id, plan.fingerprint, plan.freeze_id, mode,
                              json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False),
                              "RUNNING", self._now()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("run_id, frozen plan, or live freeze_id already registered; cannot restart budget") from exc
        return {**admission, "run_id": run_id, "ledger_path": str(self.path)}

    def reserve_scope(self, run_id: str, scope_id: str, max_trials: int) -> dict:
        _identifier("scope_id", scope_id)
        _positive_integer("max_trials", max_trials)
        with self._transaction() as conn:
            row = self._run(conn, run_id)
            if row["status"] != "RUNNING":
                raise ValueError("run is no longer active")
            plan = self._plan(row)
            scopes = conn.execute("SELECT COUNT(*), COALESCE(SUM(reserved_trials), 0) FROM scopes WHERE run_id = ?",
                                  (run_id,)).fetchone()
            if max_trials > plan.max_trials_per_scope:
                raise ValueError("scope reservation exceeds frozen per-scope budget")
            if scopes[0] >= plan.max_outer_scopes:
                raise ValueError("outer scope count exceeds frozen cap")
            if scopes[1] + max_trials > plan.max_total_trials:
                raise ValueError("reservation exceeds frozen total budget")
            try:
                conn.execute("INSERT INTO scopes VALUES (?, ?, ?, NULL, 'RESERVED', ?, NULL)",
                             (run_id, scope_id, max_trials, self._now()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("scope already reserved; interrupted reservations cannot be retried") from exc
        return {"run_id": run_id, "scope_id": scope_id, "reserved_trials": max_trials,
                "status": "RESERVED"}

    def complete_scope(self, run_id: str, scope_id: str, actual_trials: int, *, success: bool = True) -> dict:
        _identifier("scope_id", scope_id)
        if type(actual_trials) is not int or actual_trials < 0:
            raise ValueError("actual_trials must be a nonnegative integer")
        if type(success) is not bool:
            raise ValueError("success must be boolean")
        with self._transaction() as conn:
            row = self._run(conn, run_id)
            self._plan(row)
            if row["status"] != "RUNNING":
                raise ValueError("run is no longer active")
            scope = conn.execute("SELECT * FROM scopes WHERE run_id = ? AND scope_id = ?", (run_id, scope_id)).fetchone()
            if scope is None or scope["status"] != "RESERVED":
                raise ValueError("scope is missing or already completed")
            if actual_trials > scope["reserved_trials"]:
                raise ValueError("actual_trials exceeds reservation")
            status = "COMPLETED" if success else "FAILED"
            conn.execute("UPDATE scopes SET actual_trials = ?, status = ?, finished_at = ? WHERE run_id = ? AND scope_id = ?",
                         (actual_trials, status, self._now(), run_id, scope_id))
        return {"run_id": run_id, "scope_id": scope_id, "actual_trials": actual_trials,
                "charged_trials": scope["reserved_trials"], "status": status}

    def finish_run(self, run_id: str, *, success: bool = True) -> dict:
        if type(success) is not bool:
            raise ValueError("success must be boolean")
        with self._transaction() as conn:
            row = self._run(conn, run_id)
            plan = self._plan(row)
            if row["status"] != "RUNNING":
                raise ValueError("run is already finished")
            unfinished = conn.execute("SELECT COUNT(*) FROM scopes WHERE run_id = ? AND status != 'COMPLETED'", (run_id,)).fetchone()[0]
            count = conn.execute("SELECT COUNT(*) FROM scopes WHERE run_id = ?", (run_id,)).fetchone()[0]
            if success and (unfinished or not count):
                raise ValueError("cannot finish successfully with absent, failed, or unfinished scopes")
            if success and row["mode"] == "development" and count != plan.max_outer_scopes:
                raise ValueError("cannot finish live research before all five outer scopes complete")
            conn.execute("UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                         ("ENGINEERING_COMPLETED" if success and row["mode"] == "synthetic" else
                          "ECONOMIC_VALIDATION_PENDING" if success else "FAILED",
                          self._now(), run_id))
        return self.snapshot(run_id)

    def snapshot(self, run_id: str) -> dict:
        conn = self._connect()
        try:
            row = self._run(conn, run_id)
            scopes = [dict(value) for value in conn.execute("SELECT * FROM scopes WHERE run_id = ? ORDER BY rowid", (run_id,))]
            plan = self._plan(row).to_dict()
            charged = sum(value["reserved_trials"] for value in scopes)
            return {"run_id": run_id, "plan_fingerprint": row["plan_hash"], "freeze_id": row["freeze_id"],
                    "mode": row["mode"], "status": row["status"], "scopes": scopes,
                    "max_total_trials": plan["max_total_trials"], "charged_trials": charged,
                    "actual_trials": sum(value["actual_trials"] or 0 for value in scopes),
                    "remaining_trials": plan["max_total_trials"] - charged,
                    "research_passed": False, "holdout_read": False,
                    "production_allowed": False, "sota_promoted": False, "stage5_allowed": False}
        finally:
            conn.close()
