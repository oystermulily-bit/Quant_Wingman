"""Non-promotable, score-only exchange files for the offline fitting demo.

This module never reads research inputs, opens Holdout, saves executable model
pickles, applies a whitelist, or produces portfolio/trading recommendations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"
ARTIFACT_KIND = "OFFLINE_DEMO_SCORE_BUNDLE"
COMPLETION_STATUS = "OFFLINE_DEMO_COMPLETED_NOT_VALIDATED"
SCORE_COLUMNS = (
    "date", "code", "score", "score_available_at", "fold_id", "is_member", "signal_valid",
)
FALSE_FLAGS = (
    "research_go", "holdout_read", "sota_promoted", "production_allowed", "whitelist_applied",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BANNED_KEYS = frozenset({
    "weight", "weights", "target_weight", "target_weights", "allocation", "allocations",
    "recommendation", "recommendations", "order", "orders", "trade_order", "trade_orders",
})


def _reject_sensitive_path(path: Path) -> None:
    for part in path.parts:
        token = part.lower().replace("-", "_")
        if "holdout" in token or "sealed" in token:
            raise ValueError("Holdout/sealed paths are forbidden in the offline demo")
        if re.match(r"stage\d", token):
            raise ValueError("historical stage directories cannot be demo input/output targets")


def validate_demo_input_path(source: str | Path) -> Path:
    """Validate a prepared input path without opening it.

    The caller must additionally validate the file's declared data role, all
    timestamps, the panel boundary and its fingerprint. A filename is NOT proof
    that the payload is Development. No directory traversal or auto-discovery
    of research files is performed here.
    """
    supplied = Path(source)
    _reject_sensitive_path(supplied)
    resolved = supplied.resolve(strict=True)
    _reject_sensitive_path(resolved)
    if not resolved.is_file() or resolved.suffix.lower() not in {".npz", ".json"}:
        raise ValueError("prepared demo inputs must be an explicit NPZ or JSON file")
    role_named = any(
        "development" in part.lower() or "synthetic" in part.lower()
        for part in resolved.parts
    )
    if not role_named:
        raise ValueError("prepared input path must explicitly identify Development or synthetic data")
    return resolved


def _json_ready(value: Any) -> bytes:
    """Require ordinary finite JSON, and forbid a trading-output payload."""
    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                if key.lower() in _BANNED_KEYS:
                    raise ValueError(f"score-only bundle cannot contain trading field {key!r}")
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ValueError("report must contain finite JSON values only") from exc


def _validate_report(report: dict[str, Any], row_count: int) -> None:
    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary")
    if not isinstance(report.get("run_id"), str) or not report["run_id"].strip():
        raise ValueError("run_id must be nonempty")
    if report.get("data_role") not in {"development", "synthetic"}:
        raise ValueError("data_role must be development or synthetic")
    if report.get("status") != COMPLETION_STATUS:
        raise ValueError("offline demonstration cannot claim validated or production status")
    for key in FALSE_FLAGS:
        if report.get(key) is not False:
            raise ValueError(f"{key} must be explicitly false")
    for key, expected in (("schema_version", SCHEMA_VERSION), ("artifact_kind", ARTIFACT_KIND)):
        if key in report and report[key] != expected:
            raise ValueError(f"invalid {key}")
    fingerprints = report.get("fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {"model", "formula", "fold", "data"}:
        raise ValueError("model/formula/fold/data fingerprints are required")
    if any(not isinstance(v, str) or not _SHA256.fullmatch(v) for v in fingerprints.values()):
        raise ValueError("fingerprints must be lowercase SHA-256 hex digests")
    if type(report.get("expected_score_rows")) is not int or report["expected_score_rows"] < 1:
        raise ValueError("expected_score_rows must be a positive integer")
    if report["expected_score_rows"] != row_count:
        raise ValueError("score row count differs from the original evaluation grid")
    selected = report.get("selected_features")
    if not isinstance(selected, list) or not selected or any(
        not isinstance(item, str) or not item.strip() for item in selected
    ) or len(set(selected)) != len(selected):
        raise ValueError("selected_features must be nonempty, unique feature IDs")
    if not isinstance(report.get("formulas"), list):
        raise ValueError("formulas must be a list (empty is allowed when no formula was generated)")
    if not isinstance(report.get("model_evidence"), dict) or not report["model_evidence"]:
        raise ValueError("nonempty model_evidence is required")
    _json_ready(report)


def _validated_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scores, pd.DataFrame) or set(scores.columns) != set(SCORE_COLUMNS):
        raise ValueError("scores must contain exactly the seven score-only contract columns")
    if scores.columns.duplicated().any() or scores.empty:
        raise ValueError("score columns must be unique and the evaluation grid nonempty")
    result = scores.loc[:, SCORE_COLUMNS].copy()
    for col in ("code", "fold_id"):
        if result[col].map(lambda x: not isinstance(x, str) or not x.strip()).any():
            raise ValueError(f"{col} must contain nonempty strings")
    for col in ("is_member", "signal_valid"):
        if result[col].map(lambda x: not isinstance(x, (bool, np.bool_))).any():
            raise ValueError(f"{col} must contain explicit booleans, without missing values")
        result[col] = result[col].astype(bool)
    dates = []
    available = []
    for date, timestamp in zip(result["date"], result["score_available_at"]):
        day = pd.Timestamp(date)
        if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
            raise ValueError("date must be a timezone-naive calendar date without time-of-day")
        known_at = pd.Timestamp(timestamp)
        if pd.isna(known_at) or known_at.tzinfo is None:
            raise ValueError("score_available_at must be an explicit timezone-aware timestamp")
        if known_at.tz_convert("Asia/Shanghai").date() != day.date():
            raise ValueError("score availability must belong to its Asia/Shanghai signal date")
        dates.append(day)
        available.append(known_at.tz_convert("UTC"))
    result["date"] = pd.DatetimeIndex(dates)
    result["score_available_at"] = pd.DatetimeIndex(available)
    if result.duplicated(["date", "code", "fold_id"]).any():
        raise ValueError("duplicate date/code/fold score rows")
    try:
        numeric = pd.to_numeric(result["score"], errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("score must be numeric or missing") from exc
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("infinite scores are not permitted")
    expected_valid = numeric.notna() & result["is_member"]
    if not result["signal_valid"].equals(expected_valid):
        raise ValueError("signal_valid must exactly identify finite member scores")
    if (numeric.notna() & ~result["is_member"]).any():
        raise ValueError("nonmember rows must retain a missing score")
    result["score"] = numeric
    # No dropna, ranking, membership filtering or missing-value imputation here.
    return result


def write_demo_bundle(output_dir: str | Path, report: dict[str, Any], scores: pd.DataFrame) -> dict[str, str]:
    """Write a new, score-only bundle, without overwriting an existing run.

    ``manifest.json`` is written last and is the only completion marker. If a
    filesystem/Parquet failure occurs, the partial directory is intentionally
    preserved without a completed manifest; rerun into another new directory.
    The basename must start with ``offline_demo_``. Parent must already exist,
    preventing accidental creation of a broad or inferred directory tree.
    """
    supplied = Path(output_dir)
    _reject_sensitive_path(supplied)
    destination = supplied.resolve()
    _reject_sensitive_path(destination)
    if not destination.name.startswith("offline_demo_") or destination.name == "offline_demo_":
        raise ValueError("output must be an explicitly named offline_demo_* directory")
    if destination.exists() or supplied.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing demo output: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("output parent must already exist")
    normalized_scores = _validated_scores(scores)
    _validate_report(report, len(normalized_scores))
    evidence = {
        "selected_features.json": report["selected_features"],
        "formulas.json": report["formulas"],
        "model_evidence.json": report["model_evidence"],
    }
    evidence_bytes = {name: _json_ready(value) for name, value in evidence.items()}
    destination.mkdir(exist_ok=False)
    paths: dict[str, str] = {}
    files: dict[str, Any] = {}
    score_path = destination / "scores.parquet"
    # Directory creation above is exclusive; no prior run files can be replaced.
    with score_path.open("xb") as stream:
        normalized_scores.to_parquet(stream, index=False)
    files["scores.parquet"] = {"sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                              "rows": len(normalized_scores)}
    paths["scores"] = str(score_path)
    for name, payload in evidence_bytes.items():
        with (destination / name).open("xb") as stream:
            stream.write(payload)
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest()}
        paths[name.removesuffix(".json")] = str(destination / name)
    member_count = int(normalized_scores["is_member"].sum())
    valid_count = int(normalized_scores["signal_valid"].sum())
    manifest = {
        **report,
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score_columns": list(SCORE_COLUMNS),
        "availability_semantics": "signal_time_Asia_Shanghai;timezone_aware;not_file_creation_time",
        "missing_policy": "preserve_original_evaluation_grid_no_zero_fill",
        "score_rows": len(normalized_scores),
        "member_rows": member_count,
        "valid_member_rows": valid_count,
        "member_coverage": valid_count / member_count if member_count else None,
        "files": files,
    }
    with (destination / "manifest.json").open("xb") as stream:
        stream.write(_json_ready(manifest))
    paths["manifest"] = str(destination / "manifest.json")
    return paths
