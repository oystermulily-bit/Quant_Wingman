"""Read-only /api/v2 research-status contract after Stage-4 KEEP_1D.

This is not a recommendation engine. It never starts RD-Agent, never reads
Holdout prices, and never returns tradable target weights.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STAGE4_REPORT_CANDIDATES = (
    ROOT / "experiments" / "stage4_hs300_v2_execution_ledger" / "stage4_report.json",
    ROOT / "experiments" / "stage4_hs300_v2" / "stage4_report.json",
)
SCHEMA_VERSION = "w1ngman_api_v2"
TZ = ZoneInfo("Asia/Shanghai")
SNAPSHOT_ID = "csi300_2014_present_v2"
ALLOWED_EXPERIMENT_IDS = {
    "stage4_hs300_v2_execution_ledger",
    "stage3_hs300_v2_execution_ledger",
    "stage4_hs300_v2",
    "stage3_hs300_v2",
}


def _now() -> str:
    return datetime.now(TZ).isoformat()


def _load_stage4() -> dict[str, Any]:
    for path in STAGE4_REPORT_CANDIDATES:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("missing frozen Stage-4 report")


def envelope(
    *,
    status: str,
    reason_codes: list[str],
    payload: dict[str, Any] | None = None,
    data_version: str | None = SNAPSHOT_ID,
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "request_id": str(uuid.uuid4()),
        "generated_at": _now(),
        "status": status,
        "reason_codes": reason_codes,
        "data_version": data_version,
        "model_version": None,
    }
    if payload:
        body.update(payload)
    return body


def system_status() -> dict[str, Any]:
    report = _load_stage4()
    return envelope(
        status="MODEL_NOT_VALIDATED",
        reason_codes=[
            "STATISTICAL_GATE_KEEP_1D_BASELINE",
            "STAGE5_NOT_ALLOWED",
            "HOLDOUT_SEALED",
            "RD_AGENT_CLOSED",
        ],
        payload={
            "system_status": report.get("system_status", "MODEL_NOT_VALIDATED"),
            "statistical_gate": report.get("statistical_gate", "KEEP_1D_BASELINE"),
            "horizon_conclusion": report.get("horizon_conclusion", "KEEP_1D_BASELINE"),
            "allowed_horizons": report.get("allowed_horizons", [1]),
            "rd_agent_allowed": False,
            "holdout_read": False,
            "data_gate": "PASSED",
            "portfolio_gate": "REFERENCE_ONLY",
            "product_gate": "PARTIAL",
            "production_eligible": False,
            "protocol": report.get("protocol"),
            "snapshot_id": report.get("snapshot_id", SNAPSHOT_ID),
            "next_allowed": [
                "retain 1-day SIMPLE_ENSEMBLE_V1 reference",
                "do not start Stage 5 / RD-Agent",
                "new research requires a re-frozen Stage-2 hypothesis",
            ],
        },
    )


def data_status() -> dict[str, Any]:
    report = _load_stage4()
    oof = report.get("oof_scope") or {}
    return envelope(
        status="DATA_READY_FOR_DEVELOPMENT",
        reason_codes=["DATA_GATE_PASSED", "HOLDOUT_PRICES_SEALED"],
        payload={
            "snapshot_id": report.get("snapshot_id", SNAPSHOT_ID),
            "research_start": "2014-01-02",
            "oof_start": oof.get("oof_start"),
            "oof_end": oof.get("oof_end"),
            "oof_date_count": oof.get("oof_date_count"),
            "holdout_start": "2024-08-26",
            "holdout_sealed": True,
            "industry_coverage_warning_rows": 13845,
        },
    )


def experiment_status(experiment_id: str) -> dict[str, Any]:
    if experiment_id not in ALLOWED_EXPERIMENT_IDS:
        return envelope(
            status="MODEL_NOT_VALIDATED",
            reason_codes=["UNKNOWN_EXPERIMENT"],
            payload={"experiment_id": experiment_id, "found": False},
        )
    report = _load_stage4()
    summary = {
        "experiment_id": experiment_id,
        "found": True,
        "protocol": report.get("protocol") if experiment_id.startswith("stage4") else report.get("stage3_protocol"),
        "statistical_gate": report.get("statistical_gate"),
        "system_status": report.get("system_status"),
        "rd_agent_allowed": False,
        "holdout_read": False,
        "baseline_1d_median_net_sharpe": (report.get("baseline_1d") or {}).get("median_sharpe"),
        "go_3d_passed": ((report.get("go") or {}).get("3") or {}).get("passed"),
        "go_5d_passed": ((report.get("go") or {}).get("5") or {}).get("passed"),
    }
    return envelope(
        status="MODEL_NOT_VALIDATED",
        reason_codes=["STATISTICAL_GATE_KEEP_1D_BASELINE"],
        payload={"experiment": summary},
    )


def refuse_recommendation(*, requested_horizon: int | None = None) -> dict[str, Any]:
    return envelope(
        status="MODEL_NOT_VALIDATED",
        reason_codes=[
            "STATISTICAL_GATE_KEEP_1D_BASELINE",
            "STAGE5_NOT_ALLOWED",
            "HOLDOUT_SEALED",
            "NO_TRADABLE_WEIGHTS",
        ],
        payload={
            "schema_version": "w1ngman_recommendation_v2",
            "as_of": None,
            "horizon": requested_horizon,
            "unit": "ACCOUNT_WEIGHT",
            "market_risk_budget": None,
            "cash_weight": 1.0,
            "expected_turnover": 0.0,
            "positions": [],
            "execution_window": None,
        },
    )
