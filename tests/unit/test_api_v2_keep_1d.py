from __future__ import annotations

from web import app as web_app
from web.v2_status import data_status, experiment_status, refuse_recommendation, system_status


def _assert_no_secrets_or_local_paths(payload: object) -> None:
    text = str(payload)
    lowered = text.lower()
    assert "D:/" not in text
    assert "D:\\" not in text
    assert "api_key" not in lowered
    assert "password" not in lowered
    assert "webhook" not in lowered


def test_system_status_is_keep_1d_and_not_production() -> None:
    payload = system_status()
    assert payload["schema_version"] == "w1ngman_api_v2"
    assert payload["status"] == "MODEL_NOT_VALIDATED"
    assert payload["statistical_gate"] == "KEEP_1D_BASELINE"
    assert payload["rd_agent_allowed"] is False
    assert payload["holdout_read"] is False
    assert payload["production_eligible"] is False
    assert payload["allowed_horizons"] == [1]
    assert "STAGE5_NOT_ALLOWED" in payload["reason_codes"]
    _assert_no_secrets_or_local_paths(payload)


def test_data_status_reports_snapshot_without_local_paths() -> None:
    payload = data_status()
    assert payload["status"] == "DATA_READY_FOR_DEVELOPMENT"
    assert payload["snapshot_id"] == "csi300_2014_present_v2"
    assert payload["holdout_sealed"] is True
    assert payload["oof_start"] == "2018-04-03"
    assert payload["oof_end"] == "2024-08-22"
    _assert_no_secrets_or_local_paths(payload)


def test_experiment_status_summarizes_stage4_without_go() -> None:
    for experiment_id in ("stage4_hs300_v2_execution_ledger", "stage4_hs300_v2"):
        payload = experiment_status(experiment_id)
        assert payload["experiment"]["found"] is True
        assert payload["experiment"]["statistical_gate"] == "KEEP_1D_BASELINE"
        assert payload["experiment"]["rd_agent_allowed"] is False
        assert payload["experiment"]["go_3d_passed"] is False
        assert payload["experiment"]["go_5d_passed"] is False
        _assert_no_secrets_or_local_paths(payload)


def test_unknown_experiment_returns_reason_code() -> None:
    payload = experiment_status("stage5_rd_agent")
    assert payload["status"] == "MODEL_NOT_VALIDATED"
    assert "UNKNOWN_EXPERIMENT" in payload["reason_codes"]
    assert payload["found"] is False


def test_recommendation_skeleton_returns_cash_and_no_positions() -> None:
    payload = refuse_recommendation(requested_horizon=5)
    assert payload["schema_version"] == "w1ngman_recommendation_v2"
    assert payload["status"] == "MODEL_NOT_VALIDATED"
    assert payload["positions"] == []
    assert payload["cash_weight"] == 1.0
    assert payload["horizon"] == 5
    assert payload["expected_turnover"] == 0.0
    assert "NO_TRADABLE_WEIGHTS" in payload["reason_codes"]
    _assert_no_secrets_or_local_paths(payload)


def test_v2_routes_are_registered() -> None:
    paths = {getattr(route, "path", None) for route in web_app.app.routes}
    assert "/api/v2/system-status" in paths
    assert "/api/v2/data-status" in paths
    assert "/api/v2/research/experiments/{experiment_id}" in paths
    assert "/api/v2/portfolio/recommendation" in paths


def test_v2_recommendation_route_refuses() -> None:
    payload = web_app.api_v2_portfolio_recommendation(
        web_app.RecommendationRequest(requested_horizon=3)
    )
    assert payload["positions"] == []
    assert payload["status"] == "MODEL_NOT_VALIDATED"
    assert payload["cash_weight"] == 1.0
    assert payload["horizon"] == 3
    assert "STAGE5_NOT_ALLOWED" in payload["reason_codes"]
