"""Import, isolation, immutable selection and HTTP workflow regression checks."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.portfolio_api import get_portfolio_service, router
from web.portfolio_service import PortfolioError, PortfolioService, SCORE_COLUMNS, STATUS


def _write_bundle(root: Path, *, role="synthetic", count=3, invalid=True, name="offline_demo_fixture") -> Path:
    folder = root / name
    folder.mkdir()
    rows = []
    for day in ("2020-01-02", "2020-01-03"):
        for i in range(count):
            valid = not (invalid and i == count - 1)
            rows.append({"date": pd.Timestamp(day), "code": f"{i+1:06d}.SZ",
                         "score": (0.25 if i < 2 else 0.1) if valid else None,
                         "score_available_at": pd.Timestamp(day + "T13:00:00Z"), "fold_id": "outer_1",
                         "is_member": True, "signal_valid": valid})
    frame = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    frame.to_parquet(folder / "scores.parquet", index=False)
    report = {"schema_version": "1.0", "artifact_kind": "OFFLINE_DEMO_SCORE_BUNDLE", "run_id": "demo_fixture",
              "status": STATUS, "data_role": role, "research_go": False, "holdout_read": False,
              "sota_promoted": False, "production_allowed": False, "whitelist_applied": False,
              "fingerprints": {key: sha256(key.encode()).hexdigest() for key in ("model", "formula", "fold", "data")},
              "selected_features": ["factor_A"], "formulas": [], "model_evidence": {"estimator": "test_fixture"},
              "expected_score_rows": len(rows), "score_rows": len(rows), "score_columns": SCORE_COLUMNS,
              "member_rows": len(rows), "valid_member_rows": int(frame.signal_valid.sum()),
              "member_coverage": float(frame.signal_valid.mean()), "files": {}, "warnings": []}
    for filename, value in (("selected_features.json", report["selected_features"]),
                            ("formulas.json", report["formulas"]), ("model_evidence.json", report["model_evidence"])):
        (folder / filename).write_text(json.dumps(value), encoding="utf-8")
    for filename in ("scores.parquet", "selected_features.json", "formulas.json", "model_evidence.json"):
        report["files"][filename] = {"sha256": sha256((folder / filename).read_bytes()).hexdigest()}
    report["files"]["scores.parquet"]["rows"] = len(rows)
    path = folder / "manifest.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _change_manifest(path, edit):
    report = json.loads(path.read_text(encoding="utf-8"))
    edit(report)
    path.write_text(json.dumps(report), encoding="utf-8")


def _rewrite_scores(path, edit):
    target = path.parent / "scores.parquet"
    frame = pd.read_parquet(target)
    edit(frame)
    frame.to_parquet(target, index=False)
    _change_manifest(path, lambda report: report["files"]["scores.parquet"].update(sha256=sha256(target.read_bytes()).hexdigest()))


@pytest.fixture
def service(tmp_path):
    return PortfolioService(workspace=tmp_path, default_bundles=())


@pytest.fixture
def workflow(tmp_path, service):
    path = _write_bundle(tmp_path)
    bundle = service.import_bundle(str(path))
    snapshot = service.get_scores(bundle["bundle_id"], "2020-01-02", "outer_1")
    return path, bundle, snapshot


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_portfolio_service] = lambda: service
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client


PREFIX = "/api/v2/offline-portfolio"


def test_sort_preserves_missing_members_ties_codes_and_synthetic_identity(service, workflow):
    path, bundle, snapshot = workflow
    assert len(snapshot["rows"]) == 3
    assert [row["code"] for row in snapshot["rows"]] == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert [row["rank"] for row in snapshot["rows"]] == [1, 2, None]
    assert snapshot["rows"][-1]["score"] is None
    assert snapshot["is_complete_universe"] is False
    assert "SYNTHETIC_DATA_NOT_MARKET_EVIDENCE" in snapshot["warnings"]
    assert snapshot["production_allowed"] is False
    assert service.import_bundle(str(path)) == bundle
    assert len(service.list_bundles()["bundles"]) == 1


def test_snapshot_survives_source_changes_restart_and_is_date_specific(tmp_path, service, workflow):
    path, bundle, snapshot = workflow
    other = service.get_scores(bundle["bundle_id"], "2020-01-03", "outer_1")
    assert other["snapshot_id"] != snapshot["snapshot_id"]
    (path.parent / "scores.parquet").write_bytes(b"changed source")
    fresh = PortfolioService(workspace=tmp_path, default_bundles=())
    assert fresh.get_snapshot(snapshot["snapshot_id"]) == snapshot
    with pytest.raises(PortfolioError, match="SHA-256"):
        fresh.import_bundle(str(path))


@pytest.mark.parametrize("edit", [
    lambda m: m.update(production_allowed=True),
    lambda m: m.update(holdout_read=0),
    lambda m: m.update(status="VALIDATED"),
    lambda m: m.update(expected_score_rows=1),
    lambda m: m.update(member_rows=999),
    lambda m: m.update(valid_member_rows=999),
    lambda m: m.update(member_coverage=1.0),
    lambda m: m["fingerprints"].update(data="0" * 64),
    lambda m: m.update(selected_features=[]),
    lambda m: m.update(target_weight=0.1),
    lambda m: m["files"].update({"../scores.parquet": m["files"].pop("scores.parquet")}),
])
def test_rejects_malformed_or_promoted_bundles(tmp_path, service, edit):
    path = _write_bundle(tmp_path)
    _change_manifest(path, edit)
    with pytest.raises(PortfolioError):
        service.import_bundle(str(path))
    assert service.list_bundles()["bundles"] == []


@pytest.mark.parametrize("edit", [
    lambda f: f.__setitem__("signal_valid", True),
    lambda f: f.__setitem__("code", 1),
    lambda f: f.__setitem__("extra_label", 0.02),
    lambda f: f.__setitem__("score_available_at", pd.Timestamp("2020-01-02T13:00:00")),
    lambda f: f.__setitem__("score", float("inf")),
    lambda f: (f.__setitem__("date", pd.Timestamp("2024-08-26")),
               f.__setitem__("score_available_at", pd.Timestamp("2024-08-26T13:00:00Z"))),
])
def test_rejects_bad_score_schema_timing_and_flags(tmp_path, service, edit):
    path = _write_bundle(tmp_path)
    _rewrite_scores(path, edit)
    with pytest.raises(PortfolioError):
        service.import_bundle(str(path))


def test_development_requires_complete_300_members_per_period(tmp_path, service):
    incomplete = _write_bundle(tmp_path, role="development", name="offline_demo_small")
    with pytest.raises(PortfolioError, match="300"):
        service.import_bundle(str(incomplete))
    full = _write_bundle(tmp_path, role="development", count=300, name="offline_demo_full")
    bundle = service.import_bundle(str(full))
    snapshot = service.get_scores(bundle["bundle_id"], "2020-01-02", "outer_1")
    assert snapshot["member_count"] == 300 and snapshot["valid_count"] == 299
    assert snapshot["is_complete_universe"] is True
    assert len(snapshot["rows"]) == 300
    assert "SYNTHETIC_DATA_NOT_MARKET_EVIDENCE" not in snapshot["warnings"]


def test_rejects_external_and_sealed_paths_before_file_read(tmp_path, service, monkeypatch):
    forbidden = tmp_path / "sealed" / "offline_demo_x" / "manifest.json"
    reads = []
    monkeypatch.setattr(service, "_read", lambda *args: reads.append(args))
    with pytest.raises(PortfolioError, match="Holdout"):
        service.import_bundle(str(forbidden))
    with pytest.raises(PortfolioError):
        service.import_bundle(str(tmp_path.parent / "manifest.json"))
    assert not reads


def test_file_path_in_manifest_is_never_followed(tmp_path, service):
    path = _write_bundle(tmp_path)
    _change_manifest(path, lambda m: m["files"]["scores.parquet"].update(path="../../sealed/data.parquet"))
    assert service.import_bundle(str(path))["score_rows"] == 6


def test_whitelist_versions_are_immutable_validate_subset_and_detect_conflicts(service, workflow):
    snapshot = workflow[2]
    first = service.save_whitelist(snapshot["snapshot_id"], "我的选择", ["000001.SZ"])
    second = service.save_whitelist(snapshot["snapshot_id"], "调整选择", ["000002.SZ"], first["whitelist_id"], 1)
    assert second["version"] == 2
    assert service.get_whitelist(first["whitelist_id"], 1) == first
    assert service.get_whitelist(first["whitelist_id"]) == second
    assert service.list_whitelists()["whitelists"] == [second]
    with pytest.raises(PortfolioError) as conflict:
        service.save_whitelist(snapshot["snapshot_id"], "过期编辑", [], first["whitelist_id"], 1)
    assert conflict.value.status_code == 409
    with pytest.raises(PortfolioError):
        service.save_whitelist(snapshot["snapshot_id"], "错误成员", ["NO_SUCH_STOCK"])
    with pytest.raises(PortfolioError):
        service.save_whitelist(snapshot["snapshot_id"], "重复成员", ["000001.SZ", "000001.SZ"])


def test_http_full_flow_restores_snapshot_version_and_plan(client, workflow):
    snapshot = workflow[2]
    assert client.get(PREFIX + "/scores", params={"bundle_id": snapshot["bundle_id"]}).status_code == 422
    saved = client.post(PREFIX + "/whitelists", json={"snapshot_id": snapshot["snapshot_id"], "name": "演示选择",
                                                       "codes": ["000001.SZ", "000003.SZ"]})
    assert saved.status_code == 200
    whitelist = saved.json()
    restored = client.get(PREFIX + "/snapshots/" + whitelist["snapshot_id"])
    assert restored.json() == snapshot
    assert client.get(PREFIX + "/whitelists").json()["whitelists"] == [whitelist]
    response = client.post(PREFIX + "/plans", json={"whitelist_id": whitelist["whitelist_id"], "whitelist_version": 1})
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "SELECTED_SCORE_MISSING"
    assert client.get(PREFIX + "/analysis",params={"snapshot_id":snapshot["snapshot_id"]}).json()["available"] is False


def test_post_requires_json_and_local_same_origin(client, workflow):
    request = {"snapshot_id": workflow[2]["snapshot_id"], "name": "名单", "codes": []}
    for origin in ("https://evil.example", "null", "http://localhost:8765", "http://127.0.0.1:8000"):
        response = client.post(PREFIX + "/whitelists", json=request, headers={"Origin": origin})
        assert response.status_code == 403
    good = client.post(PREFIX + "/whitelists", json=request, headers={"Origin": "http://127.0.0.1:8765"})
    assert good.status_code == 200
    bad_type = client.post(PREFIX + "/whitelists", content=json.dumps(request), headers={"Content-Type": "text/plain"})
    assert bad_type.status_code == 415


def test_api_rejects_coercion_and_redacts_path_errors(client, tmp_path):
    bad = client.post(PREFIX + "/plans", json={"whitelist_id": "x", "whitelist_version": True, "market_budget": "1"})
    assert bad.status_code == 422
    error = client.post(PREFIX + "/bundles/import", json={"manifest_path": str(tmp_path / "missing" / "manifest.json")})
    assert error.status_code == 400
    assert str(tmp_path) not in error.text


def test_default_registration_is_explicit_not_directory_discovery(tmp_path):
    first = _write_bundle(tmp_path, name="offline_demo_first")
    _write_bundle(tmp_path, name="offline_demo_unrequested")
    service = PortfolioService(workspace=tmp_path, default_bundles=(str(first),))
    assert len(service.list_bundles()["bundles"]) == 1
