import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from factor_research.demo_artifacts import (
    ARTIFACT_KIND, COMPLETION_STATUS, FALSE_FLAGS, validate_demo_input_path, write_demo_bundle,
)


def example_scores():
    return pd.DataFrame({
        "date": ["2020-01-02"] * 3,
        "code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "score": [0.2, np.nan, np.nan],
        "score_available_at": ["2020-01-02T21:00:00+08:00"] * 3,
        "fold_id": ["outer_1"] * 3,
        "is_member": [True, True, False],
        "signal_valid": [True, False, False],
    })


def example_report():
    return {
        "run_id": "test_synthetic_1", "data_role": "synthetic", "status": COMPLETION_STATUS,
        **{key: False for key in FALSE_FLAGS},
        "fingerprints": {key: hashlib.sha256(key.encode()).hexdigest()
                         for key in ("model", "formula", "fold", "data")},
        "expected_score_rows": 3,
        "selected_features": ["MOM_20", "LOW_VOL_20"],
        "formulas": [],
        "model_evidence": {"estimator": "fixed_lightgbm", "fit_scope": "outer_train_only"},
    }


def test_bundle_roundtrip_preserves_missing_members_and_nonmembers(tmp_path):
    scores = example_scores()
    original = scores.copy(deep=True)
    paths = write_demo_bundle(tmp_path / "offline_demo_smoke", example_report(), scores)
    manifest = json.loads((tmp_path / "offline_demo_smoke" / "manifest.json").read_text("utf-8"))
    saved = pd.read_parquet(paths["scores"])
    assert manifest["artifact_kind"] == ARTIFACT_KIND
    assert manifest["member_rows"] == 2
    assert manifest["valid_member_rows"] == 1
    assert manifest["member_coverage"] == 0.5
    assert len(saved) == 3 and saved.score.isna().sum() == 2
    assert str(saved.score_available_at.dt.tz) == "UTC"
    assert saved.score_available_at.iloc[0].hour == 13
    assert not any(manifest[key] for key in FALSE_FLAGS)
    assert set(paths) == {"scores", "selected_features", "formulas", "model_evidence", "manifest"}
    for filename, detail in manifest["files"].items():
        assert hashlib.sha256((tmp_path / "offline_demo_smoke" / filename).read_bytes()).hexdigest() == detail["sha256"]
    pd.testing.assert_frame_equal(scores, original)


@pytest.mark.parametrize("flag", FALSE_FLAGS)
def test_forbids_any_promotable_flag(tmp_path, flag):
    report = example_report()
    report[flag] = True
    with pytest.raises(ValueError, match=flag):
        write_demo_bundle(tmp_path / "offline_demo_forbidden", report, example_scores())
    assert not (tmp_path / "offline_demo_forbidden").exists()


@pytest.mark.parametrize("field,value", [
    ("data_role", "holdout"), ("data_role", "test"), ("status", "GO"),
    ("expected_score_rows", 2), ("expected_score_rows", True),
    ("selected_features", []), ("selected_features", ["x", "x"]),
    ("formulas", None), ("model_evidence", {}),
    ("fingerprints", {"model": "a" * 64}),
    ("schema_version", "2.0"),
])
def test_rejects_invalid_report(tmp_path, field, value):
    report = example_report()
    report[field] = value
    with pytest.raises(ValueError):
        write_demo_bundle(tmp_path / "offline_demo_forbidden", report, example_scores())


def test_rejects_nested_trading_payload(tmp_path):
    report = example_report()
    report["model_evidence"]["target_weights"] = {"000001.SZ": 0.5}
    with pytest.raises(ValueError, match="trading field"):
        write_demo_bundle(tmp_path / "offline_demo_forbidden", report, example_scores())


@pytest.mark.parametrize("name", ["holdout/offline_demo_x", "sealed/offline_demo_x", "stage4r_old/offline_demo_x", "stage3r_old", "normal_run", "offline_demo_"])
def test_rejects_sensitive_or_non_demo_destinations(tmp_path, name):
    with pytest.raises(ValueError):
        write_demo_bundle(tmp_path / name, example_report(), example_scores())


def test_rejects_existing_directory_and_never_overwrites(tmp_path):
    destination = tmp_path / "offline_demo_existing"
    destination.mkdir()
    old = destination / "manifest.json"
    old.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_demo_bundle(destination, example_report(), example_scores())
    assert old.read_text("utf-8") == "original"


@pytest.mark.parametrize("column,value", [
    ("score_available_at", "2020-01-02T21:00:00"),
    ("score_available_at", "2020-01-03T21:00:00+08:00"),
    ("date", "2020-01-02T01:00:00"),
    ("date", "2020-01-02T00:00:00+08:00"),
    ("code", ""), ("fold_id", ""), ("score", np.inf),
    ("signal_valid", False), ("is_member", 1),
])
def test_rejects_unsafe_scores(tmp_path, column, value):
    scores = example_scores()
    scores[column] = scores[column].astype(object)
    scores.loc[0, column] = value
    with pytest.raises(ValueError):
        write_demo_bundle(tmp_path / "offline_demo_forbidden", example_report(), scores)


def test_rejects_extra_score_columns(tmp_path):
    scores = example_scores().assign(target_weight=0.05)
    with pytest.raises(ValueError, match="seven"):
        write_demo_bundle(tmp_path / "offline_demo_forbidden", example_report(), scores)


def test_rejects_injected_nonmember_score(tmp_path):
    scores = example_scores()
    scores.loc[2, "score"] = 0.7
    with pytest.raises(ValueError, match="nonmember"):
        write_demo_bundle(tmp_path / "offline_demo_forbidden", example_report(), scores)


def test_rejects_duplicate_grid_rows(tmp_path):
    scores = example_scores()
    scores.loc[1, "code"] = scores.loc[0, "code"]
    with pytest.raises(ValueError, match="duplicate"):
        write_demo_bundle(tmp_path / "offline_demo_forbidden", example_report(), scores)


def test_parquet_failure_never_leaves_completed_manifest(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("parquet failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    destination = tmp_path / "offline_demo_partial"
    with pytest.raises(RuntimeError):
        write_demo_bundle(destination, example_report(), example_scores())
    assert destination.is_dir()
    assert not (destination / "manifest.json").exists()


def test_input_path_only_resolves_explicit_prepared_development_file(tmp_path):
    development = tmp_path / "development_panel.npz"
    development.write_bytes(b"not opened by path validator")
    assert validate_demo_input_path(development) == development.resolve()


@pytest.mark.parametrize("name", ["panel.npz", "holdout_development.npz", "sealed_synthetic.json", "development.csv"])
def test_input_path_rejects_unscoped_or_forbidden_file(tmp_path, name):
    source = tmp_path / name
    source.write_bytes(b"not opened")
    with pytest.raises(ValueError):
        validate_demo_input_path(source)
