"""CLI and input-boundary checks use only temporary synthetic payloads."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from scripts import run_offline_factor_demo as cli
from factor_research.joint_runner import JointResearchRunner


def prepared_bundle(tmp_path):
    panel, folds = cli.synthetic_bundle()
    directory = tmp_path / "development_fixture"
    directory.mkdir()
    manifest = {
        "data_role": "development", "codes": list(panel.codes),
        "feature_ids": list(panel.features), "signal_at": list(panel.signal_at),
        "label_end_at": list(panel.label_end_at), "holdout_start": panel.holdout_start,
        "feature_contract_hash": "a" * 64, "arrays_filename": "development_arrays.npz",
        "outer_folds": [asdict(fold) for fold in folds], "label_horizon_bars": 5,
    }
    (directory / "development_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    arrays = {"target": panel.target, "member_mask": panel.member_mask}
    arrays.update({"feature__" + name: value for name, value in panel.features.items()})
    arrays.update({"available__" + name: value for name, value in panel.available_at_ns.items()})
    np.savez(directory / "development_arrays.npz", **arrays)
    return directory, manifest, arrays


def replace_manifest(directory, manifest):
    (directory / "development_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_synthetic_shapes_and_explicit_nanoseconds():
    panel, folds = cli.synthetic_bundle()
    prepared = JointResearchRunner._prepare(panel)
    JointResearchRunner()._validate_folds(folds, panel.target.shape[1], prepared[2], prepared[3])
    assert panel.data_role == "synthetic"
    assert panel.target.shape == (16, 320)
    assert set(panel.features) == {"A", "B", "C"}
    assert all(name.startswith("SYNTHETIC_") for name in panel.codes)
    assert panel.available_at_ns["A"].dtype == np.dtype("int64")
    assert panel.available_at_ns["A"][0, 0] == prepared[2][0]
    assert all(value == 0 for value in prepared[5].values())


def test_load_prepared_development_roundtrip(tmp_path):
    directory, _, arrays = prepared_bundle(tmp_path)
    panel, folds = cli.load_development_bundle(directory)
    assert panel.data_role == "development"
    assert panel.feature_contract_hash == "a" * 64
    assert len(panel.feature_manifest_hash) == 64
    np.testing.assert_array_equal(panel.target, arrays["target"])
    assert len(folds) == 2 and all(len(f.inner_folds) == 3 for f in folds)


@pytest.mark.parametrize("case", ["signal_boundary", "label_boundary", "role", "naive",
                                  "traversal", "absolute", "unknown", "duplicate", "fold_leak",
                                  "moved_holdout", "trusted_signal_boundary", "trusted_label_boundary",
                                  "missing_horizon", "wrong_horizon", "bool_horizon", "float_horizon"])
def test_metadata_rejected_before_numeric_payload_is_opened(tmp_path, monkeypatch, case):
    directory, manifest, _ = prepared_bundle(tmp_path)
    if case == "signal_boundary":
        manifest["holdout_start"] = manifest["signal_at"][-1]
    elif case == "label_boundary":
        manifest["label_end_at"][0] = manifest["holdout_start"]
    elif case == "role":
        manifest["data_role"] = "holdout"
    elif case == "naive":
        manifest["signal_at"][0] = "2015-01-01T13:00:00"
    elif case == "traversal":
        manifest["arrays_filename"] = "../development_arrays.npz"
    elif case == "absolute":
        manifest["arrays_filename"] = "C:\\sealed\\data.npz"
    elif case == "unknown":
        manifest["automatic_data_source"] = "unused"
    elif case == "duplicate":
        manifest["feature_ids"].append("A")
    elif case == "fold_leak":
        manifest["outer_folds"][0]["inner_folds"][0]["val_end"] = 181
    elif case == "moved_holdout":
        # Even old input data does not permit a later caller-declared boundary.
        manifest["holdout_start"] = "2030-01-01T00:00:00Z"
    elif case == "trusted_signal_boundary":
        manifest["holdout_start"] = cli.TRUSTED_DEVELOPMENT_END
        manifest["signal_at"][-1] = cli.TRUSTED_DEVELOPMENT_END
        manifest["label_end_at"][-1] = "2024-09-02T00:00:00+08:00"
    elif case == "trusted_label_boundary":
        manifest["holdout_start"] = cli.TRUSTED_DEVELOPMENT_END
        manifest["label_end_at"][-1] = cli.TRUSTED_DEVELOPMENT_END
    elif case == "missing_horizon":
        del manifest["label_horizon_bars"]
    elif case == "wrong_horizon":
        manifest["label_horizon_bars"] = 3
    elif case == "bool_horizon":
        manifest["label_horizon_bars"] = True
    elif case == "float_horizon":
        manifest["label_horizon_bars"] = 5.0
    replace_manifest(directory, manifest)

    def forbidden_load(*args, **kwargs):
        pytest.fail("numeric payload must not be opened after invalid metadata")

    monkeypatch.setattr(cli.np, "load", forbidden_load)
    with pytest.raises(ValueError):
        cli.load_development_bundle(directory)


@pytest.mark.parametrize("case", ["extra", "numeric_members", "bad_availability", "bad_shape", "object"])
def test_numeric_contract_rejections(tmp_path, case):
    directory, _, arrays = prepared_bundle(tmp_path)
    if case == "extra":
        arrays["undeclared"] = np.array([1.0])
    elif case == "numeric_members":
        arrays["member_mask"] = arrays["member_mask"].astype(int)
    elif case == "bad_availability":
        arrays["available__A"] = arrays["available__A"].astype(float)
    elif case == "bad_shape":
        arrays["target"] = arrays["target"][:, :-1]
    elif case == "object":
        arrays["feature__A"] = arrays["feature__A"].astype(object)
    np.savez(directory / "development_arrays.npz", **arrays)
    with pytest.raises(ValueError):
        cli.load_development_bundle(directory)


@pytest.mark.parametrize("name", ["holdout", "sealed_cache", "my_HOLDOUT_copy"])
def test_sensitive_path_rejected_without_any_open(tmp_path, monkeypatch, name):
    monkeypatch.setattr(cli.json, "loads", lambda *a, **kw: pytest.fail("must not read manifest"))
    with pytest.raises(ValueError, match="Holdout/sealed"):
        cli.load_development_bundle(tmp_path / name)


def test_manifest_symlink_cannot_escape_bundle(tmp_path):
    directory, _, _ = prepared_bundle(tmp_path)
    manifest_path = directory / "development_manifest.json"
    outside = tmp_path / "other_manifest.json"
    manifest_path.replace(outside)
    try:
        manifest_path.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit test symlinks")
    with pytest.raises(ValueError, match="escape|symlinks"):
        cli.load_development_bundle(directory)


def test_existing_output_is_rejected_before_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run_pipeline", lambda *a: pytest.fail("must not compute"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--synthetic", "--output", str(tmp_path)])
    assert exc.value.code == 2


def test_cli_passes_synthetic_role_and_reports_nonvalidation(tmp_path, monkeypatch, capsys):
    def run(panel, folds, *, config):
        assert panel.data_role == "synthetic" and len(folds) == 2
        assert config.proposal_mode == "offline_replay"
        assert config.allow_network is False and config.rd_model == ""
        return {"status": "OFFLINE_DEMO_COMPLETED_NOT_VALIDATED"}, "fake_test_scores"

    def write(output, report, scores):
        assert output == (tmp_path / "demo").resolve()
        assert scores == "fake_test_scores"
        return {"report": str(output / "demo_report.json")}

    monkeypatch.setattr(cli, "_run_pipeline", run)
    monkeypatch.setattr(cli, "write_demo_bundle", write)
    assert cli.main(["--synthetic", "--output", str(tmp_path / "demo")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data_role"] == "synthetic"
    assert result["holdout_read"] is False
    assert result["production_allowed"] is False
    assert result["sota_promoted"] is False


@pytest.mark.parametrize("options", [
    ["--proposal-mode", "rd_agent"],
    ["--proposal-mode", "rd_agent", "--rd-model", "explicit-test-model"],
    ["--proposal-mode", "rd_agent", "--allow-network"],
    ["--proposal-mode", "rd_agent", "--allow-network", "--rd-model", "   "],
    ["--allow-network"],
    ["--rd-model", "unused-model"],
])
def test_remote_mode_requires_complete_opt_in_before_data_load(tmp_path, monkeypatch, options):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid proposal configuration must stop before reading arrays or running models")

    monkeypatch.setattr(cli, "load_development_bundle", forbidden)
    monkeypatch.setattr(cli, "_run_pipeline", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--output", str(tmp_path / "demo"), "--development-bundle",
                  str(tmp_path / "development_not_opened"), *options])
    assert exc.value.code == 2


def test_explicit_remote_config_is_forwarded_without_using_network(tmp_path, monkeypatch):
    calls = []

    def fake_run(panel, folds, *, config):
        calls.append(config)
        assert config.proposal_mode == "rd_agent" and config.allow_network is True
        assert config.rd_model == "explicit-test-model"
        return {"status": "OFFLINE_DEMO_COMPLETED_NOT_VALIDATED"}, "fake_test_scores"

    monkeypatch.setattr(cli, "_run_pipeline", fake_run)
    monkeypatch.setattr(cli, "write_demo_bundle", lambda *args: {"report": "test-only"})
    assert cli.main(["--synthetic", "--output", str(tmp_path / "demo"),
                     "--proposal-mode", "rd_agent", "--allow-network",
                     "--rd-model", "explicit-test-model"]) == 0
    assert len(calls) == 1


@pytest.mark.parametrize("args", [[], ["--synthetic", "--development-bundle", "x"],
                                  ["--allow-holdout"], ["--whitelist", "x"]])
def test_cli_has_no_ambiguous_or_unsafe_mode(tmp_path, args):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--output", str(tmp_path / "demo"), *args])
    assert exc.value.code == 2
