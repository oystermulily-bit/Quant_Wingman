from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_research.group_evaluator import SelectionFold
from factor_research.joint_runner import JointPanel, JointResearchRunner, OuterFold
from factor_research.lgbm_baseline import FixedLightGBMBaseline


def fixture_panel():
    rng = np.random.default_rng(19)
    n, t = 16, 260
    features = {name: rng.normal(size=(n, t)) for name in ("a", "b")}
    times = pd.date_range("2015-01-01", periods=t, tz="UTC")
    y = features["a"] + .01*rng.normal(size=(n, t))
    # pandas may store date ranges in microseconds; Timestamp.value is ns.
    ns = np.array([x.value for x in times], dtype=np.int64)
    available = {k: np.broadcast_to(ns, (n, t)).copy() for k in features}
    return JointPanel(features, y, np.ones((n, t), dtype=bool),
                      tuple(f"stock{i}" for i in range(n)), tuple(x.isoformat() for x in times),
                      tuple(x.isoformat() for x in times+pd.Timedelta(days=5)),
                      available, "2017-01-01T00:00:00Z", data_role="synthetic")


def fixture_folds():
    inner = (SelectionFold(0, 40, 60, 80), SelectionFold(0, 70, 90, 110),
             SelectionFold(0, 100, 120, 140))
    return (OuterFold("outer1", 0, 160, 180, 200, inner),
            OuterFold("outer2", 0, 210, 230, 250, inner))


@pytest.fixture
def fake_model(monkeypatch):
    calls = []

    def fit(self, x, y, train, valid):
        calls.append({"shape": y.shape, "target": y.copy(), "train": train.copy(), "valid": valid.copy()})
        if x is None:
            return np.full(len(valid), y.ravel()[train].mean())
        flat = x.reshape(-1, x.shape[-1])
        design = np.column_stack((np.ones(len(flat)), flat))
        weights = np.linalg.lstsq(design[train], y.ravel()[train], rcond=None)[0]
        return design[valid] @ weights

    monkeypatch.setattr(FixedLightGBMBaseline, "_fit_predict", fit)
    return calls


def test_nested_selection_outer_predictions_and_non_go(fake_model):
    panel, folds = fixture_panel(), fixture_folds()
    result = JointResearchRunner().run_synthetic(panel, folds, signal_output="model_prediction")
    assert result["status"] == "SYNTHETIC_ENGINEERING_ONLY"
    assert result["economic_status"] == "ECONOMIC_VALIDATION_PENDING"
    assert not any(result[x] for x in ("research_go", "holdout_read", "production_allowed", "sota_promoted"))
    assert all("a" in f["selection"]["selected"] for f in result["folds"])
    assert all(f["outer_rank_ic"] > .95 for f in result["folds"])
    assert result["predictions"].shape == panel.target.shape
    assert np.isnan(result["predictions"][:, :180]).all()
    assert np.isfinite(result["predictions"][:, 180:200]).all()
    # Outer model receives only training labels, never test/purge/embargo labels.
    for call in fake_model:
        if call["shape"][1] in (200, 250):
            forbidden = np.ones(call["target"].size, dtype=bool)
            forbidden[call["train"]] = False
            assert np.isnan(call["target"].ravel()[forbidden]).all()
    final_fit = next(c for c in fake_model if c["shape"][1] == 250)
    assert not any(200 <= idx % 250 < 205 for idx in final_fit["train"])


def test_future_outer_labels_do_not_change_selection_or_predictions(fake_model):
    panel, folds = fixture_panel(), fixture_folds()
    runner = JointResearchRunner()
    first = runner.run_synthetic(panel, folds, signal_output="model_prediction")
    changed = panel.target.copy()
    changed[:, 230:250] *= -1000
    second = runner.run_synthetic(replace(panel, target=changed), folds, signal_output="model_prediction")
    assert [x["selection"]["selected"] for x in first["folds"]] == [x["selection"]["selected"] for x in second["folds"]]
    np.testing.assert_allclose(first["predictions"], second["predictions"], equal_nan=True)
    assert first["folds"][1]["outer_rank_ic"] > 0 > second["folds"][1]["outer_rank_ic"]


def test_outer_features_not_used_for_training_cleanup(fake_model):
    panel = fixture_panel()
    changed = {k: v.copy() for k, v in panel.features.items()}
    changed["b"][:, 230:250] = 999
    r = JointResearchRunner()
    a = r.run_synthetic(panel, fixture_folds(), signal_output="model_prediction")
    b = r.run_synthetic(replace(panel, features=changed), fixture_folds(), signal_output="model_prediction")
    assert [x["selection"]["excluded_features"] for x in a["folds"]] == [x["selection"]["excluded_features"] for x in b["folds"]]
    assert [x["selection"]["selected"] for x in a["folds"]] == [x["selection"]["selected"] for x in b["folds"]]


def test_late_and_missing_availability_remain_missing(fake_model):
    panel = fixture_panel()
    available = {k: v.copy() for k, v in panel.available_at_ns.items()}
    available["a"][0, 180] += 1  # Same date, but one nanosecond too late.
    available["b"][1, 180] = np.iinfo(np.int64).min
    r = JointResearchRunner().run_synthetic(replace(panel, available_at_ns=available), fixture_folds(),
                                          signal_output="model_prediction")
    assert np.isnan(r["predictions"][:2, 180]).all()
    assert r["unavailable_member_days"] == {"a": 1, "b": 1}
    assert r["folds"][0]["member_days"] == 16*20


@pytest.mark.parametrize("bad", ["holdout", "naive", "missing_availability", "overlapping_outer", "inner_leak", "labels_leak", "ambiguous_output"])
def test_fail_closed(fake_model, bad):
    panel, folds = fixture_panel(), fixture_folds()
    output = "model_prediction"
    if bad == "holdout":
        panel = replace(panel, holdout_start=panel.signal_at[240])
    elif bad == "naive":
        panel = replace(panel, signal_at=tuple(s[:19] for s in panel.signal_at))
    elif bad == "missing_availability":
        panel = replace(panel, available_at_ns={})
    elif bad == "overlapping_outer":
        folds = (folds[0], replace(folds[1], test_start=190))
    elif bad == "inner_leak":
        folds = (replace(folds[0], inner_folds=folds[0].inner_folds+(SelectionFold(0, 170, 180, 195),)), folds[1])
    elif bad == "labels_leak":
        labels = list(panel.label_end_at)
        labels[0] = panel.signal_at[181]
        panel = replace(panel, label_end_at=tuple(labels))
    else:
        output = None
    with pytest.raises(ValueError):
        JointResearchRunner().run_synthetic(panel, folds, signal_output=output)


def test_real_panel_cannot_use_synthetic_entry(fake_model):
    with pytest.raises(ValueError, match="synthetic entry"):
        JointResearchRunner().run_synthetic(replace(fixture_panel(), data_role="development"),
                                            fixture_folds(), signal_output="model_prediction")


def test_equal_weight_is_explicit_and_different_output_mode(fake_model):
    r = JointResearchRunner().run_synthetic(fixture_panel(), fixture_folds(), signal_output="equal_weight")
    assert r["signal_output"] == "equal_weight"
    finite = r["predictions"][np.isfinite(r["predictions"])]
    assert (finite > 0).all() and (finite <= 1).all()


def test_insufficient_outer_support_does_not_delete_denominator(fake_model):
    p = fixture_panel()
    p.features["b"][:4, 180:200] = np.nan
    with pytest.raises(ValueError, match="outer signal coverage"):
        JointResearchRunner().run_synthetic(p, fixture_folds(), signal_output="model_prediction")


def test_no_candidate_is_not_reported_as_validated_model(monkeypatch):
    def constant(self, x, y, train, valid):
        return np.zeros(len(valid))
    monkeypatch.setattr(FixedLightGBMBaseline, "_fit_predict", constant)
    r = JointResearchRunner().run_synthetic(fixture_panel(), fixture_folds(), signal_output="model_prediction")
    assert all(f["status"] == "NO_CANDIDATE" for f in r["folds"])
    assert np.isnan(r["predictions"]).all()
    assert not r["research_go"]
    assert r["prediction_status"] == "NO_JOINT_CANDIDATE"
    assert r["usable_outer_folds"] == 0


def test_joint_pair_reaches_outer_test_without_singleton_gate(monkeypatch):
    p = fixture_panel()
    p = replace(p, target=p.features["a"]*p.features["b"])

    def interaction(self, x, y, train, valid):
        if x is None or x.shape[2] < 2:
            return np.zeros(len(valid))
        flat = x.reshape(-1, x.shape[-1])
        return flat[valid, 0]*flat[valid, 1]

    monkeypatch.setattr(FixedLightGBMBaseline, "_fit_predict", interaction)
    r = JointResearchRunner().run_synthetic(p, fixture_folds(), signal_output="model_prediction",
                                          pair_seeds=(("a", "b"),))
    for fold in r["folds"]:
        assert fold["selection"]["selected"] == ("a", "b")
        assert fold["selection"]["transitions"][0]["phase"] == "pair"
        assert fold["outer_rank_ic"] == pytest.approx(1.)


def test_unfrozen_live_plan_is_denied_before_model_or_ledger(tmp_path):
    from factor_research.joint_protocol import FrozenJointPlan
    plan = FrozenJointPlan(design_approved=True)
    with pytest.raises(ValueError, match="admission denied"):
        JointResearchRunner().run_development(replace(fixture_panel(), data_role="development"),
                                              fixture_folds(), plan=plan, run_id="not-started")
    assert not list(tmp_path.iterdir())


def test_plan_checker_does_not_start_training_or_create_ledger(tmp_path):
    import json
    from factor_research.joint_protocol import FrozenJointPlan
    from scripts.check_joint_research_plan import check_plan
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(FrozenJointPlan(design_approved=True).to_dict()), encoding="utf-8")
    result = check_plan(path)
    assert result["status"] == "PLAN_CHECK_BLOCKED"
    assert not result["training_started"] and not result["research_go"]
    assert sorted(x.name for x in tmp_path.iterdir()) == ["draft.json"]


def frozen_engineering_inputs(tmp_path):
    """Fake data exercises live admission; this is not an economic experiment."""
    from factor_research.joint_protocol import FrozenJointPlan, canonical_fingerprint
    p = fixture_panel()
    indices = np.arange(300) % 16
    manifest_hash = canonical_fingerprint(("a", "b"))
    p = replace(p, features={k: v[indices].copy() for k, v in p.features.items()},
                target=p.target[indices].copy(), member_mask=p.member_mask[indices].copy(),
                available_at_ns={k: v[indices].copy() for k, v in p.available_at_ns.items()},
                codes=tuple(f"fake{i}" for i in range(300)), data_role="development",
                feature_manifest_hash=manifest_hash, feature_contract_hash="c"*64)
    inner = fixture_folds()[0].inner_folds
    folds = tuple(OuterFold(f"outer{i}", 0, c-20, c, c+10, inner)
                  for i, c in enumerate((180, 195, 210, 225, 240)))
    runner = JointResearchRunner()
    plan = FrozenJointPlan(hypothesis_id="ENGINEERING_FAKE_DATA", design_approved=True,
                           execution_frozen=True, feature_ids=("a", "b"),
                           feature_manifest_hash=manifest_hash, feature_contract_hash=p.feature_contract_hash,
                           data_fingerprint=runner._prepare(p)[-1], split_fingerprint=runner.split_identity(folds),
                           config_fingerprint=runner.config_identity(signal_output="model_prediction", pair_seeds=()),
                           implementation_hash=runner.implementation_identity(), signal_output="model_prediction",
                           freeze_id="ENGINEERING_TEST_ONLY", risk_policy_hash="a"*64,
                           concentration_policy_hash="b"*64, ledger_directory=str(tmp_path.resolve()))
    return p, folds, runner, plan


def test_live_admission_and_persistent_budget_connection_on_fake_data(fake_model, monkeypatch, tmp_path):
    p, folds, runner, plan = frozen_engineering_inputs(tmp_path)
    monkeypatch.setattr(FixedLightGBMBaseline, "_module", staticmethod(lambda: object()))
    result = runner.run_development(p, folds, plan=plan, run_id="fake-only")
    assert result["status"] == "JOINT_OUTER_PREDICTIONS_READY"
    assert result["economic_status"] == "ECONOMIC_VALIDATION_PENDING"
    assert len(result["budget_ledger"]["scopes"]) == 5
    assert not result["research_go"]
    with pytest.raises(ValueError, match="already registered"):
        runner.run_development(p, folds, plan=plan, run_id="another-name")


@pytest.mark.parametrize("field", ["feature_contract_hash", "feature_manifest_hash", "data_fingerprint",
                                  "split_fingerprint", "config_fingerprint", "implementation_hash", "budget"])
def test_live_fingerprint_or_budget_mismatch_stops_before_training(fake_model, tmp_path, field):
    p, folds, runner, plan = frozen_engineering_inputs(tmp_path)
    if field in ("feature_contract_hash", "feature_manifest_hash"):
        p = replace(p, **{field: "e"*64})
    elif field == "budget":
        plan = replace(plan, max_total_trials=480)
    else:
        plan = replace(plan, **{field: "e"*64})
    with pytest.raises(ValueError):
        runner.run_development(p, folds, plan=plan, run_id="blocked")
    assert fake_model == []
    assert not list(tmp_path.iterdir())


def test_real_lightgbm_nested_smoke():
    pytest.importorskip("lightgbm")
    r = JointResearchRunner().run_synthetic(fixture_panel(), fixture_folds(), signal_output="model_prediction")
    assert r["status"] == "SYNTHETIC_ENGINEERING_ONLY"
    assert not r["research_go"]
