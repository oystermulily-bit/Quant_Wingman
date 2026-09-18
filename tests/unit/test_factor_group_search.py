from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_research.group_search import GroupEvaluation, GroupSearchConfig, StepwiseGroupSearch
from factor_research.group_evaluator import (
    FixedLightGBMGroupEvaluator, SelectionFold, clean_training_candidates,
)


def scorer(values, *, coverage=1.0):
    def evaluate(group):
        score = values.get(group, 0.0)
        return GroupEvaluation("ctx", "sample", (score,) * 3, coverage)
    return evaluate


def search(values, candidates=("a", "b", "c"), config=None, **kwargs):
    return StepwiseGroupSearch(config).run(
        list(candidates), scorer(values), data_role="development_inner", **kwargs,
    )


def test_forward_rechecks_candidates_and_backward_removes_redundancy():
    result = search({(): 0, ("a",): .02, ("b",): .018, ("c",): .01,
                     ("a", "b"): .03, ("a", "c"): .022,
                     ("b", "c"): .04, ("a", "b", "c"): .0402})
    assert result.selected == ("b", "c")
    trials = [x for x in result.trials if x["phase"] == "forward"]
    assert any(x["baseline"] == () and x["candidate"] == ("b",) for x in trials)
    assert any(x["baseline"] == ("a",) and x["candidate"] == ("a", "b") for x in trials)
    assert result.transitions[-1]["phase"] == "backward"
    assert not result.sota_promoted and not result.stage5_allowed


def test_atomic_pair_can_pass_when_both_singletons_fail():
    result = search({("a", "b"): .02}, pair_seeds=(("b", "a"),))
    assert result.selected == ("a", "b")
    assert result.transitions[0]["phase"] == "pair"


def test_pair_uses_two_factor_slots():
    cfg = GroupSearchConfig(max_group_size=1)
    result = search({("a", "b"): .02}, config=cfg, pair_seeds=(("a", "b"),))
    assert result.selected == ()
    assert not any(x["phase"] == "pair" for x in result.trials)


def test_cap_counts_initial_columns_and_backward_runs_at_cap():
    cfg = GroupSearchConfig(max_group_size=2)
    result = search({("a", "b"): .02, ("a",): .0199}, config=cfg, initial=("a", "b"))
    assert result.selected == ("a",)
    with pytest.raises(ValueError, match="initial group exceeds"):
        search({}, config=cfg, initial=("a", "b", "c"))


def test_deletion_tolerance_cannot_accumulate():
    result = search({("a", "b", "c"): .03, ("a", "b"): .0296, ("a",): .0292},
                    initial=("a", "b", "c"))
    assert result.selected == ("a", "b")


@pytest.mark.parametrize("role", ["holdout", "development_outer", "development"])
def test_search_rejects_non_inner_roles(role):
    with pytest.raises(ValueError, match="development_inner"):
        StepwiseGroupSearch().run(["a"], scorer({}), data_role=role)


def test_budget_does_not_pick_an_arbitrary_prefix():
    cfg = GroupSearchConfig(forward_budget=2)
    result = search({("a",): .1}, config=cfg)
    assert not result.trials
    assert result.selected == ()
    assert result.stop_reason == "FORWARD_BUDGET_EXHAUSTED"


def test_total_budget_and_failed_evaluations_are_counted():
    def bad(group):
        if group:
            raise RuntimeError("test failure")
        return GroupEvaluation("ctx", "sample", (0., 0., 0.), 1.)
    result = StepwiseGroupSearch(GroupSearchConfig(max_trials=3)).run(
        ["a", "b", "c"], bad, data_role="development_inner", pair_seeds=(("a", "b"),))
    assert len(result.trials) == 3
    assert all("error" in x for x in result.trials)
    assert result.evaluation_calls == 4  # One initial baseline + three trials.
    assert result.stop_reason == "PAIR_BUDGET_EXHAUSTED"
    assert result.status == "INNER_SELECTION_INCOMPLETE"


def test_missing_backend_is_not_reported_as_no_increment():
    def evaluate(group):
        if group:
            raise RuntimeError("required model backend unavailable")
        return GroupEvaluation("ctx", "sample", (0.,) * 3, 1.)
    result = StepwiseGroupSearch().run(["a"], evaluate, data_role="development_inner")
    assert result.stop_reason == "EVALUATION_ERRORS"
    assert result.status == "INNER_SELECTION_INCOMPLETE"


@pytest.mark.parametrize("change", ["sample", "context", "folds", "coverage", "nan"])
def test_fail_closed_when_comparison_evidence_changes(change):
    def evaluate(group):
        value = GroupEvaluation("ctx", "sample", (.02,) * 3, 1.)
        if group:
            return {"sample": replace(value, sample_id="other"),
                    "context": replace(value, context_id="other"),
                    "folds": replace(value, fold_scores=(.1,) * 4),
                    "coverage": replace(value, coverage=.99),
                    "nan": replace(value, fold_scores=(float("nan"),) * 3)}[change]
        return value
    result = StepwiseGroupSearch().run(["a"], evaluate, data_role="development_inner")
    assert result.selected == ()
    assert "error" in result.trials[0]


def test_positive_fold_requirement_and_coverage():
    def evaluate(group):
        return GroupEvaluation("ctx", "sample", (.1, -.01, -.01) if group else (0.,) * 3, 1.)
    result = StepwiseGroupSearch().run(["a"], evaluate, data_role="development_inner")
    assert result.selected == ()
    result = StepwiseGroupSearch().run(["a"], scorer({}, coverage=.84), data_role="development_inner")
    assert result.stop_reason == "BASELINE_SUPPORT_INSUFFICIENT"
    assert not result.trials


def test_deterministic_tie_break_cache_and_no_cycles():
    result = search({("a",): .02, ("b",): .02, ("a", "b"): .03,
                     ("b", "c"): .031, ("a", "b", "c"): .0312})
    assert result.transitions[0]["after"] == ("a",)
    assert result.cache_hits > 0
    states = [x["after"] for x in result.transitions]
    assert len(states) == len(set(states))
    assert len(result.trials) <= 480


def test_invalid_pair_and_config_rejected():
    with pytest.raises(ValueError, match="two distinct"):
        search({}, pair_seeds=(("a", "a"),))
    with pytest.raises(ValueError, match="candidate manifest"):
        search({}, candidates=tuple(str(x) for x in range(66)))
    with pytest.raises(ValueError, match="add threshold"):
        GroupSearchConfig(min_add_delta=.0004)


def test_training_cleaning_does_not_use_future_output_or_labels():
    features = {"a": np.array([[1., 2., 3., 4.]]),
                "b": np.array([[1., 2., 99., 99.]]),
                "c": np.array([[2., 2., 3., 4.]]),
                "d": np.array([[np.nan] * 4])}
    kept, excluded = clean_training_candidates(features, np.array([[True, True, False, False]]))
    assert list(kept) == ["a"]
    assert excluded["b"] == "training_output_duplicate_of:a"
    assert "c" in excluded and "d" in excluded


def evaluator_inputs():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 220))
    dates = pd.date_range("2020-01-01", periods=220, tz="UTC")
    return dict(
        features={"a": x, "b": rng.normal(size=x.shape)}, target=x.copy(),
        member_mask=np.ones(x.shape, dtype=bool),
        folds=(SelectionFold(0, 40, 60, 80), SelectionFold(0, 100, 120, 140),
               SelectionFold(0, 160, 180, 200)),
        signal_at=dates, label_end_at=dates + pd.Timedelta(days=5),
        selection_end_at="2020-09-01T00:00:00Z", holdout_start="2021-01-01T00:00:00Z",
        data_fingerprint="synthetic-only", data_role="development_inner",
    )


def test_evaluator_uses_same_support_and_original_member_denominator(monkeypatch):
    inputs = evaluator_inputs()
    inputs["features"]["b"][0, :] = np.nan
    evaluator = FixedLightGBMGroupEvaluator(**inputs)
    captured = []

    def predict(features, target, train, valid):
        captured.append((train.copy(), valid.copy()))
        return np.zeros(len(valid)) if features is None else features.reshape(-1, features.shape[2])[valid, 0]

    monkeypatch.setattr(evaluator.model, "_fit_predict", predict)
    base, candidate = evaluator(()), evaluator(("a",))
    assert candidate.mean_score > base.mean_score
    assert base.sample_id == candidate.sample_id
    assert base.coverage == pytest.approx(11 / 12)
    for first, second in zip(captured[:3], captured[3:]):
        assert all(np.array_equal(x, y) for x, y in zip(first, second))
    assert not any(140 <= index % 220 < 145 for index in captured[2][0])  # Embargo.
    inputs["features"]["a"][:] = -99
    assert not np.all(evaluator.x[:, :, 0] == -99)


@pytest.mark.parametrize("bad", ["role", "boundary", "naive", "overlap", "late_label", "short_cross_section"])
def test_evaluator_rejects_unsafe_or_insufficient_inputs(bad):
    inputs = evaluator_inputs()
    if bad == "role":
        inputs["data_role"] = "holdout"
    elif bad == "boundary":
        inputs["selection_end_at"] = "2020-07-01T00:00:00Z"
    elif bad == "naive":
        inputs["signal_at"] = inputs["signal_at"].tz_localize(None)
    elif bad == "overlap":
        inputs["folds"] = (SelectionFold(0, 40, 60, 80), SelectionFold(0, 40, 70, 90))
    elif bad == "late_label":
        ends = list(inputs["label_end_at"])
        ends[0] = inputs["signal_at"][65]
        inputs["label_end_at"] = ends
    else:
        inputs["features"]["b"][:3, 60] = np.nan
    with pytest.raises(ValueError):
        FixedLightGBMGroupEvaluator(**inputs)


def test_real_lightgbm_group_search_on_synthetic_panel():
    pytest.importorskip("lightgbm")
    evaluator = FixedLightGBMGroupEvaluator(**evaluator_inputs())
    result = StepwiseGroupSearch().run(list(evaluator.names), evaluator, data_role="development_inner")
    assert "a" in result.selected
    assert not result.sota_promoted
