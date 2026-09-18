"""Array-only formula bridge checks; no Holdout, network or credentials."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from factor_research.demo_formulas import (
    FEEDBACK_RANK_SEMANTICS, NAT_NS, FormulaLimits, FormulaSpec, FormulaValidationError, OfflineReplaySource,
    RestrictedRDFormulaSource, evaluate_formula, sanitise_inner_feedback, validate_formula,
)


def panel():
    x = {"X": np.array([[1., 2.], [2., 0.], [3., np.nan]]),
         "RES_MOM_120_20": np.array([[3., 4.], [2., 5.], [1., 6.]])}
    times = np.array([100, 200], dtype=np.int64)
    availability = {k: np.broadcast_to(times - 10, (3, 2)).copy() for k in x}
    return x, availability, times, np.ones((3, 2), dtype=bool)


def test_custom_context_feature_not_limited_to_stack_vm():
    x, a, t, m = panel()
    result = evaluate_formula(FormulaSpec("f", ("X", "RES_MOM_120_20", "SUB")), x, a, t, m)
    np.testing.assert_allclose(result.values[:, 0], [-2, 0, 2])
    assert result.dependencies == ("RES_MOM_120_20", "X")
    assert np.isnan(result.values[2, 1])
    assert result.available_at_ns[2, 1] == NAT_NS
    np.testing.assert_array_equal(result.valid_mask, np.isfinite(result.values))


@pytest.mark.parametrize("tokens", [("X", "NEG"), ("X", "ABS"),
                                    ("X", "RES_MOM_120_20", "MUL"),
                                    ("X", "RES_MOM_120_20", "ADD")])
def test_nan_infects_all_operations(tokens):
    x, a, t, m = panel()
    a["X"][0, 0] = 101  # Same-date but after the exact signal instant.
    result = evaluate_formula(FormulaSpec("f", tokens), x, a, t, m)
    assert np.isnan(result.values[0, 0])
    assert np.isnan(result.values[2, 1])
    assert result.available_at_ns[0, 0] == NAT_NS


def test_division_zero_overflow_and_missing_times_remain_nan():
    x, a, t, m = panel()
    result = evaluate_formula(FormulaSpec("f", ("RES_MOM_120_20", "X", "DIV")), x, a, t, m)
    assert np.isnan(result.values[1, 1])
    x["X"][0, 0] = 1e308
    result = evaluate_formula(FormulaSpec("f", ("X", "X", "MUL")), x, a, t, m)
    assert np.isnan(result.values[0, 0])
    a["X"][1, 0] = NAT_NS
    result = evaluate_formula(FormulaSpec("f", ("X",)), x, a, t, m)
    assert np.isnan(result.values[1, 0])


def test_exact_time_allowed_later_one_ns_masked():
    x, a, t, m = panel()
    a["X"][0, 0] = t[0]
    a["X"][1, 0] = t[0] + 1
    result = evaluate_formula(FormulaSpec("f", ("X",)), x, a, t, m)
    assert result.values[0, 0] == 1
    assert np.isnan(result.values[1, 0])


def test_rank_membership_ties_and_aggregate_availability():
    x, a, t, m = panel()
    x["X"][:, 0] = [1., 1., 9.]
    a["X"][:, 0] = [80, 99, 100]
    m[2, 0] = False
    result = evaluate_formula(FormulaSpec("f", ("X", "CS_RANK")), x, a, t, m)
    np.testing.assert_allclose(result.values[:2, 0], [.75, .75])
    assert np.isnan(result.values[2, 0])
    np.testing.assert_array_equal(result.available_at_ns[:2, 0], [99, 99])
    assert result.available_at_ns[2, 0] == NAT_NS


def test_late_peer_never_changes_current_cross_section():
    x, a, t, m = panel()
    a["X"][2, 0] = 101
    first = evaluate_formula(FormulaSpec("f", ("X", "CS_RANK")), x, a, t, m)
    x["X"][2, 0] = -1e200
    second = evaluate_formula(FormulaSpec("f", ("X", "CS_RANK")), x, a, t, m)
    np.testing.assert_allclose(first.values, second.values, equal_nan=True)
    np.testing.assert_allclose(first.values[:2, 0], [.5, 1])


@pytest.mark.parametrize("tokens", [(), ("ADD",), ("X", "X"), ("Y",),
                                    ("X", "SHIFT_MINUS_1"), ("X", "eval"),
                                    ("C:/data/labels.parquet",), (42,)])
def test_reject_invalid_or_unsafe_structure(tokens):
    with pytest.raises(FormulaValidationError):
        validate_formula(FormulaSpec("f", tokens), ("X",))


def test_explicit_complexity_limits():
    spec = FormulaSpec("f", ("X", "NEG", "ABS"))
    for limits in (FormulaLimits(max_tokens=2), FormulaLimits(max_depth=2),
                   FormulaLimits(max_operators=1)):
        with pytest.raises(FormulaValidationError):
            validate_formula(spec, ("X",), limits=limits)


@pytest.mark.parametrize("field", ["max_tokens", "max_depth", "max_operators"])
def test_limits_reject_boolean(field):
    with pytest.raises(ValueError):
        FormulaLimits(**{field: True})


def test_input_arrays_unmodified_and_only_dependencies_required():
    x, a, t, m = panel()
    x_before, a_before = x["X"].copy(), a["X"].copy()
    del a["RES_MOM_120_20"]
    evaluate_formula(FormulaSpec("f", ("X", "NEG")), x, a, t, m)
    np.testing.assert_allclose(x_before, x["X"], equal_nan=True)
    np.testing.assert_array_equal(a_before, a["X"])


@pytest.mark.parametrize("change", ["numeric_members", "float_availability", "missing_availability",
                                   "float_signal", "unordered_signal", "shape"])
def test_bad_input_contract_fails_closed(change):
    x, a, t, m = panel()
    if change == "numeric_members": m = m.astype(int)
    if change == "float_availability": a["X"] = a["X"].astype(float)
    if change == "missing_availability": del a["X"]
    if change == "float_signal": t = t.astype(float)
    if change == "unordered_signal": t = t[::-1]
    if change == "shape": x["X"] = x["X"][:, :1]
    with pytest.raises(FormulaValidationError):
        evaluate_formula(FormulaSpec("f", ("X",)), x, a, t, m)


def test_replay_is_deterministic_not_llm_and_preserves_invalid_trials():
    one = OfflineReplaySource(("X", "RES_MOM_120_20"))
    two = OfflineReplaySource(("X", "RES_MOM_120_20"))
    assert one.generate(round_index=0, count=7) == two.generate(round_index=0, count=7)
    assert all(spec.source == "offline_replay" for spec in one.generate(round_index=1, count=7))
    assert one.audit_records[0]["network_called"] is False
    replay = OfflineReplaySource(("X",), replay=[("NOT_ALLOWED",), ("X", "NEG"), ("X", "NEG")])
    candidates = replay.generate(round_index=0, count=3)
    assert len(candidates) == 3  # Invalid and duplicates are NOT silently discarded.
    assert candidates[1].tokens == candidates[2].tokens
    with pytest.raises(FormulaValidationError):
        validate_formula(candidates[0], ("X",))


def test_small_default_budget_interleaves_joint_formulas_and_is_bounded():
    names = ("X", "Y", "Z")
    source = OfflineReplaySource(names)
    first = source.generate(round_index=0, count=4)
    second = source.generate(round_index=1, count=4)
    assert first[0].tokens == ("X", "CS_RANK")
    assert first[1].tokens == ("X", "CS_RANK", "Y", "CS_RANK", "MUL")
    assert len(first) == len(second) == 4
    assert all(candidate.source == "offline_replay" for candidate in first + second)
    assert all(validate_formula(candidate, names) for candidate in first + second)
    assert set(candidate.formula_id for candidate in first).isdisjoint(
        candidate.formula_id for candidate in second)
    assert source.generate(round_index=100, count=4) == []
    assert source.audit_records[0]["feedback_rank_semantics"] == FEEDBACK_RANK_SEMANTICS


def test_single_feature_replay_does_not_invent_second_dependency():
    source = OfflineReplaySource(("X",))
    proposals = source.generate(round_index=0, count=4)
    assert [candidate.tokens for candidate in proposals] == [
        ("X", "CS_RANK"), ("X", "NEG"), ("X", "ABS")]


def test_feedback_whitelist_blocks_paths_raw_data_and_outer_fields():
    feedback = [{"formula": ["X", "NEG"], "rank": 1, "status": "inner_elite",
                 "outer_score": .9, "date": "2026-01-01", "labels_path": "secret"},
                {"formula": ["secret/path"], "rank": 1},
                {"formula": ["X"], "rank": 1, "status": "secret/path"},
                {"formula": ["X"], "rank": True}]
    result = sanitise_inner_feedback(feedback, ("X",))
    assert result == [{"formula": ["X", "NEG"], "rank": 1, "status": "inner_elite"}]
    assert "secret" not in json.dumps(result)


def test_new_sources_do_not_inherit_feedback_or_outer_scope():
    old = OfflineReplaySource(("X",))
    old.generate(round_index=0, count=1, feedback=[{"formula": ["X"], "rank": 1}])
    fresh = OfflineReplaySource(("X",))
    fresh.generate(round_index=0, count=1)
    assert fresh.audit_records[0]["inner_feedback"] == []
    for role in ("development_outer", "holdout", "development"):
        with pytest.raises(FormulaValidationError):
            fresh.generate(round_index=0, count=1, data_role=role)


def test_network_source_requires_explicit_opt_in_without_importing_transport():
    with pytest.raises(FormulaValidationError, match="explicit"):
        RestrictedRDFormulaSource(("X",), project_root="irrelevant", model="configured")
    # Construction alone never reads settings/credentials or calls a service.
    source = RestrictedRDFormulaSource(("X",), project_root="irrelevant", model="configured",
                                       allow_network=True)
    assert source.audit_records == []


def test_mock_rd_transport_prompt_only_selected_vocab_and_safe_feedback(monkeypatch):
    captured = {}

    class FakeRD:
        def __init__(self, *args, **kwargs): pass

        @staticmethod
        def _extract_json(value): return json.loads(value)

        def generate(self, **kwargs):
            captured["system"] = self._system_prompt()
            captured["user"] = self._user_prompt(**kwargs)
            rows = self._extract_json('{"formulas":[{"tokens":["X","NEG"]},'
                                      '{"tokens":["UNKNOWN"]},null,{"tokens":["X","NEG"]}]}')
            return [self._validate(row["tokens"]) for row in rows["formulas"]]

    fake = SimpleNamespace(RDFormulaGenerator=FakeRD, RDFormulaLimits=lambda: None,
                           FormulaCandidate=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, "model_core.rd_agent_generator", fake)
    source = RestrictedRDFormulaSource(("X",), project_root="never/read", model="test", allow_network=True)
    candidates = source.generate(round_index=0, count=4,
                                  feedback=[{"formula": ["X"], "rank": 1,
                                             "labels_path": "secret", "outer_score": 9}])
    assert len(candidates) == 4
    assert candidates[1].tokens == ("UNKNOWN",)
    assert candidates[2].tokens == ()
    assert candidates[0].tokens == candidates[3].tokens
    assert all(x.source == "rd_agent_network" for x in candidates)
    assert "secret" not in captured["user"]
    assert "outer_score" not in captured["user"]
    assert '"features": ["X"]' in captured["system"]
    assert "never/read" not in captured["system"] + captured["user"]
    assert json.loads(captured["user"])["feedback_rank_semantics"] == FEEDBACK_RANK_SEMANTICS
    assert "not feature importance" in captured["system"]
    assert source.audit_records[0]["feedback_rank_semantics"] == FEEDBACK_RANK_SEMANTICS
