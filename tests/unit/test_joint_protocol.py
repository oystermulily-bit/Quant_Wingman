"""S2 admission, durable reservations, and strict serialization boundaries."""
from dataclasses import replace
import json
import sqlite3

import pytest

from factor_research.joint_protocol import (
    FrozenJointPlan, RunLedger, canonical_fingerprint,
)


def _frozen(directory, **overrides):
    ids = ("low_vol", "weak_a", "weak_b")
    values = dict(
        hypothesis_id="S2_TEST_HYPOTHESIS", design_approved=True,
        execution_frozen=True, feature_ids=ids,
        feature_manifest_hash=canonical_fingerprint(ids),
        feature_contract_hash="0" * 64, signal_output="model_prediction",
        data_fingerprint="a" * 64, split_fingerprint="b" * 64,
        config_fingerprint="c" * 64, implementation_hash="d" * 64,
        risk_policy_hash="e" * 64, concentration_policy_hash="f" * 64,
        freeze_id="TEST_ONLY_FREEZE", ledger_directory=str(directory.resolve()),
        max_trials_per_scope=12, max_total_trials=60,
    )
    values.update(overrides)
    return FrozenJointPlan(**values)


def test_draft_authorization_never_implies_live_research():
    draft = FrozenJointPlan(design_approved=True)
    result = draft.validate(mode="synthetic")
    assert result["status"] == "ENGINEERING_ONLY"
    assert result["research_evidence"] is False
    with pytest.raises(ValueError, match="execution_frozen"):
        draft.validate(mode="development")


def test_live_admission_has_no_d21_requirement_and_no_release_permission(tmp_path):
    admitted = _frozen(tmp_path / "new").validate(mode="development")
    assert admitted["status"] == "JOINT_RESEARCH_ADMITTED"
    for key in ("research_passed", "production_allowed", "holdout_allowed",
                "holdout_read", "sota_allowed", "sota_promoted", "stage5_allowed",
                "research_evidence", "evidence_generated"):
        assert admitted[key] is False


@pytest.mark.parametrize("name", [
    "hypothesis_id", "freeze_id", "ledger_directory", "feature_manifest_hash",
    "feature_contract_hash", "signal_output",
    "data_fingerprint", "split_fingerprint", "config_fingerprint", "implementation_hash",
    "risk_policy_hash", "concentration_policy_hash",
])
def test_live_required_field_missing_fails_closed(tmp_path, name):
    plan = replace(_frozen(tmp_path / "new"), **{name: ""})
    with pytest.raises(ValueError, match=name):
        plan.validate(mode="development")


@pytest.mark.parametrize("mode", ["holdout", "production", "sota", "Development", None])
def test_non_development_roles_rejected(mode):
    with pytest.raises(ValueError, match="only synthetic or development"):
        FrozenJointPlan().validate(mode=mode)


@pytest.mark.parametrize("field,value", [
    ("design_approved", 1), ("execution_frozen", "true"),
    ("max_trials_per_scope", True), ("max_total_trials", 10.0),
    ("max_outer_scopes", 0), ("max_trials_per_scope", -1),
    ("horizon", True), ("signal_output", "auto_select"),
    ("feature_ids", ["a"]), ("feature_ids", ("a", "a")),
    ("data_fingerprint", "bad"), ("implementation_hash", "A" * 64),
    ("hypothesis_id", None), ("risk_policy_hash", None),
])
def test_structural_types_are_strict(field, value):
    with pytest.raises(ValueError):
        FrozenJointPlan(**{field: value})


def test_manifest_and_whole_config_hashes_are_binding(tmp_path):
    plan = _frozen(tmp_path / "new")
    with pytest.raises(ValueError, match="ordered feature_ids"):
        replace(plan, feature_ids=tuple(reversed(plan.feature_ids)))
    changed = replace(plan, config_fingerprint="9" * 64)
    assert changed.fingerprint != plan.fingerprint
    assert canonical_fingerprint({"a": 1, "b": 2}) == canonical_fingerprint({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        canonical_fingerprint({"bad": float("nan")})


def test_plan_json_roundtrip_rejects_unknown_or_coerced_types(tmp_path):
    plan = _frozen(tmp_path / "new")
    value = json.loads(json.dumps(plan.to_dict()))
    assert FrozenJointPlan.from_dict(value) == plan
    with pytest.raises(ValueError, match="unknown plan fields"):
        FrozenJointPlan.from_dict({**value, "holdout_allowed": True})
    with pytest.raises(ValueError, match="boolean"):
        FrozenJointPlan.from_dict({**value, "execution_frozen": 1})
    with pytest.raises(ValueError, match="array"):
        FrozenJointPlan.from_dict({**value, "feature_ids": "abc"})


def test_live_plan_is_bound_to_one_ledger_directory(tmp_path):
    plan = _frozen(tmp_path / "canonical")
    other = RunLedger(tmp_path / "other")
    with pytest.raises(ValueError, match="differs from the frozen plan"):
        other.begin_run(plan, "run", mode="development")
    with pytest.raises(ValueError, match="absolute"):
        replace(plan, ledger_directory="relative").validate()


def test_live_requires_five_outer_scopes(tmp_path):
    plan = _frozen(tmp_path / "new", max_outer_scopes=2, max_total_trials=24)
    plan.validate(mode="synthetic")
    with pytest.raises(ValueError, match="five"):
        plan.validate(mode="development")


def test_signal_output_is_not_silently_selected_by_draft_or_json():
    assert FrozenJointPlan().signal_output == ""
    assert FrozenJointPlan.from_dict({"design_approved": True}).signal_output == ""


def test_ledger_refuses_unrelated_existing_directory(tmp_path):
    directory = tmp_path / "unrelated"
    directory.mkdir()
    (directory / "user_data.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated"):
        RunLedger(directory)
    assert (directory / "user_data.txt").read_text(encoding="utf-8") == "keep"
    assert not (directory / RunLedger.filename).exists()


def test_same_plan_run_or_freeze_cannot_restart_after_reopen(tmp_path):
    directory = tmp_path / "ledger"
    plan = _frozen(directory)
    ledger = RunLedger(directory)
    ledger.begin_run(plan, "one", mode="development")
    reopened = RunLedger(directory)
    with pytest.raises(ValueError, match="cannot restart"):
        reopened.begin_run(plan, "two", mode="development")
    with pytest.raises(ValueError, match="cannot restart"):
        reopened.begin_run(replace(plan, freeze_id="different"), "one", mode="development")
    with pytest.raises(ValueError, match="cannot restart"):
        reopened.begin_run(replace(plan, config_fingerprint="9" * 64), "three", mode="development")
    assert reopened.snapshot("one")["status"] == "RUNNING"


def test_failed_scope_keeps_entire_reserved_budget(tmp_path):
    directory = tmp_path / "ledger"
    plan = _frozen(directory, max_total_trials=18)
    ledger = RunLedger(directory)
    ledger.begin_run(plan, "run", mode="development")
    ledger.reserve_scope("run", "outer-0", 12)
    outcome = ledger.complete_scope("run", "outer-0", 2, success=False)
    assert outcome["charged_trials"] == 12
    snapshot = ledger.snapshot("run")
    assert snapshot["actual_trials"] == 2
    assert snapshot["charged_trials"] == 12
    assert snapshot["remaining_trials"] == 6
    with pytest.raises(ValueError, match="total budget"):
        ledger.reserve_scope("run", "outer-1", 12)
    with pytest.raises(ValueError, match="scope already reserved"):
        ledger.reserve_scope("run", "outer-0", 1)
    with pytest.raises(ValueError, match="failed"):
        ledger.finish_run("run", success=True)
    assert ledger.finish_run("run", success=False)["status"] == "FAILED"


def test_crashed_scope_is_persistent_and_not_retryable(tmp_path):
    directory = tmp_path / "ledger"
    ledger = RunLedger(directory)
    ledger.begin_run(_frozen(directory), "run", mode="development")
    ledger.reserve_scope("run", "outer-0", 12)
    reopened = RunLedger(directory)
    assert reopened.snapshot("run")["charged_trials"] == 12
    with pytest.raises(ValueError, match="already reserved"):
        reopened.reserve_scope("run", "outer-0", 12)
    with pytest.raises(ValueError, match="unfinished"):
        reopened.finish_run("run")


def test_complete_five_scopes_is_economics_pending_not_go(tmp_path):
    directory = tmp_path / "ledger"
    ledger = RunLedger(directory)
    ledger.begin_run(_frozen(directory), "run", mode="development")
    for number in range(5):
        ledger.reserve_scope("run", f"outer-{number}", 12)
        ledger.complete_scope("run", f"outer-{number}", 3)
    with pytest.raises(ValueError, match="scope count"):
        ledger.reserve_scope("run", "outer-5", 1)
    result = ledger.finish_run("run")
    assert result["status"] == "ECONOMIC_VALIDATION_PENDING"
    assert result["actual_trials"] == 15
    assert result["charged_trials"] == 60
    assert not result["research_passed"] and not result["stage5_allowed"]
    with pytest.raises(ValueError, match="no longer active"):
        ledger.reserve_scope("run", "extra", 1)


def test_partial_live_outer_evidence_cannot_finish_successfully(tmp_path):
    directory = tmp_path / "ledger"
    ledger = RunLedger(directory)
    ledger.begin_run(_frozen(directory), "run", mode="development")
    ledger.reserve_scope("run", "outer-0", 12)
    ledger.complete_scope("run", "outer-0", 1)
    with pytest.raises(ValueError, match="all five outer scopes"):
        ledger.finish_run("run")


def test_modified_stored_budget_fails_fingerprint_check(tmp_path):
    directory = tmp_path / "ledger"
    ledger = RunLedger(directory)
    plan = _frozen(directory)
    ledger.begin_run(plan, "run", mode="development")
    # Simulate corruption or a manual edit; never accept the edited allowance.
    corrupted = plan.to_dict()
    corrupted["max_trials_per_scope"] = 24
    corrupted["max_total_trials"] = 120
    connection = sqlite3.connect(ledger.path)
    try:
        connection.execute("UPDATE runs SET plan_json = ? WHERE run_id = ?",
                           (json.dumps(corrupted), "run"))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ledger.reserve_scope("run", "outer-0", 12)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ledger.snapshot("run")


def test_scope_counters_and_state_transitions_are_strict(tmp_path):
    ledger = RunLedger(tmp_path / "new")
    ledger.begin_run(FrozenJointPlan(), "synthetic")
    with pytest.raises(ValueError, match="positive integer"):
        ledger.reserve_scope("synthetic", "outer-0", True)
    with pytest.raises(ValueError, match="per-scope"):
        ledger.reserve_scope("synthetic", "outer-0", 481)
    ledger.reserve_scope("synthetic", "outer-0", 10)
    for value in (True, -1, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            ledger.complete_scope("synthetic", "outer-0", value)
    with pytest.raises(ValueError, match="exceeds reservation"):
        ledger.complete_scope("synthetic", "outer-0", 11)
    with pytest.raises(ValueError, match="boolean"):
        ledger.complete_scope("synthetic", "outer-0", 1, success=1)
    ledger.complete_scope("synthetic", "outer-0", 0)
    with pytest.raises(ValueError, match="already completed"):
        ledger.complete_scope("synthetic", "outer-0", 0)
    result = ledger.finish_run("synthetic")
    assert result["status"] == "ENGINEERING_COMPLETED"
    json.dumps(result, allow_nan=False)
