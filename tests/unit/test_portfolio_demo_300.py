"""Boundaries and full-grid integrity of the explicitly synthetic UI fixture."""
import numpy as np
import pandas as pd
import pytest

from factor_research.joint_runner import JointResearchRunner
from scripts import build_portfolio_demo_300 as demo


def test_synthetic_universe_reproducible_and_time_isolated():
    panel, folds = demo.synthetic_bundle()
    repeat, repeat_folds = demo.synthetic_bundle()
    assert panel.data_role == "synthetic"
    assert panel.codes == tuple(f"SYNTHETIC_{i:03d}" for i in range(300))
    assert panel.target.shape == (300, 320)
    assert repeat_folds == folds
    np.testing.assert_array_equal(panel.target, repeat.target)
    for name, values in panel.features.items():
        np.testing.assert_array_equal(values, repeat.features[name])
        assert np.isnan(values).sum() == 1
        assert np.isnan(values[-1, folds[-1].test_end - 1])
    prepared = JointResearchRunner._prepare(panel)
    JointResearchRunner()._validate_folds(folds, panel.target.shape[1], prepared[2], prepared[3])
    assert panel.member_mask.all()
    assert all(np.isfinite(panel.features[name][:, :folds[0].train_end]).all()
               for name in panel.features)


def sample_grid():
    panel, folds = demo.synthetic_bundle()
    records = []
    for fold in folds:
        for i in range(fold.test_start, fold.test_end):
            day = pd.Timestamp(panel.signal_at[i]).date().isoformat()
            for code in panel.codes:
                valid = not (code == panel.codes[-1] and i == folds[-1].test_end - 1)
                records.append({"date": day, "fold_id": fold.fold_id, "code": code,
                                "is_member": True, "signal_valid": valid,
                                "score": 1.0 if valid else np.nan})
    return panel, folds, pd.DataFrame(records)


def test_missing_member_stays_in_each_300_row_period():
    panel, folds, scores = sample_grid()
    demo.validate_full_grid(scores, panel, folds)
    assert len(scores) == 12000
    assert scores["signal_valid"].sum() == 11999


@pytest.mark.parametrize("case", ["drop_missing", "replace_member", "drop_date", "fill_missing", "nonmember"])
def test_full_grid_rejects_loss_or_fabrication(case):
    panel, folds, scores = sample_grid()
    if case == "drop_missing":
        scores = scores[scores.signal_valid]
    elif case == "replace_member":
        scores.loc[0, "code"] = "OTHER_SYNTHETIC"
    elif case == "drop_date":
        scores = scores[scores.date != scores.date.iloc[0]]
    elif case == "fill_missing":
        scores.loc[~scores.signal_valid, "score"] = 0
        scores["signal_valid"] = True
    else:
        scores.loc[0, "is_member"] = False
    with pytest.raises(ValueError):
        demo.validate_full_grid(scores, panel, folds)


@pytest.mark.parametrize("name", ["existing", "wrong_prefix", "offline_demo_holdout", "stage9/offline_demo_new"])
def test_invalid_output_rejected_before_fitting(tmp_path, monkeypatch, name):
    output = tmp_path / name
    if name == "existing":
        output.mkdir()
    monkeypatch.setattr(demo, "_run_pipeline", lambda *a: pytest.fail("must not run fitting"))
    with pytest.raises(SystemExit) as exc:
        demo.main(["--output", str(output)])
    assert exc.value.code == 2


def test_budget_fixed_offline_and_covers_both_folds():
    config = demo.demo_config()
    assert config.proposal_mode == "offline_replay"
    assert config.allow_network is False and config.rd_model == ""
    assert config.seed == demo.SEED
    reserved = 2 * (config.formula_rounds * (config.proposals_per_round + config.search_trials_per_round + 1) + 1)
    assert reserved <= config.max_total_opportunities
