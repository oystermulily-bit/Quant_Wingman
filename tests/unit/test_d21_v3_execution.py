from dataclasses import asdict

import pytest

from research_d21_v2.protocol import D21V2Config
from research_d21_v2.runner import Stage3RResearchRunner
from research_d21_v3_execution import D21V3Config, Stage3V3Runner, Stage4V3Runner


def test_economic_and_execution_parameters_unchanged():
    old, new = asdict(D21V2Config()), asdict(D21V3Config())
    changed = {key for key in old if old[key] != new[key]}
    assert changed == {'hypothesis_id', 'signal_version', 'stage3r_protocol',
                       'stage4r_protocol', 'loo_universe', 'min_oof_coverage'}
    assert new['min_oof_coverage'] == .85 and old['min_oof_coverage'] == .95
    assert Stage3RResearchRunner.completion_status == 'STAGE3R_D21_V2_COMPLETED'


def test_ablation_only_changes_b_d_signal_version():
    for old, new in zip(Stage3RResearchRunner.experiments, Stage3V3Runner.experiments):
        changed = {key for key in old if old[key] != new[key]}
        assert changed == ({'signal_version'} if old['experiment_id'] in ['B', 'D'] else set())


@pytest.mark.parametrize('passed,expected', [(True, 'HORIZON_FEASIBLE_5D_V3'), (False, 'D21_V3_FAILED')])
def test_v3_status_never_opens_rd_agent(passed, expected):
    gate = Stage4V3Runner.classify({'passed': passed}, True)
    assert gate['statistical_gate'] == expected
    assert not gate['rd_agent_allowed'] and not gate['holdout_read'] and not gate['production_eligible']


def test_coverage_remains_fail_closed():
    assert Stage4V3Runner.classify({'passed': True}, False)['statistical_gate'] == 'INSUFFICIENT_EVIDENCE'
    assert Stage4V3Runner.use_full_oof_coverage
