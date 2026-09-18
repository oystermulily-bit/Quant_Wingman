"""Independent v3 execution using unchanged v2 economics and the strict timed signal build."""
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from research_d21_v2.protocol import D21V2Config, EXPERIMENTS
from research_d21_v2.runner import Stage3RResearchRunner, _atomic_json
from research_d21_v2.gate import Stage4RFeasibilityRunner, classify_d21_v2
from research_d21_v3_preflight import fingerprint

ROOT = Path(__file__).resolve().parent
TIMED = ROOT / 'experiments/d21_v3_reference_only_timing_guard_20260914'
NORMALIZED = ROOT / 'experiments/d21_v3_timing_normalized_20260914'
SOURCE = ROOT / 'experiments/d21_v3_source_reverified_host_20260914'
SNAPSHOT = Path('D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2')
STAGE3 = ROOT / 'experiments/stage3r_d21_v3_reference_only_h5'
STAGE4 = ROOT / 'experiments/stage4r_d21_v3_reference_only_h5'


@dataclass(frozen=True)
class D21V3Config(D21V2Config):
    hypothesis_id: str = 'H3_5D_LONG_SHRINK_TECH_REFERENCE_ONLY_V1'
    signal_version: str = 'LONG_SHRINK_TECH_REFERENCE_ONLY_V1'
    stage3r_protocol: str = 'w1ngman_stage3r_d21_v3_reference_only_h5_v1'
    stage4r_protocol: str = 'w1ngman_stage4r_d21_v3_reference_only_h5_v1'
    loo_universe: str = 'PIT_CSI300_UNION_CSI500_TECH_REFERENCE_ONLY'
    min_oof_coverage: float = .85


def admit():
    source = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    norm = json.loads((NORMALIZED / 'report.json').read_text(encoding='utf-8'))
    timed = json.loads((TIMED / 'report.json').read_text(encoding='utf-8'))
    if source['status'] != 'REFERENCE_SOURCE_REVERIFIED' or not source['original_values_and_membership_match']:
        raise ValueError('Fresh source verification incomplete')
    if source['request_manifest_sha256'] != fingerprint(SOURCE / 'raw_request_manifest.json'):
        raise ValueError('Request ledger changed')
    if source['row_lineage_sha256'] != fingerprint(SOURCE / 'reference_row_lineage.parquet'):
        raise ValueError('Reference row lineage changed')
    requests = json.loads((SOURCE / 'raw_request_manifest.json').read_text(encoding='utf-8'))
    if len(requests) != 18 or any(not r['success'] for r in requests):
        raise ValueError('Expected 18 completed source requests')
    for request in requests:
        path = SOURCE / request['response_path']
        if path.resolve().parent != SOURCE.resolve() or fingerprint(path) != request['response_sha256']:
            raise ValueError('Source response changed or invalid path')
        if pd.Timestamp(request['completed_at']) < pd.Timestamp(request['requested_at']):
            raise ValueError('Invalid request timestamps')
    raw_hash = source['original_input_sha256']
    if norm['original_payload_sha256'] != raw_hash or timed['original_source_sha256'] != raw_hash:
        raise ValueError('Source/normalization/build chain mismatch')
    if fingerprint(NORMALIZED / 'development_candidate_inputs_timed.parquet') != norm['output_sha256']:
        raise ValueError('Normalized input changed')
    if timed['source_sha256'] != norm['output_sha256']:
        raise ValueError('Timed feature input mismatch')
    for relative, sha in timed['dependencies_sha256'].items():
        if fingerprint(ROOT / relative) != sha:
            raise ValueError('Signal dependency code changed')
    if fingerprint(ROOT / 'research_d21_v3_reference_only.py') != timed['code_sha256']:
        raise ValueError('Signal construction code changed')
    if timed['oof_member_days'] != 460500 or not timed['scenarios'][0]['coverage_only_passed']:
        raise ValueError('Full OOF coverage failed')
    if any(r.get('holdout_read') or r.get('labels_read') for r in [source, norm, timed]):
        raise ValueError('Preflight isolation violation')
    return {'status': 'REFERENCE_DATA_GATE_PASSED_DERIVED_TIME_POLICY',
            'source_report_sha256': fingerprint(SOURCE / 'report.json'),
            'normalized_report_sha256': fingerprint(NORMALIZED / 'report.json'),
            'timed_report_sha256': fingerprint(TIMED / 'report.json'),
            'signal_file_sha256': fingerprint(TIMED / 'development_reference_only_signals.parquet'),
            'full_oof_coverage': timed['scenarios'][0]['coverage'],
            'native_announcement_verified': False, 'time_policy': 'D05_D06_DERIVED_POLICY',
            'fresh_request_lineage_verified': True,
            'holdout_read': False, 'economic_gate_evaluated': False}


class Stage3V3Runner(Stage3RResearchRunner):
    completion_status = 'STAGE3R_D21_V3_COMPLETED'
    experiments = tuple(dict(s, signal_version=D21V3Config().signal_version)
                        if s['experiment_id'] in ['B', 'D'] else dict(s) for s in EXPERIMENTS)

    def __init__(self, admission):
        super().__init__(D21V3Config())
        self.admission = admission

    def build_slow(self, dev_panel, dev_bars, trading_dates):
        path = TIMED / 'development_reference_only_signals.parquet'
        if fingerprint(path) != self.admission['signal_file_sha256']:
            raise ValueError('Signals changed after admission')
        slow = pd.read_parquet(path)
        pd.testing.assert_frame_equal(
            slow[['date', 'code']].sort_values(['date', 'code']).reset_index(drop=True),
            dev_panel[['date', 'code']].sort_values(['date', 'code']).reset_index(drop=True), check_dtype=False)
        slow['signal_version'] = self.config.signal_version
        return slow


class Stage4V3Runner(Stage4RFeasibilityRunner):
    expected_stage3_status = 'STAGE3R_D21_V3_COMPLETED'
    use_full_oof_coverage = True

    @staticmethod
    def classify(go, coverage_ok):
        result = classify_d21_v2(go, coverage_ok=coverage_ok)
        for key in ['statistical_gate', 'horizon_conclusion']:
            result[key] = result[key].replace('5D_V2', '5D_V3').replace('D21_V2_FAILED', 'D21_V3_FAILED')
        return result


def main():
    if STAGE3.exists() or STAGE4.exists():
        raise FileExistsError('Refuse overwriting independent stage artifacts')
    admission = admit()
    STAGE3.mkdir()
    _atomic_json(STAGE3 / 'reference_data_gate.json', admission)
    _atomic_json(STAGE3 / 'run_plan.json', {'config': asdict(D21V3Config()),
        'source_directory': str(SOURCE), 'timed_directory': str(TIMED),
        'holdout_read': False, 'stage5_allowed': False,
        'implementation_hashes': {p: fingerprint(ROOT / p) for p in
          ['research_d21_v3_execution.py', 'research_d21_v2/runner.py', 'research_d21_v2/gate.py',
           'research_d21_v2/backtest.py', 'research_stage3/backtest.py']}})
    print('STAGE3R_V3_START', flush=True)
    state = {'pid': os.getpid(), 'stage3r': 'RUNNING', 'stage4r': 'NOT_RUN',
             'holdout_read': False, 'stage5_allowed': False}
    _atomic_json(STAGE3 / 'run_state.json', state)
    try:
        report = Stage3V3Runner(admission).run(SNAPSHOT, STAGE3)
    except Exception as exc:
        state.update(stage3r='FAILED', error_type=type(exc).__name__)
        _atomic_json(STAGE3 / 'run_state.json', state)
        raise
    if report['status'] != Stage3V3Runner.completion_status:
        raise ValueError(f"Stage3 stopped: {report['status']}")
    if report['config_hash'] != D21V3Config().fingerprint():
        raise ValueError('Stage3 config changed')
    print('STAGE4R_V3_START', flush=True)
    state.update(stage3r=report['status'], stage4r='RUNNING')
    _atomic_json(STAGE3 / 'run_state.json', state)
    try:
        result = Stage4V3Runner(D21V3Config()).run(snapshot_dir=SNAPSHOT, stage3r_dir=STAGE3, output_dir=STAGE4)
    except Exception as exc:
        state.update(stage4r='FAILED', error_type=type(exc).__name__)
        _atomic_json(STAGE3 / 'run_state.json', state)
        raise
    state.update(stage4r=result['status'])
    _atomic_json(STAGE3 / 'run_state.json', state)
    print(json.dumps({'status': result['status'], 'go': result.get('go'),
                      'holdout_read': False, 'stage5_allowed': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
