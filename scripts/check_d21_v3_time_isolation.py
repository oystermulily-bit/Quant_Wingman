"""Read-only diagnosis of v3 timing guards; writes only independent audit evidence."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.hs300_panel import HS300PanelDataManager
from data_pipeline.hs300.time_policy import available_at
from research_d21_v3_preflight import fingerprint
from research_d21_v3_reference_only import build
from data_pipeline.hs300.input_timing import REQUIRED_TIMES
from research_d21_v3_tech_expansion import expand_intervals

SNAPSHOT = Path('D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2')
SOURCE = ROOT / 'experiments/d21_v3_csi500_tech_preflight_20260914'
OUTPUT = ROOT / 'experiments/d21_v3_time_isolation_fixed_check_20260914'


def adversarial_checks():
    dates = pd.date_range('2020-01-01', periods=4)
    history = pd.DataFrame({'code': ['A', 'A'], 'start': ['20200102', '20200104'],
                            'end': ['20200102', None], 'industry': ['old', 'new']})
    expanded = expand_intervals(history, dates, 'code', 'start', 'end', ('industry',))
    boundary = list(expanded.date) == [dates[1], dates[3]]
    # Changing future classification and exit metadata must not alter the prefix.
    changed = history.copy()
    changed.loc[1, ['industry', 'end']] = ['sentinel', '20300101']
    before = expand_intervals(history, dates[:3], 'code', 'start', 'end', ('industry',))
    after = expand_intervals(changed, dates[:3], 'code', 'start', 'end', ('industry',))
    future_interval_safe = before.equals(after)
    panel = pd.DataFrame({'date': [dates[1]], 'code': ['A']})
    industry = pd.DataFrame({'code': ['A'], 'valid_from': [dates[1]], 'valid_to': [pd.NaT],
        'industry_code': ['late'], 'known_at': [pd.Timestamp('2020-01-02 23:00', tz='Asia/Shanghai')],
        'known_at_source': ['NATIVE'], 'effective_at': [dates[1]]})
    joined = HS300PanelDataManager._join_industry_intervals(panel, industry)
    known_after_signal_rejected = bool(joined.industry_code.isna().all())
    sample = pd.DataFrame({'date': [dates[1]] * 3, 'code': ['A', 'B', 'C'],
        'industry_code': ['sector'] * 3, 'r_1d': [.01, .02, .03],
        'pool_origin': ['original_csi300', 'added_csi500_tech', 'added_csi500_tech'],
        'available_at': [pd.Timestamp('2020-01-02 21:00', tz='Asia/Shanghai')] * 3})
    for column in REQUIRED_TIMES:
        sample[column] = sample['available_at']
    baseline = build(sample)
    sample.loc[sample.code.eq('C'), 'available_at'] = pd.Timestamp('2020-01-03 21:00', tz='Asia/Shanghai')
    delayed = build(sample)
    # A delayed reference must be excluded or explicitly rejected; builder currently ignores it.
    late_reference_ignored_in_calculation = bool(np.allclose(baseline.epsilon, delayed.epsilon, equal_nan=True))
    sample.loc[sample.code.eq('C'), 'r_1d'] = 999
    influenced = build(sample)
    late_reference_changes_target = not bool(np.allclose(delayed.epsilon, influenced.epsilon, equal_nan=True))
    return {
        'effective_interval_inclusive_boundaries_and_gaps': boundary,
        'future_interval_mutation_prefix_unchanged': future_interval_safe,
        'industry_known_23h_rejected_at_21h_signal': known_after_signal_rejected,
        'v3_builder_enforces_reference_available_at': not (late_reference_ignored_in_calculation and late_reference_changes_target),
        'findings_are_synthetic_not_proof_of_actual_future_return_leak': True,
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refuse overwriting diagnostic evidence')
    lock_path = SNAPSHOT / 'splits/holdout.lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    split = json.loads((ROOT / 'experiments/stage3r_d21_v2_h2_5d/split_plan.json').read_text(encoding='utf-8'))
    start = pd.Timestamp(lock['holdout_start'])
    scans = []
    # Read date columns only from Development/standardized files, never sealed payloads or labels.
    paths = [SNAPSHOT / 'standardized' / name for name in
             ['daily_bars.parquet', 'execution_bars.parquet', 'execution_status.parquet',
              'trading_status.parquet', 'universe_membership.parquet']]
    paths += [SOURCE / 'development_candidate_inputs.parquet', SOURCE / 'added_candidate_members.parquet',
              ROOT / 'experiments/d21_v3_reference_only_20260914/development_reference_only_signals.parquet']
    for path in paths:
        d = pd.read_parquet(path, columns=['date']).date
        scans.append({'path': str(path), 'rows': len(d), 'date_min': str(d.min()),
                      'date_max': str(d.max()), 'missing_dates': int(d.isna().sum()),
                      'holdout_rows': int(d.ge(start).sum()), 'sha256': fingerprint(path)})
    timing = {}
    for table in ['daily_bars', 'execution_bars', 'universe_membership']:
        path = SNAPSHOT / f'standardized/{table}.parquet'
        columns = ['date', 'available_at']
        fields = pq.read_schema(path).names
        source_col = 'available_at_source' if 'available_at_source' in fields else None
        if source_col:
            columns.append(source_col)
        frame = pd.read_parquet(path, columns=columns)
        observed = pd.to_datetime(frame.available_at, utc=True)
        signal = available_at(frame.date).dt.tz_convert('UTC')
        timing[table] = {'missing_available_at': int(observed.isna().sum()),
                        'later_than_signal_21h': int(observed.gt(signal).sum()),
                        'source_values': frame[source_col].value_counts(dropna=False).to_dict() if source_col else 'NOT_PRESENT'}
    schemas = {p.name: pq.read_schema(p).names for p in sorted(SOURCE.glob('bars_*.parquet'))}
    timing['added_reference_bars'] = {'batches': len(schemas),
        'batches_missing_available_at': sum('available_at' not in cols for cols in schemas.values()),
        'actual_availability_verified': False}
    probes = adversarial_checks()
    guards_passed = all(probes.values())
    report = {'status': 'GUARDS_FIXED_SOURCE_TIMING_METADATA_PENDING' if guards_passed else 'TIMING_GUARD_GAPS_FOUND',
        'adversarial_checks': probes,
        'availability_scan': timing, 'development_date_scans': scans,
        'holdout_lock_matches_split': all(lock[k] == split[k] for k in
                                         ['holdout_start', 'holdout_date_count', 'holdout_date_hash']),
        'development_excludes_holdout': all(x['holdout_rows'] == 0 and x['missing_dates'] == 0 for x in scans),
        'holdout_lock_sha256': fingerprint(lock_path),
        'holdout_payload_read': False, 'labels_read': False, 'economic_gate_evaluated': False,
        'scope': 'Logical/file separation and current inputs checked; no claim of OS-level denial or complete historical access audit',
        'production_code_modified': True, 'stage5_allowed': False,
        'code_sha256': fingerprint(Path(__file__)),
        'implementation_hashes': {p: fingerprint(ROOT / p) for p in
          ['research_d21_v3_reference_only.py', 'research_d21_v3_tech_expansion.py', 'data_pipeline/hs300_panel.py']}}
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['development_date_scans', 'implementation_hashes']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
