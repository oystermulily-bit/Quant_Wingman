"""Frozen v3 label-free build: expanded references, CSI300 targets only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data_pipeline.hs300.input_timing import REQUIRED_TIMES, mask_unavailable, signal_times
from data_pipeline.hs300.time_policy import available_at

from research_d21_v3_preflight import SOURCE as BASE, fingerprint, residuals, signals
from research_d21_v3_coverage85 import summarize

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/d21_v3_csi500_tech_preflight_20260914'
OUTPUT = ROOT / 'experiments/d21_v3_reference_only_timing_guard_20260914'
SPEC = ('shrink_only', 'shrink', 60, 54, 120, 20, 90, 60, 5, 50)


def build(frame):
    if frame.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate reference member day')
    if not frame.pool_origin.isin(['original_csi300', 'added_csi500_tech']).all():
        raise ValueError('Unknown reference origin')
    frame = mask_unavailable(frame)
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    # Compute reference cross-sections FIRST, then mask non-target dates BEFORE rolling.
    full = residuals(frame)
    target = full.loc[full.pool_origin.eq('original_csi300')].copy()
    matrix = target.pivot(index='date', columns='code', values='shrink').reindex(index=dates)
    values = signals(matrix, SPEC)
    di, ci = dates.get_indexer(target.date), matrix.columns.get_indexer(target.code)
    names = {'low_vol': 'IDIO_LOW_VOL_60', 'momentum': 'RES_MOM_120_20',
             'efficiency': 'RES_TREND_EFF_60_5'}
    ranks = []
    for key, name in names.items():
        target[name] = values[key].to_numpy()[di, ci]
        target.loc[~target.membership_time_usable, name] = np.nan
        clipped = target.groupby('date')[name].transform(
            lambda s: s.clip(lower=s.quantile(.01), upper=s.quantile(.99)))
        target[name + '_rank'] = clipped.groupby(target.date).rank(method='average', pct=True)
        ranks.append(name + '_rank')
    target['valid_slow_signal'] = np.isfinite(target[list(names.values())]).all(axis=1)
    target['slow_residual_ensemble'] = target[ranks].mean(axis=1).where(target.valid_slow_signal)
    target['epsilon'] = target['shrink']
    return target


def main(timed_input=None):
    if OUTPUT.exists():
        raise FileExistsError('Refuse overwriting evidence')
    raw_source = SOURCE / 'development_candidate_inputs.parquet'
    prior = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    if fingerprint(raw_source) != prior['file_hashes'][raw_source.name]:
        raise ValueError('Source hash changed')
    source = Path(timed_input) if timed_input is not None else raw_source
    split = json.loads((BASE / 'split_plan.json').read_text(encoding='utf-8'))
    columns = ['date', 'code', 'industry_code', 'r_1d', 'pool_origin', *REQUIRED_TIMES]
    schema = pq.read_schema(source).names
    missing = set(REQUIRED_TIMES) - set(schema)
    if missing:
        raise ValueError(f'Source not timing-normalized: {sorted(missing)}; do not infer availability inside builder')
    columns += [c for c in ['signal_at', 'available_at'] if c in schema]
    columns += [c + '_source' for c in REQUIRED_TIMES if c + '_source' in schema]
    frame = pd.read_parquet(source, columns=columns)
    frame['date'] = pd.to_datetime(frame.date)
    if not frame.date.between(split['development_start'], split['development_end']).all():
        raise ValueError('Dates outside Development')
    if not signal_times(frame).eq(available_at(frame.date).dt.tz_convert('UTC')).all():
        raise ValueError('Frozen v3 research requires the 21:00 Asia/Shanghai signal cutoff')
    if timed_input is not None:
        payload = ['date', 'code', 'industry_code', 'r_1d', 'pool_origin']
        raw = pd.read_parquet(raw_source, columns=payload)
        pd.testing.assert_frame_equal(
            frame[payload].sort_values(['date', 'code']).reset_index(drop=True),
            raw.sort_values(['date', 'code']).reset_index(drop=True), check_dtype=False)
    original = pd.read_parquet(BASE / 'development_features_slow.parquet', columns=['date', 'code'])
    target = build(frame)
    pd.testing.assert_frame_equal(
        target[['date', 'code']].sort_values(['date', 'code']).reset_index(drop=True),
        original.sort_values(['date', 'code']).reset_index(drop=True))
    if not target.groupby('date').size().eq(300).all():
        raise ValueError('Trade universe changed')
    oof = np.logical_or.reduce([target.date.between(f['validation_start'], f['validation_end'])
                               for f in split['folds']])
    detail = target.loc[oof, ['date', 'code', 'industry_code']].copy()
    detail['reference_only_long_shrink'] = target.loc[oof, 'valid_slow_signal']
    rows, daily, industry = summarize(detail, scenarios=['reference_only_long_shrink'])
    report = {
        'status': 'COVERAGE_PASSED_DATA_AUDIT_PENDING' if rows[0]['coverage_only_passed'] else 'INSUFFICIENT_EVIDENCE',
        'hypothesis_id': 'H3_5D_LONG_SHRINK_TECH_REFERENCE_ONLY_V1',
        'coverage_threshold': .85, 'oof_member_days': len(detail), 'oof_dates': detail.date.nunique(),
        'development_adaptive': True, 'holdout_read': False, 'labels_read': False,
        'stage5_allowed': False, 'economic_gate_evaluated': False,
        'data_limitations': prior['data_limitations'],
        'source_sha256': fingerprint(source), 'original_source_sha256': fingerprint(raw_source),
        'source_timing_lineage_verified': False,
        'split_sha256': fingerprint(BASE / 'split_plan.json'),
        'code_sha256': fingerprint(Path(__file__)),
        'dependencies_sha256': {p: fingerprint(ROOT / p) for p in
                                ['research_d21_v3_preflight.py', 'research_d21_v3_coverage85.py',
                                 'data_pipeline/hs300/input_timing.py']},
        'scenarios': rows,
    }
    OUTPUT.mkdir(exist_ok=False)
    target.to_parquet(OUTPUT / 'development_reference_only_signals.parquet', index=False)
    detail.to_parquet(OUTPUT / 'coverage_evidence.parquet', index=False)
    daily.to_parquet(OUTPUT / 'daily_coverage.parquet', index=False)
    industry.to_json(OUTPUT / 'industry_coverage.json', orient='records', indent=2)
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timed-input', type=Path,
                        help='Explicitly timing-normalized copy; original data values must match frozen source')
    main(parser.parse_args().timed_input)
