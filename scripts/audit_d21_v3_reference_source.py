"""Audit reference-only source lineage and chronology, without labels or PnL."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_d21_v3_preflight import SOURCE as BASE, fingerprint
from research_d21_v3_tech_expansion import additions, expand_intervals, normalize_returns

SOURCE = ROOT / 'experiments/d21_v3_csi500_tech_preflight_20260914'
OUTPUT = ROOT / 'experiments/d21_v3_reference_source_audit_20260914'


def same_members(left, right):
    cols = ['date', 'code', 'industry_code']
    pd.testing.assert_frame_equal(
        left[cols].sort_values(['date', 'code']).reset_index(drop=True),
        right[cols].sort_values(['date', 'code']).reset_index(drop=True), check_dtype=False)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refuse overwriting audit')
    prior = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    checks = {}
    hashes = {name: fingerprint(SOURCE / name) for name in prior['file_hashes']}
    checks['frozen_source_hashes_match'] = hashes == prior['file_hashes']
    if not checks['frozen_source_hashes_match']:
        raise ValueError('Frozen source hash changed')
    split = json.loads((BASE / 'split_plan.json').read_text(encoding='utf-8'))
    base = pd.read_parquet(BASE / 'development_features_slow.parquet',
                           columns=['date', 'code', 'industry_code', 'r_1d'])
    dates = pd.DatetimeIndex(sorted(base.date.unique()))
    if not base.date.between(split['development_start'], split['development_end']).all():
        raise ValueError('Base outside Development')
    idx = pd.read_parquet(SOURCE / 'index500_intervals.parquet')
    tech = pd.read_parquet(SOURCE / 'technology_intervals.parquet')
    expanded = expand_intervals(idx.loc[idx.INDEX_CODE.eq('000905.SH')], dates,
                                'CON_CODE', 'INDATE', 'OUTDATE')
    checks['daily_csi500_exactly_500'] = bool(expanded.groupby('date').size().reindex(dates).eq(500).all())
    industry = expand_intervals(tech, dates, 'stock_code', 'in_date', 'out_date',
                                ('industry_code', 'industry_name'))
    added = additions(expanded, industry, base)
    same_members(added, pd.read_parquet(SOURCE / 'added_candidate_members.parquet'))
    checks['historical_added_members_rebuilt'] = True
    checks['technology_codes_only'] = bool(added.industry_code.isin(['801080.SI', '801750.SI', '801770.SI']).all())
    cutoff = dates[len(dates) // 2]
    # Rebuild on a shorter calendar; this checks prefix consistency, not vendor revision history.
    prefix_idx = expand_intervals(idx.loc[idx.INDEX_CODE.eq('000905.SH')], dates[dates <= cutoff],
                                  'CON_CODE', 'INDATE', 'OUTDATE')
    prefix_tech = expand_intervals(tech, dates[dates <= cutoff], 'stock_code', 'in_date', 'out_date',
                                   ('industry_code', 'industry_name'))
    prefix = additions(prefix_idx, prefix_tech, base[base.date <= cutoff])
    same_members(prefix, added[added.date <= cutoff])
    checks['interval_prefix_calendar_consistent'] = True
    bars = pd.concat([pd.read_parquet(p) for p in sorted(SOURCE.glob('bars_*.parquet'))], ignore_index=True)
    status = pd.concat([pd.read_parquet(p) for p in sorted(SOURCE.glob('status_*.parquet'))], ignore_index=True)
    returns = normalize_returns(bars, status, dates)
    combined = pd.read_parquet(SOURCE / 'development_candidate_inputs.parquet',
                               columns=['date', 'code', 'industry_code', 'r_1d', 'pool_origin'])
    stored = combined[combined.pool_origin.eq('added_csi500_tech')]
    same_members(stored, added)
    rebuilt = added.merge(returns, on=['date', 'code'], how='left', validate='one_to_one')
    pair = rebuilt.merge(stored[['date', 'code', 'r_1d']], on=['date', 'code'], suffixes=('_new', '_stored'), validate='one_to_one')
    np.testing.assert_allclose(pair.r_1d_new, pair.r_1d_stored, equal_nan=True, rtol=0, atol=1e-12)
    checks['reference_returns_rebuilt'] = True
    status['date'] = pd.to_datetime(status.TRADE_DATE.astype(str))
    status = status.rename(columns={'MARKET_CODE': 'code'})
    joined = stored.merge(status[['date', 'code', 'IS_SUSP_SEC']], on=['date', 'code'], how='left', validate='one_to_one')
    suspended = joined.IS_SUSP_SEC.astype(str).eq('1')
    checks['suspended_references_have_no_finite_return'] = not bool(np.isfinite(joined.loc[suspended, 'r_1d']).any())
    schemas = {n: pq.read_schema(SOURCE / n).names for n in
               ['index500_intervals.parquet', 'technology_intervals.parquet', 'bars_0000.parquet', 'status_0000.parquet']}
    manifest_paths = list(SOURCE.glob('*manifest*')) + list(SOURCE.glob('*request*'))
    issues = [
        {'code': 'RAW_REQUEST_LINEAGE_MISSING', 'severity': 'BLOCKING',
         'evidence': [str(p.relative_to(SOURCE)) for p in manifest_paths],
         'detail': 'No per-request manifest in this source bundle. Existing plan and file hashes do not supply request/completion times, SDK version, or per-batch request IDs required by data_dictionary.md.'},
        {'code': 'REFERENCE_TIMING_SCHEMA_NOT_NORMALIZED', 'severity': 'BLOCKING',
         'detail': 'Reference inputs lack available_at/source and row lineage. D05/D06 allow explicitly DERIVED_POLICY effective-day/21:00 timing; do not fabricate native announcement or historical download times.'},
        {'code': 'NATIVE_ANNOUNCEMENT_UNVERIFIED', 'severity': 'DISCLOSURE',
         'detail': 'Effective intervals are present, but native announcement/revision availability is not independently verified. Missing native announcement alone is not a new hard threshold.'},
    ]
    if manifest_paths:
        raise ValueError('New lineage evidence found; inspect before classifying as missing')
    report = {
        'status': 'DATA_EVIDENCE_INCOMPLETE', 'data_gate_passed': False,
        'checks': checks, 'all_structural_checks_passed': all(checks.values()),
        'source_file_hashes': hashes, 'raw_schemas': schemas, 'issues': issues,
        'reference_member_days': len(added), 'reference_stocks_ever': int(added.code.nunique()),
        'suspended_reference_member_days': int(suspended.sum()),
        'holdout_read': False, 'labels_read': False, 'economic_gate_evaluated': False,
        'stage3r_allowed': False, 'stage4r_allowed': False, 'stage5_allowed': False,
        'coverage_threshold_unchanged': .85,
        'code_sha256': fingerprint(Path(__file__)),
        'next_action': 'Recover original request records from source owner or collect a new immutable Development-only source version with complete request lineage, normalize timing under existing D05/D06, rerun audit and coverage before economics.',
    }
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_file_hashes', 'raw_schemas']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
