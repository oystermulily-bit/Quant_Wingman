"""Disclose existing label-free candidates at 85%; does not select or freeze one."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/d21_v3_coverage_preflight_20260914'
OUTPUT = ROOT / 'experiments/d21_v3_coverage85_sourcecheck_20260914'
SCENARIOS = ['baseline', 'shrink_only', 'short_only', 'shrink_and_short']


def summarize(frame, scenarios=None):
    if frame.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate member-day')
    counts = frame.groupby('date').size()
    if not counts.eq(300).all():
        raise ValueError('All original 300 member rows must remain')
    rows, daily_rows, industry_rows = [], [], []
    for scenario in (SCENARIOS if scenarios is None else scenarios):
        if frame[scenario].isna().any() or not frame[scenario].isin([True, False]).all():
            raise ValueError('Invalid validity evidence')
        daily = frame.groupby('date')[scenario].agg(['size', 'sum'])
        daily.columns = ['member_count', 'valid_count']
        daily['coverage'] = daily.valid_count / daily.member_count
        daily['below_85'] = daily.coverage.lt(.85)
        daily['scenario'] = scenario
        daily_rows.append(daily.reset_index())
        # Missing industry is a reporting bucket only; never an imputed feature.
        industry = frame.assign(industry_bucket=frame.industry_code.fillna('UNKNOWN'))
        industry = industry.groupby('industry_bucket')[scenario].agg(['size', 'sum'])
        industry.columns = ['member_days', 'valid_member_days']
        industry['coverage'] = industry.valid_member_days / industry.member_days
        industry['scenario'] = scenario
        industry_rows.append(industry.reset_index())
        rows.append({
            'scenario': scenario, 'coverage': float(frame[scenario].mean()),
            'coverage_only_passed': bool(frame[scenario].mean() >= .85),
            'daily_valid_min': int(daily.valid_count.min()),
            'daily_valid_quantiles': {str(q): float(daily.valid_count.quantile(q))
                                     for q in [0, .05, .25, .5, .75, .95, 1]},
            'days_below_85': int(daily.below_85.sum()),
            'fraction_days_below_85': float(daily.below_85.mean()),
            'industry_coverage_below_85': industry[industry.coverage.lt(.85)].reset_index().to_dict('records'),
        })
    return rows, pd.concat(daily_rows), pd.concat(industry_rows)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refuse to overwrite evidence')
    path = SOURCE / 'coverage_evidence.parquet'
    prior = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    frame = pd.read_parquet(path, columns=['date', 'code', 'industry_code', *SCENARIOS])
    if len(frame) != prior['oof_member_days'] or frame.date.nunique() != prior['oof_dates']:
        raise ValueError('Original OOF denominator changed')
    rows, daily, industry = summarize(frame)
    report = {
        'status': 'COVERAGE_DISCLOSURE_COMPLETE_SOURCE_SUPPLEMENT_BLOCKED',
        'overall_threshold': .85, 'daily_industry_checks': 'DISCLOSURE_ONLY_NO_NEW_HARD_THRESHOLDS',
        'oof_member_days': len(frame), 'oof_dates': frame.date.nunique(),
        'labels_read': False, 'holdout_read': False, 'rd_agent_allowed': False,
        'economic_gate_evaluated': False, 'economic_thresholds': 'UNCHANGED_D18_V2',
        'v3_signal_definition_frozen': False, 'supplemented_rows': 0,
        'source_check': {
            'sdk_installed': True, 'callable_market_connector': False,
            'AD_credentials_present_process_user_machine': False,
            'local_snapshots': ['csi300_2010_present_v1', 'csi300_2014_present_v2'],
            'reason': 'Environment missing; existing MCP configuration found subsequently; see separate source probe',
            'probe_report': 'experiments/d21_v3_source_probe_20260914/report.json',
        },
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scenarios': rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    daily.to_parquet(OUTPUT / 'daily_coverage.parquet', index=False)
    industry.to_json(OUTPUT / 'industry_coverage.json', orient='records', indent=2)
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in rows:
        print(row['scenario'], round(row['coverage']*100, 2), row['daily_valid_min'],
              row['days_below_85'], round(row['fraction_days_below_85']*100, 2))


if __name__ == '__main__':
    main()
