"""Reassess existing expanded-pool evidence; never selects a signal contract."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from research_d21_v3_coverage85 import SCENARIOS, summarize

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/d21_v3_csi500_tech_preflight_20260914'
OUTPUT = ROOT / 'experiments/d21_v3_expanded_coverage85_20260914'


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refuse overwriting evidence')
    path = SOURCE / 'coverage_evidence.parquet'
    prior = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != prior['file_hashes'][path.name]:
        raise ValueError('Source evidence changed')
    frame = pd.read_parquet(path, columns=['date', 'code', 'industry_code', 'pool_origin', *SCENARIOS])
    if frame.duplicated(['date', 'code']).any() or len(frame) != prior['oof_member_days']:
        raise ValueError('Expanded member-day denominator changed')
    if not frame.pool_origin.isin(['original_csi300', 'added_csi500_tech']).all():
        raise ValueError('Unknown pool origin')
    original = frame.loc[frame.pool_origin.eq('original_csi300')].copy()
    if len(original) != 460500 or original.date.nunique() != 1535:
        raise ValueError('Original OOF member days changed')
    rows, daily, industry = summarize(original)
    for row in rows:
        name = row['scenario']
        row['expanded_pool_coverage_disclosure_only'] = float(frame[name].mean())
        row['added_pool_coverage_disclosure_only'] = float(frame.loc[frame.pool_origin.eq('added_csi500_tech'), name].mean())
    report = {
        'status': 'COVERAGE_REASSESSMENT_COMPLETE_SIGNAL_CONTRACT_PENDING',
        'coverage_threshold': .85,
        'gate_denominator': 'ALL_ORIGINAL_CSI300_OOF_MEMBER_DAYS',
        'oof_member_days': len(original), 'oof_dates': original.date.nunique(),
        'expanded_member_days_disclosure_only': len(frame),
        'economic_thresholds': 'UNCHANGED_D18_V2',
        'economic_gate_evaluated': False, 'development_adaptive': True,
        'signal_contract_frozen': False, 'stage5_allowed': False,
        'labels_read': False, 'holdout_read': False,
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scenarios': rows,
    }
    OUTPUT.mkdir(exist_ok=False)
    daily.to_parquet(OUTPUT / 'daily_coverage.parquet', index=False)
    industry.to_json(OUTPUT / 'industry_coverage.json', orient='records', indent=2)
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
