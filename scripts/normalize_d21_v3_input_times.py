"""Explicit D05/D06 normalization; does not invent request logs or native timestamps."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.hs300.input_timing import REQUIRED_TIMES
from data_pipeline.hs300.time_policy import available_at
from research_d21_v3_preflight import fingerprint

SOURCE = ROOT / 'experiments/d21_v3_csi500_tech_preflight_20260914'
BASE = ROOT / 'experiments/stage3r_d21_v2_h2_5d'
OUTPUT = ROOT / 'experiments/d21_v3_timing_normalized_20260914'


def previous_quote_times(rows, quotes, dates):
    calendar = pd.DataFrame({'date': dates, 'previous_date': pd.Series(dates).shift(1)})
    prior = quotes[['date', 'code', 'quote_available_at']].rename(
        columns={'date': 'previous_date', 'quote_available_at': 'previous_quote_available_at'})
    keys = rows[['date', 'code']].merge(calendar, on='date', how='left', validate='many_to_one')
    return keys.merge(prior, on=['previous_date', 'code'], how='left', validate='many_to_one')


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refuse overwriting normalized evidence')
    report = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    used = {name: fingerprint(SOURCE / name) for name in report['file_hashes']}
    if used != report['file_hashes']:
        raise ValueError('Frozen source changed')
    raw = pd.read_parquet(SOURCE / 'development_candidate_inputs.parquet')
    split = json.loads((BASE / 'split_plan.json').read_text(encoding='utf-8'))
    if not raw.date.between(split['development_start'], split['development_end']).all():
        raise ValueError('Refuse non-Development data')
    dates = pd.DatetimeIndex(sorted(raw.date.unique()))
    path = BASE / 'development_features_simple.parquet'
    simple = pd.read_parquet(path, columns=['date', 'code', 'available_at', 'available_at_quote',
        'available_at_source', 'has_quote', 'available_at_status', 'available_at_source_status',
        'industry_known_at', 'industry_known_at_source'])
    original = raw[raw.pool_origin.eq('original_csi300')].merge(simple, on=['date', 'code'],
                                                               how='left', validate='one_to_one')
    original['membership_available_at'] = original['available_at']
    original['quote_available_at'] = original['available_at_quote'].where(original.has_quote.eq(True))
    original['status_available_at'] = original['available_at_status']
    # Original returns use the frozen quote/TR chain. Both quote and status are independently gated.
    original['preclose_available_at'] = original['available_at_quote']
    original['industry_available_at'] = original['industry_known_at']
    previous = previous_quote_times(original, original, dates)
    original = original.merge(previous[['date', 'code', 'previous_quote_available_at']],
                               on=['date', 'code'], how='left', validate='one_to_one')
    for key, source in [('quote_available_at', 'available_at_source'),
                        ('preclose_available_at', 'available_at_source'),
                        ('status_available_at', 'available_at_source_status'),
                        ('industry_available_at', 'industry_known_at_source')]:
        original[key + '_source'] = original[source].where(original[key].notna(), 'MISSING')
    original['membership_available_at_source'] = 'DERIVED_POLICY'
    # Actual existing quote policy must be uniform before assigning provenance to previous quotes.
    if set(simple.available_at_source.dropna()) != {'DERIVED_POLICY'}:
        raise ValueError('Mixed quote time provenance requires a native-preserving adapter')
    original['previous_quote_available_at_source'] = 'DERIVED_POLICY'
    added = raw[raw.pool_origin.eq('added_csi500_tech')].copy()
    bars = pd.concat([pd.read_parquet(p) for p in sorted(SOURCE.glob('bars_*.parquet'))], ignore_index=True)
    statuses = pd.concat([pd.read_parquet(p) for p in sorted(SOURCE.glob('status_*.parquet'))], ignore_index=True)
    # This adapter is only for the inspected legacy bundle. Never overwrite newly provided native times.
    if any('available_at' in c.lower() or 'known_at' in c.lower() for c in [*bars.columns, *statuses.columns]):
        raise ValueError('Native timing evidence found: normalize it explicitly, do not overwrite')
    bars['date'] = pd.to_datetime(bars.kline_time).dt.normalize()
    bars['quote_available_at'] = available_at(bars.date).where(bars.close.gt(0))
    statuses['date'] = pd.to_datetime(statuses.TRADE_DATE.astype(str)).dt.normalize()
    statuses = statuses.rename(columns={'MARKET_CODE': 'code'})
    for data in [bars, statuses]:
        if not data.date.between(dates.min(), dates.max()).all() or data.duplicated(['date', 'code']).any():
            raise ValueError('Invalid raw quote/status keys or dates')
    statuses['status_available_at'] = available_at(statuses.date)
    statuses['preclose_available_at'] = available_at(statuses.date).where(statuses.PRECLOSE.gt(0))
    added = added.merge(bars[['date', 'code', 'quote_available_at']], on=['date', 'code'], how='left', validate='one_to_one')
    added = added.merge(statuses[['date', 'code', 'status_available_at', 'preclose_available_at']],
                         on=['date', 'code'], how='left', validate='one_to_one')
    previous = previous_quote_times(added, bars, dates)
    added = added.merge(previous[['date', 'code', 'previous_quote_available_at']], on=['date', 'code'],
                         how='left', validate='one_to_one')
    # Previously rebuilt effective-day membership/industry state; daily 21h policy is conservative.
    added['membership_available_at'] = available_at(added.date)
    added['industry_available_at'] = available_at(added.date).where(added.industry_code.notna())
    for column in REQUIRED_TIMES:
        added[column + '_source'] = 'DERIVED_POLICY'
    columns = list(raw.columns) + list(REQUIRED_TIMES) + [c + '_source' for c in REQUIRED_TIMES]
    result = pd.concat([original[columns], added[columns]], ignore_index=True)
    # Merges may coerce pandas string extension columns to object; retain the frozen payload schema.
    for column in raw.columns:
        result[column] = result[column].astype(raw[column].dtype)
    for column in REQUIRED_TIMES:
        result[column] = pd.to_datetime(result[column], utc=True)
        result.loc[result[column].isna(), column + '_source'] = 'MISSING'
    pd.testing.assert_frame_equal(raw.sort_values(['date', 'code']).reset_index(drop=True),
        result[raw.columns].sort_values(['date', 'code']).reset_index(drop=True))
    OUTPUT.mkdir(exist_ok=False)
    target = OUTPUT / 'development_candidate_inputs_timed.parquet'
    result.to_parquet(target, index=False)
    audit = {'status': 'TIME_NORMALIZED_REQUEST_LINEAGE_STILL_MISSING',
        'normalized_at': datetime.now(timezone.utc).isoformat(),
        'policy': 'D05_D06; DERIVED_POLICY is not native announcement or historical retrieval evidence',
        'rows': len(result), 'source_values_unchanged': True, 'request_lineage_verified': False,
        'time_sources': {c: result[c + '_source'].value_counts().to_dict() for c in REQUIRED_TIMES},
        'original_payload_sha256': used['development_candidate_inputs.parquet'],
        'source_file_hashes': used, 'original_simple_sha256': fingerprint(path),
        'output_sha256': fingerprint(target), 'code_sha256': fingerprint(Path(__file__)),
        'holdout_read': False, 'labels_read': False, 'economic_gate_evaluated': False}
    (OUTPUT / 'report.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k != 'source_file_hashes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
