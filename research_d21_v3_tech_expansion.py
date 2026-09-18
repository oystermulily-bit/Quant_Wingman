"""Isolated, label-free CSI500 technology candidate-universe preflight.

Credentials are environment-only. Never loads provider model/bundle caches.
Membership uses historical effective intervals; no announcement-time claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_d21_v3_preflight import SCENARIOS, SOURCE, fingerprint, residuals, signals

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'experiments/d21_v3_csi500_tech_preflight_20260914'
PROVIDER_ROOT = Path('D:/Hulucoding/AmAzing_Data/tech_cross_section')
TECH = ('电子', '计算机', '通信')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def expand_intervals(history, dates, code_column, start_column, end_column, extra=()):
    """Inclusive OUTDATE, same convention as the existing provider/snapshot.

    Bad dates and conflicting assignments fail closed, not treated as perpetual.
    """
    items = []
    for row in history.to_dict('records'):
        start = pd.to_datetime(str(row[start_column]).removesuffix('.0'), errors='coerce')
        raw_end = row[end_column]
        end = pd.to_datetime(str(raw_end).removesuffix('.0'), errors='coerce')
        if pd.isna(start):
            raise ValueError('Missing/unparseable membership start')
        if pd.isna(end) and pd.notna(raw_end) and str(raw_end).strip() not in ('', '0', 'NaT', 'None'):
            raise ValueError('Unparseable membership end')
        if pd.notna(end) and end < start:
            raise ValueError('Reversed membership interval')
        active = dates[(dates >= start) & ((dates <= end) if pd.notna(end) else True)]
        if len(active):
            item = pd.DataFrame({'date': active, 'code': str(row[code_column]).strip().upper()})
            for column in extra:
                item[column] = row[column]
            items.append(item)
    if not items:
        return pd.DataFrame(columns=['date', 'code', *extra])
    result = pd.concat(items, ignore_index=True).drop_duplicates()
    if result.duplicated(['date', 'code']).any():
        raise ValueError('Conflicting simultaneous membership assignments')
    return result


def additions(index500, technology, original):
    tech500 = index500.merge(technology, on=['date', 'code'], validate='one_to_one')
    tagged = tech500.merge(original[['date', 'code']], on=['date', 'code'], how='left', indicator=True)
    return tagged.loc[tagged['_merge'].eq('left_only')].drop(columns='_merge').reset_index(drop=True)


def load_base():
    split = json.loads((SOURCE / 'split_plan.json').read_text(encoding='utf-8'))
    base = pd.read_parquet(SOURCE / 'development_features_slow.parquet',
                           columns=['date', 'code', 'industry_code', 'r_1d'])
    base['date'] = pd.to_datetime(base['date'])
    if not base.date.between(split['development_start'], split['development_end']).all():
        raise ValueError('Refuse dates outside Development')
    return base, split


def fetch():
    sys.path.insert(0, str(PROVIDER_ROOT / 'src'))
    from tech_cross_section.config import load_config
    from tech_cross_section.data import AmazingDataProvider, _as_frame
    from tech_cross_section.universe_export import _flatten_constituents

    base, split = load_base()
    dates = pd.DatetimeIndex(sorted(base.date.unique()))
    if OUTPUT.exists():
        raise FileExistsError('Refuse overwriting existing experimental evidence')
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / 'plan.json', {
        'experiment': 'D21_V3_CSI500_TECH_CANDIDATE_PREFLIGHT',
        'frozen_economic_protocol': False, 'development_adaptive': True,
        'candidate_pool': 'PIT CSI300 UNION (PIT CSI500 technology MINUS PIT CSI300)',
        'technology': list(TECH), 'coverage_threshold': .95,
        'scenarios_fixed_before_fetch': SCENARIOS,
        'non_candidate_residual_history': 'missing; not backfilled',
        'development_start': split['development_start'], 'development_end': split['development_end'],
        'labels_read': False, 'holdout_read': False, 'rd_agent_allowed': False,
        'interval_policy': 'INDATE <= date <= OUTDATE; effective dates, not verified announcement dates',
        'source_code_sha256': fingerprint(Path(__file__)),
        'original_source_sha256': fingerprint(SOURCE / 'development_features_slow.parquet'),
    })
    provider = AmazingDataProvider(load_config(PROVIDER_ROOT / 'config.hs300_tech.toml'))
    raw = _flatten_constituents(provider._call_info(('get_index_constituent',), ['000905.SH'], is_local=False), '000905.SH')
    raw.to_parquet(OUTPUT / 'index500_intervals.parquet', index=False)
    raw = raw.loc[raw.INDEX_CODE.eq('000905.SH')]
    idx = expand_intervals(raw, dates, 'CON_CODE', 'INDATE', 'OUTDATE')
    counts = idx.groupby('date').size().reindex(dates, fill_value=0)
    if not counts.eq(500).all():
        write_json(OUTPUT / 'index_count_failure.json', counts[counts.ne(500)].to_dict())
        raise ValueError('Historical CSI500 membership is not exactly 500; stop before inference')
    _, history = provider.discover_technology_universe(dates.min(), dates.max())
    history.to_parquet(OUTPUT / 'technology_intervals.parquet', index=False)
    tech = expand_intervals(history, dates, 'stock_code', 'in_date', 'out_date', ('industry_code', 'industry_name'))
    added = additions(idx, tech, base)
    added.to_parquet(OUTPUT / 'added_candidate_members.parquet', index=False)
    latest = added.loc[added.date.eq(dates.max())].sort_values('code')
    latest.to_csv(OUTPUT / 'added_candidates_development_end.csv', index=False, encoding='utf-8-sig')
    codes = sorted(added.code.unique())
    print(f'PIT_UNIVERSE_READY added_stocks_ever={len(codes)} latest={len(latest)} member_days={len(added)}', flush=True)
    begin, end = int(dates.min().strftime('%Y%m%d')), int(dates.max().strftime('%Y%m%d'))
    # Checkpoint small batches. A failed download retains completed source evidence.
    for offset in range(0, len(codes), 25):
        batch = codes[offset:offset+25]
        bars = _as_frame(provider._fetch_kline(batch, begin, end))
        status = provider._batch_info('get_history_stock_status', batch, begin, end)
        # APIs are bounded, and the normalizer below independently rejects future rows.
        bars.to_parquet(OUTPUT / f'bars_{offset:04d}.parquet', index=False)
        status.to_parquet(OUTPUT / f'status_{offset:04d}.parquet', index=False)
        print(f'DOWNLOAD {min(offset+25,len(codes))}/{len(codes)}', flush=True)
    write_json(OUTPUT / 'download_complete.json', {'complete': True, 'codes': len(codes), 'holdout_read': False})


def normalize_returns(bars, status, dates):
    bar = bars.rename(columns={c: str(c).lower() for c in bars.columns}).copy()
    stat = status.rename(columns={c: str(c).lower() for c in status.columns}).copy()
    date_col = next(c for c in ('kline_time', 'trade_time', 'date') if c in bar)
    bar['date'] = pd.to_datetime(bar[date_col]).dt.normalize()
    stat['date'] = pd.to_datetime(stat['trade_date'].astype(str)).dt.normalize()
    for frame in (bar, stat):
        if not frame.date.between(dates.min(), dates.max()).all():
            raise ValueError('Source returned data outside requested Development range')
        if 'code' in frame and 'market_code' in frame:
            if not frame['code'].astype(str).eq(frame['market_code'].astype(str)).all():
                raise ValueError('SDK dictionary key disagrees with row stock code')
            frame.drop(columns='market_code', inplace=True)
        else:
            frame.rename(columns={'market_code': 'code'}, inplace=True)
        if frame.duplicated(['date', 'code']).any():
            raise ValueError('Duplicate source date/code')
    joined = bar[['date', 'code', 'close']].merge(stat[['date', 'code', 'preclose']], on=['date', 'code'], how='left', validate='one_to_one')
    for col in ('close', 'preclose'):
        joined[col] = pd.to_numeric(joined[col], errors='coerce')
    matrix = joined.pivot(index='date', columns='code', values='close').reindex(dates)
    prior = matrix.shift(1).notna()
    di, ci = dates.get_indexer(joined.date), matrix.columns.get_indexer(joined.code)
    valid = joined.close.gt(0) & joined.preclose.gt(0) & prior.to_numpy()[di, ci]
    joined['r_1d'] = np.log(joined.close.where(valid) / joined.preclose.where(valid))
    return joined[['date', 'code', 'r_1d']]


def evaluate():
    if not (OUTPUT / 'download_complete.json').exists():
        raise ValueError('Incomplete source download')
    if (OUTPUT / 'report.json').exists():
        raise FileExistsError('Do not overwrite evaluated evidence')
    base, split = load_base()
    dates = pd.DatetimeIndex(sorted(base.date.unique()))
    added = pd.read_parquet(OUTPUT / 'added_candidate_members.parquet')
    bars = pd.concat([pd.read_parquet(p) for p in sorted(OUTPUT.glob('bars_*.parquet'))], ignore_index=True)
    status = pd.concat([pd.read_parquet(p) for p in sorted(OUTPUT.glob('status_*.parquet'))], ignore_index=True)
    returns = normalize_returns(bars, status, dates)
    overlap = base.merge(returns, on=['date', 'code'], suffixes=('_old','_new'))
    paired = overlap[['r_1d_old', 'r_1d_new']].dropna()
    error = (paired.r_1d_old - paired.r_1d_new).abs()
    reconciliation = {'paired_rows': len(paired), 'max_abs_difference': float(error.max()),
                      'rows_different_over_1e_6': int(error.gt(1e-6).sum())}
    write_json(OUTPUT / 'return_reconciliation.json', reconciliation)
    # Never mix conflicting return definitions to manufacture improved coverage.
    if len(paired) == 0 or error.gt(1e-6).any():
        raise ValueError('New-source returns disagree with old snapshot; reconciliation required')
    added = added.merge(returns, on=['date', 'code'], how='left', validate='one_to_one')
    base['pool_origin'] = 'original_csi300'
    added['pool_origin'] = 'added_csi500_tech'
    combined = pd.concat([base, added[base.columns]], ignore_index=True)
    if combined.duplicated(['date','code']).any():
        raise ValueError('Candidate overlap')
    combined.to_parquet(OUTPUT / 'development_candidate_inputs.parquet', index=False)
    frame = residuals(combined)
    codes = pd.Index(sorted(frame.code.unique()))
    di, ci = dates.get_indexer(frame.date), codes.get_indexer(frame.code)
    folds = [frame.date.between(f['validation_start'], f['validation_end']) for f in split['folds']]
    oof = np.logical_or.reduce(folds)
    rows = []
    detail = frame[['date','code','industry_code','pool_origin','peer_count']].copy()
    for spec in SCENARIOS:
        matrix = frame.pivot(index='date', columns='code', values=spec[1]).reindex(index=dates, columns=codes)
        values = signals(matrix, spec)
        valid = np.logical_and.reduce([np.isfinite(v.to_numpy()[di,ci]) for v in values.values()])
        detail[spec[0]] = valid
        coverage = float(valid[oof].mean())
        daily_coverage = pd.Series(valid[oof], index=frame.loc[oof, 'date']).groupby(level=0).mean()
        rows.append({'name': spec[0], 'coverage': coverage, 'coverage_passed': coverage >= .95,
                     'fold_coverage': [float(valid[f].mean()) for f in folds],
                     'daily_coverage_min': float(daily_coverage.min()),
                     'daily_coverage_median': float(daily_coverage.median()),
                     'original_csi300_coverage': float(valid[oof & frame.pool_origin.eq('original_csi300')].mean()),
                     'added_csi500_tech_coverage': float(valid[oof & frame.pool_origin.eq('added_csi500_tech')].mean()),
                     'signal_coverage': {k: float(np.isfinite(v.to_numpy()[di,ci])[oof].mean()) for k,v in values.items()}})
    detail.loc[oof].to_parquet(OUTPUT / 'coverage_evidence.parquet', index=False)
    cause = pd.Series('valid_strict_residual', index=frame.index)
    missing_industry = frame.industry_code.isna()
    missing_return = ~np.isfinite(frame.r_1d)
    cause.loc[missing_industry] = 'missing_industry'
    cause.loc[~missing_industry & missing_return] = 'missing_return'
    cause.loc[~missing_industry & ~missing_return & frame.peer_count.lt(5)] = 'fewer_than_5_peers'
    report = {
        'status': 'COVERAGE_PREFLIGHT_ONLY_NOT_A_GO_DECISION',
        'evaluation_code_sha256': fingerprint(Path(__file__)),
        'coverage_any_scenario_passed': any(r['coverage_passed'] for r in rows),
        'scenarios': rows, 'coverage_threshold': .95,
        'oof_member_days': int(oof.sum()), 'oof_dates': frame.loc[oof,'date'].nunique(),
        'added_stocks_ever': added.code.nunique(), 'latest_added_count': int(added.date.eq(dates.max()).sum()),
        'oof_daily_pool_min': int(frame.loc[oof].groupby('date').size().min()),
        'oof_daily_pool_max': int(frame.loc[oof].groupby('date').size().max()),
        'return_reconciliation': reconciliation,
        'oof_residual_cause_counts': cause[oof].value_counts().to_dict(),
        'holdout_read': False, 'labels_read': False, 'rd_agent_allowed': False,
        'economic_gate': 'NOT_EVALUATED; new universe requires separate economic protocol freeze',
        'original_gate_unchanged': 'KEEP_1D_BASELINE; D21-v2 INSUFFICIENT_EVIDENCE',
        'data_limitations': 'INDATE/OUTDATE effective history; announcement availability not independently verified',
        'file_hashes': {p.name: fingerprint(p) for p in sorted(OUTPUT.glob('*.parquet'))},
    }
    write_json(OUTPUT / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'file_hashes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['fetch', 'evaluate'])
    args = parser.parse_args()
    fetch() if args.action == 'fetch' else evaluate()
