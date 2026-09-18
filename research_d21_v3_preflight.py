"""Label-free coverage diagnostics; never opens sealed data or changes gates.

The four alternatives are fixed before execution, not selected by PnL.
Non-member history stays missing (option 2 was not authorized).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/stage3r_d21_v2_h2_5d'
OUTPUT = ROOT / 'experiments/d21_v3_coverage_preflight_20260914'
SCENARIOS = (
    ('baseline', 'strict', 60, 54, 120, 20, 90, 60, 5, 50),
    ('shrink_only', 'shrink', 60, 54, 120, 20, 90, 60, 5, 50),
    ('short_only', 'strict', 20, 18, 60, 10, 45, 20, 2, 17),
    ('shrink_and_short', 'shrink', 20, 18, 60, 10, 45, 20, 2, 17),
)


def residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Target and peers must have finite returns and a known PIT industry.

    For n<5 peers: w=n/5; reference=w*industry_LOO+(1-w)*market_LOO.
    n=0 uses market_LOO; missing industry is NOT silently mapped to market.
    """
    out = frame.copy()
    r = out['r_1d'].where(np.isfinite(out['r_1d']))
    known = out['industry_code'].notna()
    industry_r = r.where(known)
    group = industry_r.groupby([out['date'], out['industry_code']])
    n = group.transform('count') - industry_r.notna().astype(int)
    total = group.transform('sum') - industry_r.fillna(0)
    ind = total / n.where(n.gt(0))
    market = r.groupby(out['date'])
    market_n = market.transform('count') - r.notna().astype(int)
    market_mean = (market.transform('sum') - r.fillna(0)) / market_n.where(market_n.gt(0))
    eligible = known & r.notna()
    out['peer_count'] = n
    out['strict'] = (r - ind).where(eligible & n.ge(5))
    w = (n / 5).clip(0, 1)
    blended = w * ind.fillna(0) + (1 - w) * market_mean
    reference = ind.where(n.ge(5), blended)
    out['shrink'] = (r - reference).where(eligible)
    return out


def signals(matrix: pd.DataFrame, spec: tuple) -> dict[str, pd.DataFrame]:
    _, _, vol, vol_min, mom, skip, mom_min, trend, trend_skip, trend_min = spec
    old = matrix.shift(skip)
    trend_old = matrix.shift(trend_skip)
    return {
        'low_vol': -matrix.rolling(vol, min_periods=vol_min).std(ddof=0),
        'momentum': old.rolling(mom-skip, min_periods=mom_min).sum()
        * (mom-skip) / old.rolling(mom-skip, min_periods=mom_min).count(),
        'efficiency': trend_old.rolling(trend-trend_skip, min_periods=trend_min).sum()
        / (trend_old.abs().rolling(trend-trend_skip, min_periods=trend_min).sum() + 1e-12),
    }


def fingerprint(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError('Preflight output already exists; do not overwrite evidence')
    split_path = SOURCE / 'split_plan.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    source_path = SOURCE / 'development_features_slow.parquet'
    columns = ['date', 'code', 'industry_code', 'r_1d', 'epsilon', 'valid_slow_signal']
    raw = pd.read_parquet(source_path, columns=columns)
    raw['date'] = pd.to_datetime(raw['date'])
    if raw.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate date/code')
    if not raw['date'].between(split['development_start'], split['development_end']).all():
        raise ValueError('Input contains dates outside Development')
    daily_size = raw.groupby('date').size()
    if not daily_size.eq(300).all():
        raise ValueError('Expected 300 PIT member rows per day')
    frame = residuals(raw)
    np.testing.assert_allclose(frame['strict'], frame['epsilon'], rtol=1e-10, atol=1e-12, equal_nan=True)
    dates = pd.DatetimeIndex(sorted(frame['date'].unique()))
    codes = pd.Index(sorted(frame['code'].unique()))
    di, ci = dates.get_indexer(frame['date']), codes.get_indexer(frame['code'])
    fold_masks = [frame['date'].between(f['validation_start'], f['validation_end']) for f in split['folds']]
    oof = np.logical_or.reduce(fold_masks)
    results = []
    detail = frame[['date', 'code', 'industry_code', 'peer_count']].copy()
    for spec in SCENARIOS:
        matrix = frame.pivot(index='date', columns='code', values=spec[1]).reindex(index=dates, columns=codes)
        values = signals(matrix, spec)
        valid = np.logical_and.reduce([np.isfinite(x.to_numpy()[di, ci]) for x in values.values()])
        if spec[0] == 'baseline':
            np.testing.assert_array_equal(valid, raw['valid_slow_signal'].to_numpy())
        detail[spec[0]] = valid
        daily = pd.Series(valid[oof], index=frame.loc[oof, 'date']).groupby(level=0).sum()
        results.append({
            'name': spec[0], 'parameters': list(spec[1:]), 'coverage': float(valid[oof].mean()),
            'coverage_passed': bool(valid[oof].mean() >= .95),
            'fold_coverage': [float(valid[m].mean()) for m in fold_masks],
            'daily_valid_min': int(daily.min()), 'daily_valid_median': float(daily.median()),
            'daily_valid_max': int(daily.max()),
            'signal_coverage': {k: float(np.isfinite(v.to_numpy()[di, ci])[oof].mean()) for k,v in values.items()},
        })
    missing_ind = frame['industry_code'].isna()
    missing_r = ~np.isfinite(frame['r_1d'])
    causes = pd.Series('valid', index=frame.index)
    causes.loc[missing_ind] = 'missing_industry'
    causes.loc[~missing_ind & missing_r] = 'missing_return'
    causes.loc[~missing_ind & ~missing_r & frame['peer_count'].lt(5)] = 'fewer_than_5_peers'
    detail['residual_reason'] = causes
    report = {
        'status': 'PREFLIGHT_COMPLETED_NOT_A_GO_DECISION', 'development_adaptive': True,
        'holdout_read': False, 'labels_read': False, 'rd_agent_allowed': False,
        'threshold': .95, 'oof_member_days': int(oof.sum()), 'oof_dates': int(frame.loc[oof,'date'].nunique()),
        'coverage_denominator': 'all PIT member rows in original split validation dates; not PnL ledger dates',
        'source_sha256': fingerprint(source_path), 'split_sha256': fingerprint(split_path),
        'code_sha256': fingerprint(Path(__file__)), 'residual_missing_counts': causes[oof].value_counts().to_dict(),
        'scenarios': results, 'option1': 'BLOCKED_NO_VERIFIED_FULL_A_SHARE_PIT_SNAPSHOT_OR_MARKET_CONNECTOR',
        'option3': 'AUDITED_NO_VERIFIED_REPAIR_SOURCE; no fill applied',
        'option2': 'EXCLUDED; non-member residual history remains missing',
        'freeze_status': 'CANDIDATE_SPEC_ONLY; v3 economic protocol not frozen',
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    detail.loc[oof].to_parquet(OUTPUT / 'coverage_evidence.parquet', index=False)
    by_industry = detail.loc[oof].assign(industry_code=detail.loc[oof,'industry_code'].fillna('MISSING')).groupby('industry_code').agg(
        member_days=('code','size'), **{s[0]:(s[0],'mean') for s in SCENARIOS})
    by_industry.to_json(OUTPUT / 'industry_coverage.json', orient='index', indent=2)
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
