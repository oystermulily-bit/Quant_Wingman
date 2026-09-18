import numpy as np
import pandas as pd

from research_d21_v3_preflight import SCENARIOS, residuals, signals


def fixture():
    rows = []
    for t, date in enumerate(pd.bdate_range('2020-01-01', periods=150)):
        for code in range(10):
            rows.append((date, str(code), 'small' if code < 2 else 'large', .01*np.sin(t+code)))
    return pd.DataFrame(rows, columns=['date','code','industry_code','r_1d'])


def test_shrink_reference_excludes_target_and_respects_missing_industry():
    d = fixture().iloc[:10].copy()
    base = residuals(d)
    d.loc[0,'r_1d'] += .5
    changed = residuals(d)
    assert np.isclose(changed.loc[0,'shrink'] - base.loc[0,'shrink'], .5)
    assert base.loc[:1,'strict'].isna().all()
    assert base['shrink'].notna().all()
    d.loc[0,'industry_code'] = None
    assert pd.isna(residuals(d).loc[0,'shrink'])


def test_future_sentinel_and_rolling_history():
    d = fixture()
    cutoff = sorted(d.date.unique())[100]
    prefix = residuals(d[d.date <= cutoff])
    d.loc[d.date > cutoff, 'r_1d'] *= 1000000
    full = residuals(d)
    for spec in SCENARIOS:
        left = signals(prefix.pivot(index='date',columns='code',values=spec[1]), spec)
        right = signals(full.pivot(index='date',columns='code',values=spec[1]), spec)
        for key in left:
            pd.testing.assert_frame_equal(left[key], right[key].loc[:cutoff], check_exact=True)
    matrix = full.pivot(index='date',columns='code',values='shrink')
    matrix.loc[:, '0'] = np.nan
    assert signals(matrix, SCENARIOS[-1])['momentum']['0'].isna().all()
