import numpy as np
import pandas as pd
import pytest

from research_d21_v3_tech_expansion import additions, expand_intervals, normalize_returns


def test_pit_intervals_and_csi300_exclusion():
    dates = pd.date_range('2020-01-01', periods=3)
    history = pd.DataFrame({'code':['A','B'], 'start':['20200102','20200101'], 'end':[None,'20200102']})
    idx = expand_intervals(history, dates, 'code', 'start', 'end')
    assert not ((idx.code == 'A') & (idx.date == dates[0])).any()
    assert not ((idx.code == 'B') & (idx.date == dates[2])).any()
    tech = idx.assign(industry_code='tech')
    original = pd.DataFrame({'date':[dates[1]], 'code':['A']})
    added = additions(idx, tech, original)
    assert len(added) == len(idx)-1
    assert ((added.code == 'A') & (added.date == dates[2])).any()


def test_bad_end_and_conflicting_industry_fail_closed():
    dates = pd.date_range('2020-01-01', periods=2)
    bad = pd.DataFrame({'code':['A'], 'start':['20200101'], 'end':['wrong']})
    with pytest.raises(ValueError):
        expand_intervals(bad, dates, 'code', 'start', 'end')
    conflict = pd.DataFrame({'code':['A','A'], 'start':['20200101']*2, 'end':[None]*2, 'industry':['X','Y']})
    with pytest.raises(ValueError):
        expand_intervals(conflict, dates, 'code', 'start', 'end', ('industry',))


def test_return_definition_missing_previous_quote_and_future_guard():
    dates = pd.date_range('2020-01-01', periods=4)
    bars = pd.DataFrame({'MARKET_CODE':['A']*3, 'kline_time':dates[[0,1,3]], 'close':[10,11,12]})
    status = pd.DataFrame({'MARKET_CODE':['A']*3, 'TRADE_DATE':['20200101','20200102','20200104'], 'PRECLOSE':[9,10,11]})
    result = normalize_returns(bars, status, dates)
    assert np.isnan(result.r_1d.iloc[0]) and np.isnan(result.r_1d.iloc[2])
    assert np.isclose(result.r_1d.iloc[1], np.log(1.1))
    pd.testing.assert_frame_equal(result, normalize_returns(bars.assign(code='A'), status, dates))
    with pytest.raises(ValueError):
        normalize_returns(bars.assign(code='B'), status, dates)
    with pytest.raises(ValueError):
        normalize_returns(bars, status, dates[:2])
