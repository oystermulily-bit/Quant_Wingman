import numpy as np
import pandas as pd
import pytest

from data_pipeline.hs300.input_timing import REQUIRED_TIMES, mask_unavailable
from data_pipeline.hs300_panel import HS300PanelDataManager
from research_d21_v3_reference_only import build


def sample(days=1):
    rows = [(d, c, 's', r, origin) for d in pd.bdate_range('2020-01-01', periods=days)
            for c, r, origin in [('A', .01, 'original_csi300'), ('B', .02, 'added_csi500_tech'),
                                 ('C', .03, 'added_csi500_tech')]]
    frame = pd.DataFrame(rows, columns=['date', 'code', 'industry_code', 'r_1d', 'pool_origin'])
    time = frame.date.dt.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=21)
    for column in REQUIRED_TIMES:
        frame[column] = time
    return frame


@pytest.mark.parametrize('column', REQUIRED_TIMES)
@pytest.mark.parametrize('missing', [False, True])
def test_late_or_missing_reference_cannot_influence_residual(column, missing):
    frame = sample()
    row = frame.code.eq('C')
    frame.loc[row, column] = pd.NaT if missing else frame.loc[row, column] + pd.Timedelta(hours=2)
    before = build(frame)
    frame.loc[row, 'r_1d'] = 999
    after = build(frame)
    np.testing.assert_allclose(before.epsilon, after.epsilon)
    assert before.r_1d.notna().all()  # original target remains usable
    assert mask_unavailable(frame).loc[row, 'r_1d'].isna().all()


def test_exact_cutoff_and_equivalent_timezone_are_allowed():
    frame = sample()
    for column in REQUIRED_TIMES:
        frame[column] = frame[column].dt.tz_convert('UTC')
    assert mask_unavailable(frame).reference_input_usable.all()
    frame['quote_available_at'] = frame.quote_available_at.astype('datetime64[ns, UTC]')
    frame.loc[2, 'quote_available_at'] += pd.Timedelta(nanoseconds=1)
    assert not mask_unavailable(frame).loc[2, 'reference_input_usable']


def test_missing_schema_and_naive_timestamps_fail_closed():
    frame = sample()
    with pytest.raises(ValueError, match='Missing availability'):
        build(frame.drop(columns=['status_available_at']))
    frame['quote_available_at'] = frame.quote_available_at.dt.tz_localize(None)
    with pytest.raises(ValueError, match='explicit timezone'):
        build(frame)


def test_target_row_preserved_and_unavailable_history_not_backfilled():
    frame = sample(125)
    target = frame.code.eq('A')
    frame.loc[target, 'quote_available_at'] += pd.Timedelta(hours=2)
    out = build(frame)
    assert len(out) == 125
    assert out.r_1d.isna().all() and out.slow_residual_ensemble.isna().all()
    frame = sample(125)
    last = frame.date.max()
    frame.loc[target & frame.date.eq(last), 'membership_available_at'] += pd.Timedelta(hours=2)
    out = build(frame)
    assert len(out) == 125 and not out[out.date.eq(last)].valid_slow_signal.any()


def test_industry_late_same_day_and_explicit_signal_time():
    date = pd.Timestamp('2020-01-02')
    panel = pd.DataFrame({'date': [date], 'code': ['A']})
    history = pd.DataFrame({'code': ['A'], 'valid_from': [date], 'valid_to': [pd.NaT],
        'industry_code': ['s'], 'known_at': [pd.Timestamp('2020-01-02 23:00', tz='Asia/Shanghai')],
        'effective_at': [date], 'known_at_source': ['NATIVE']})
    assert HS300PanelDataManager._join_industry_intervals(panel, history).industry_code.isna().all()
    history['known_at'] = pd.Timestamp('2020-01-02 21:00', tz='Asia/Shanghai')
    assert HS300PanelDataManager._join_industry_intervals(panel, history).industry_code.notna().all()
    panel['signal_at'] = pd.Timestamp('2020-01-02 20:00', tz='Asia/Shanghai')
    assert HS300PanelDataManager._join_industry_intervals(panel, history).industry_code.isna().all()


def test_untrusted_or_inconsistent_signal_time_rejected():
    frame = sample()
    frame['signal_at'] = frame.date.dt.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=21)
    frame.loc[2, 'signal_at'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='share a signal_at'):
        build(frame)


def test_aggregate_timestamp_cannot_override_dependencies():
    frame = sample()
    frame['available_at'] = frame.quote_available_at
    frame.loc[2, 'preclose_available_at'] += pd.Timedelta(days=1)
    assert not mask_unavailable(frame).loc[2, 'reference_input_usable']
