import numpy as np
import pandas as pd

from research_d21_v3_reference_only import build
from data_pipeline.hs300.input_timing import REQUIRED_TIMES
from data_pipeline.hs300.time_policy import available_at


def fixture():
    rows = []
    for t, date in enumerate(pd.bdate_range('2020-01-01', periods=150)):
        for i in range(8):
            origin = 'original_csi300' if i < 3 or (i == 3 and t >= 100) else 'added_csi500_tech'
            rows.append((date, str(i), 'sector', .01 * np.sin(t + i), origin))
    frame = pd.DataFrame(rows, columns=['date', 'code', 'industry_code', 'r_1d', 'pool_origin'])
    for name in REQUIRED_TIMES:
        frame[name] = available_at(frame.date)
    return frame


def test_reference_only_never_enters_rank_or_prewarm():
    frame = fixture()
    out = build(frame)
    assert set(out.code) == {'0', '1', '2', '3'}
    assert out.loc[out.code.eq('3'), 'RES_MOM_120_20'].isna().all()
    assert out.loc[out.code.eq('3'), 'slow_residual_ensemble'].isna().all()
    valid = out[out.valid_slow_signal]
    assert len(valid) > 0
    np.testing.assert_allclose(valid.slow_residual_ensemble,
        valid[['IDIO_LOW_VOL_60_rank', 'RES_MOM_120_20_rank', 'RES_TREND_EFF_60_5_rank']].mean(axis=1))
    for name in ['IDIO_LOW_VOL_60', 'RES_MOM_120_20', 'RES_TREND_EFF_60_5']:
        assert out.groupby('date')[name + '_rank'].max().dropna().le(1).all()


def test_future_sentinel_and_reference_effect():
    frame = fixture()
    cutoff = frame.date.unique()[125]
    before = build(frame[frame.date <= cutoff])
    changed = frame.copy()
    changed.loc[changed.date > cutoff, 'r_1d'] += 1e6
    after = build(changed)
    pd.testing.assert_frame_equal(before.reset_index(drop=True),
                                  after[after.date <= cutoff].reset_index(drop=True))
    changed = frame.copy()
    changed.loc[changed.code.eq('7'), 'r_1d'] += .1
    other = build(changed)
    assert not np.allclose(build(frame)['epsilon'], other['epsilon'])
