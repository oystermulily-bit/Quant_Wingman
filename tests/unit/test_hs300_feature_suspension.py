from __future__ import annotations

import numpy as np

from data_pipeline.hs300.features_wingman import compute_wingman_feature_tensors
from data_pipeline.hs300.panel_tensors import HS300PanelTensors


def _panel(*, pre_gap_scale: float) -> HS300PanelTensors:
    bars = 24
    gap = 9
    close = np.linspace(10.0, 13.0, bars, dtype=np.float32)
    close[:gap] *= pre_gap_scale
    causal = np.stack((close, close + 0.2, close - 0.2, close), axis=0)[None, ...]
    volume = np.linspace(100.0, 200.0, bars, dtype=np.float32)
    raw = np.concatenate((causal, volume[None, None, :]), axis=1)
    quoted = np.ones((1, bars), dtype=bool)
    quoted[:, gap] = False
    causal[:, :, gap] = np.nan
    raw[:, :, gap] = np.nan
    membership = np.ones((1, bars), dtype=bool)
    return HS300PanelTensors(
        dates=np.arange(bars).astype("datetime64[D]"),
        symbols=np.array(["000001.SZ"], dtype=object),
        raw_ohlcv=raw,
        amount=np.ones((1, bars), dtype=np.float32),
        causal_prices=causal,
        membership_mask=membership,
        quote_valid_mask=quoted,
        buyable_mask=quoted.copy(),
        sellable_mask=quoted.copy(),
        industry_id=np.zeros((1, bars), dtype=np.int32),
        features=np.zeros((1, 0, bars), dtype=np.float32),
        industry_codes=np.array(["TEST"], dtype=object),
    )


def test_suspension_never_becomes_zero_observation_or_leaks_across_gap() -> None:
    baseline, baseline_valid, _ = compute_wingman_feature_tensors(
        _panel(pre_gap_scale=1.0), warmup_bars=2
    )
    changed, changed_valid, _ = compute_wingman_feature_tensors(
        _panel(pre_gap_scale=1000.0), warmup_bars=2
    )

    gap = 9
    assert np.isnan(baseline[:, :, gap : gap + 3]).all()
    assert not baseline_valid[:, :, gap : gap + 3].any()
    np.testing.assert_array_equal(baseline_valid[:, :, gap + 3 :], changed_valid[:, :, gap + 3 :])
    np.testing.assert_allclose(
        baseline[:, :, gap + 3 :],
        changed[:, :, gap + 3 :],
        equal_nan=True,
    )
