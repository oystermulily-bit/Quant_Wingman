from __future__ import annotations

import pandas as pd

from data_pipeline.hs300.industry_stats import prefix_loo_invariant
from data_pipeline.hs300.opentr import build_preclose_chain, prefix_matches


def test_opentr_prefix_is_unchanged_when_future_rows_are_appended() -> None:
    dates = pd.bdate_range("2014-01-02", periods=12)
    close = 10.0
    rows = []
    preclose = 10.0
    for date in dates:
        open_px = close
        high = close + 0.3
        low = close - 0.2
        new_close = close * 1.01
        rows.append(
            {
                "date": date,
                "code": "000001.SZ",
                "open": open_px,
                "high": high,
                "low": low,
                "close": new_close,
                "preclose": preclose,
            }
        )
        preclose = new_close
        close = new_close
    full = build_preclose_chain(pd.DataFrame(rows))
    cut = dates[6]
    assert prefix_matches(full, dates[dates <= cut], "000001.SZ")


def test_industry_loo_prefix_does_not_use_future_members() -> None:
    dates = pd.bdate_range("2020-01-02", periods=10)
    rows = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(["000001.SZ", "000002.SZ", "600000.SH"]):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry_code": "801180.SI" if code_idx < 2 else "801010.SI",
                    "close_tr": 10.0 + date_idx + 0.25 * code_idx,
                    "has_quote": True,
                }
            )
    panel = pd.DataFrame(rows)
    assert prefix_loo_invariant(panel, dates[5])
