from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.hs300.industry import build_industry_membership
from data_pipeline.hs300.industry_stats import industry_loo_context, prefix_loo_invariant
from data_pipeline.hs300.membership import build_universe_membership
from data_pipeline.hs300.opentr import build_preclose_chain, prefix_matches
from data_pipeline.hs300.tradability import build_trading_status


def test_industry_loo_is_prefix_invariant_and_excludes_self() -> None:
    dates = pd.bdate_range("2020-01-02", periods=8)
    rows = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(["000001.SZ", "000002.SZ", "600000.SH"]):
            close = 10.0 + date_idx + 0.5 * code_idx
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry_code": "801180.SI" if code_idx < 2 else "801010.SI",
                    "close_tr": close,
                    "has_quote": True,
                }
            )
    panel = pd.DataFrame(rows)
    context, stock = industry_loo_context(panel)
    two = stock[(stock["date"] == dates[2]) & (stock["industry_code"] == "801180.SI")]
    assert len(two) == 2
    assert two["loo_member_count"].eq(1).all()
    assert prefix_loo_invariant(panel, dates[4])


def test_opentr_first_day_and_preclose_chain() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-01-02", "2014-01-03", "2014-01-06"]),
            "code": ["000001.SZ"] * 3,
            "open": [10.0, 10.2, 9.0],
            "high": [10.5, 10.4, 9.2],
            "low": [9.8, 10.0, 8.8],
            "close": [10.1, 10.3, 9.1],
            "preclose": [10.0, 10.1, 5.15],
        }
    )
    out = build_preclose_chain(bars)
    assert out.loc[0, "close_tr"] == pytest.approx(1.0)
    assert out.loc[0, "open_tr"] == pytest.approx(1.0)
    assert out.loc[0, "high_tr"] == pytest.approx(10.5 / 10.0)
    assert out.loc[0, "low_tr"] == pytest.approx(9.8 / 10.0)
    assert out.loc[1, "open_tr"] == pytest.approx(1.0 * 10.2 / 10.1)
    assert out.loc[1, "close_tr"] == pytest.approx(1.0 * 10.3 / 10.1)
    assert out.loc[1, "high_tr"] == pytest.approx(1.0 * 10.4 / 10.1)
    assert out.loc[2, "close_tr"] == pytest.approx(out.loc[1, "close_tr"] * 9.1 / 5.15)
    assert prefix_matches(out, bars["date"].iloc[:2], "000001.SZ")


def test_opentr_breaks_when_preclose_missing() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-01-02", "2014-01-03", "2014-01-06"]),
            "code": ["000001.SZ"] * 3,
            "open": [10.0, 10.2, 10.4],
            "high": [10.5, 10.4, 10.6],
            "low": [9.8, 10.0, 10.2],
            "close": [10.1, 10.3, 10.5],
            "preclose": [10.0, 0.0, 10.3],
        }
    )
    out = build_preclose_chain(bars)
    assert bool(out.loc[0, "chain_valid"]) is True
    assert bool(out.loc[1, "chain_valid"]) is False
    assert pd.isna(out.loc[1, "open_tr"])
    assert out.loc[2, "close_tr"] == pytest.approx(1.0)


def test_membership_requires_300_and_keeps_exceptions() -> None:
    dates = pd.to_datetime(["2010-01-04", "2010-01-05"])
    rows = []
    for date in dates:
        for idx in range(300):
            rows.append(
                {
                    "INDEX_CODE": "000300.SH",
                    "CON_CODE": f"{idx:06d}.SZ",
                    "TRADE_DATE": date.strftime("%Y%m%d"),
                    "WEIGHT": 100.0 / 300.0,
                }
            )
    rows.append(
        {
            "INDEX_CODE": "000300.SH",
            "CON_CODE": "000000.SZ",
            "TRADE_DATE": "20100106",
            "WEIGHT": 100.0,
        }
    )
    members, exceptions = build_universe_membership(pd.DataFrame(rows))
    assert members.groupby("date")["code"].nunique().loc[pd.Timestamp("2010-01-04")] == 300
    assert not exceptions.empty
    assert "member_count=1" in str(exceptions["reason"].iloc[-1])


def test_limit_up_open_is_not_buyable() -> None:
    status = pd.DataFrame(
        {
            "MARKET_CODE": ["000001.SZ"],
            "TRADE_DATE": ["20140102"],
            "PRECLOSE": [10.0],
            "HIGH_LIMITED": [11.0],
            "LOW_LIMITED": [9.0],
            "IS_ST_SEC": [0],
            "IS_SUSP_SEC": [0],
            "IS_WD_SEC": [0],
            "IS_XR_SEC": [0],
        }
    )
    bars = pd.DataFrame(
        {
            "date": [pd.Timestamp("2014-01-02")],
            "code": ["000001.SZ"],
            "open": [11.0],
            "volume": [1_000_000],
            "amount": [11_000_000],
        }
    )
    out = build_trading_status(status, bars)
    assert bool(out.loc[0, "can_buy_open"]) is False
    assert out.loc[0, "buy_block_reason"] == "LIMIT_UP_OPEN"
    assert bool(out.loc[0, "can_sell_open"]) is True


def test_industry_interval_does_not_use_indate_as_native_known_at() -> None:
    base = pd.DataFrame(
        {
            "INDEX_CODE": ["801180.SI"],
            "LEVEL_TYPE": [1],
            "LEVEL1_NAME": ["房地产"],
        }
    )
    constituents = pd.DataFrame(
        {
            "INDEX_CODE": ["801180.SI"],
            "CON_CODE": ["000002.SZ"],
            "INDATE": [20100104],
            "OUTDATE": [20151231],
        }
    )
    out = build_industry_membership(
        constituents, base, ["000002.SZ"], expected_l1_count=1
    )
    membership, exceptions = out
    assert exceptions.empty
    assert membership.loc[0, "known_at_source"] == "DERIVED_POLICY"
    assert pd.notna(membership.loc[0, "known_at"])
    assert membership.loc[0, "valid_from"] == pd.Timestamp("2010-01-04")
