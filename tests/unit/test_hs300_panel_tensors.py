from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.hs300.artifacts import write_holdout_dates
from data_pipeline.hs300.panel_tensors import pack_panel_tensors
from data_pipeline.hs300_panel import HS300PanelDataManager


def _tiny_panel(dates: int = 6, symbols: int = 3) -> pd.DataFrame:
    calendar = pd.bdate_range("2020-01-02", periods=dates)
    rows = []
    for date_idx, date in enumerate(calendar):
        available = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21)
        for code_idx, code in enumerate(["000001.SZ", "000002.SZ", "600000.SH"][:symbols]):
            quoted = not (date_idx == 1 and code == "000002.SZ")
            base = 10.0 + code_idx + 0.1 * date_idx
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open_raw": base if quoted else np.nan,
                    "high_raw": (base + 0.2) if quoted else np.nan,
                    "low_raw": (base - 0.2) if quoted else np.nan,
                    "close_raw": (base + 0.05) if quoted else np.nan,
                    "volume": 1_000_000 if quoted else np.nan,
                    "amount": 10_000_000 if quoted else np.nan,
                    "open_tr": 1.0 + 0.01 * date_idx if quoted else np.nan,
                    "high_tr": 1.02 + 0.01 * date_idx if quoted else np.nan,
                    "low_tr": 0.98 + 0.01 * date_idx if quoted else np.nan,
                    "close_tr": 1.01 + 0.01 * date_idx if quoted else np.nan,
                    "has_quote": quoted,
                    "can_buy_open": quoted,
                    "can_sell_open": quoted,
                    "industry_code": "801180.SI" if code_idx < 2 else "801010.SI",
                    "available_at": available,
                }
            )
    return pd.DataFrame(rows)


def test_pack_panel_tensors_keeps_nan_and_daily_membership() -> None:
    panel = _tiny_panel()
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    tensors = pack_panel_tensors(panel, dates, expected_members=3)
    assert tensors.features.shape == (3, 0, 6)
    assert tensors.membership_mask.sum(axis=0).tolist() == [3] * 6
    idx_000002 = list(tensors.symbols).index("000002.SZ")
    assert not bool(tensors.quote_valid_mask[idx_000002, 1])
    assert np.isnan(tensors.raw_ohlcv[idx_000002, :, 1]).all()
    assert np.isnan(tensors.amount[idx_000002, 1])
    assert np.isnan(tensors.causal_prices[idx_000002, :, 1]).all()
    assert not bool(tensors.buyable_mask[idx_000002, 1])
    assert np.isfinite(tensors.raw_ohlcv[idx_000002, 0, 0])
    quoted = np.broadcast_to(tensors.quote_valid_mask[:, None, :], tensors.raw_ohlcv.shape)
    assert np.isfinite(tensors.raw_ohlcv[quoted]).all()
    assert np.isnan(tensors.raw_ohlcv[~quoted]).all()
    assert tensors.industry_id.min() == 0


def test_load_labels_refuses_holdout_request(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    splits = tmp_path / "splits"
    splits.mkdir()
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02", "2024-08-26"]),
            "code": ["000001.SZ", "000001.SZ"],
            "value": [0.01, 0.02],
        }
    )
    frame.to_parquet(labels_dir / "development_labels.parquet", index=False)
    (splits / "holdout.lock.json").write_text(
        json.dumps(
            {
                "holdout_start": "2024-08-26",
                "holdout_date_count": 1,
                "holdout_date_hash": "abc",
                "readable_metrics": False,
            }
        ),
        encoding="utf-8",
    )
    manager = HS300PanelDataManager(tmp_path, expected_members=300, required_start=None)
    with pytest.raises(PermissionError, match="Holdout labels"):
        manager.load_labels(development_only=False)
    with pytest.raises(PermissionError, match="holdout dates"):
        manager.load_labels(development_only=True)


def test_write_holdout_dates_matches_protocol_hash(tmp_path: Path) -> None:
    dates = pd.bdate_range("2010-01-04", periods=900)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "is_trading_day": True,
            "is_complete_session": True,
        }
    )
    info = write_holdout_dates(tmp_path, calendar)
    holdout = pd.read_parquet(tmp_path / "splits" / "holdout_dates.parquet")
    assert len(holdout) == info["holdout_date_count"]
    assert holdout["role"].eq("HOLDOUT").all()
    assert info["holdout_date_count"] >= 480
    roles = pd.read_parquet(tmp_path / "splits" / "date_roles.parquet")
    assert set(roles["role"]) >= {"TRAIN", "PURGE", "OOF_VALIDATION", "HOLDOUT"}
