from __future__ import annotations

import pandas as pd

from data_pipeline import parquet_manager as pm


def test_chinese_qfq_schema_loads_and_uses_internal_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(pm.Config, "MIN_BARS", 3)
    path = tmp_path / "601899_qfq_daily.parquet"
    frame = pd.DataFrame({
        "日期": pd.date_range("2026-07-20", periods=6, freq="B"),
        "股票代码": ["601899.SH"] * 6,
        "开盘": [20.0, 20.2, 20.1, 20.4, 20.8, 21.0],
        "最高": [20.4, 20.5, 20.4, 20.9, 21.1, 21.3],
        "最低": [19.8, 20.0, 19.9, 20.2, 20.6, 20.8],
        "收盘": [20.2, 20.1, 20.3, 20.8, 21.0, 21.2],
        "成交量_股": [1000, 1200, 900, 1500, 1300, 1600],
    })
    frame.to_parquet(path, index=False)

    info = pm.inspect_parquet_file(path)
    assert info["symbol"] == "601899.SH"
    assert info["timeframe"] == "D1"
    assert info["bars"] == 6

    manager = pm.ParquetDataManager(path)
    manager.load()
    assert manager.symbol == "601899.SH"
    assert manager.timeframe == "D1"
    assert manager.raw_dict["close"].shape == (1, 6)
    assert int(manager.raw_dict["time"][0, 0]) > 1_700_000_000
