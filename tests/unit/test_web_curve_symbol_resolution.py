from types import SimpleNamespace

import pandas as pd

import web.app as web_app


def test_overview_uses_internal_symbol_for_qfq_curve(tmp_path, monkeypatch):
    data_path = tmp_path / "601899_qfq_daily.parquet"
    bars = 3000
    pd.DataFrame(
        {
            "交易日期": pd.date_range("2025-01-01", periods=bars, freq="D"),
            "股票代码": ["601899.SH"] * bars,
            "开盘": [10.0] * bars,
            "最高": [10.2] * bars,
            "最低": [9.8] * bars,
            "收盘": [10.1] * bars,
            "成交量": [1000] * bars,
        }
    ).to_parquet(data_path)

    requested_symbols = []
    history = {
        "step": [0, 1, 2, 3],
        "best_score": [0.29] * 4,
        "val_score": [-0.62, -0.33, -0.25, -0.27],
        "generator_backend": ["rd_agent"] * 4,
    }

    def fake_progress(symbol):
        requested_symbols.append(symbol)
        return SimpleNamespace(
            symbol=symbol,
            current_step=4,
            train_steps=30,
            best_score=0.29,
            formula_decoded="RET",
            status="in_progress",
            history=history,
            checkpoint_path="checkpoints/ckpt_601899.SH_rd_step_0004.pt",
            has_strategy=True,
        )

    monkeypatch.setattr(
        web_app,
        "load_settings",
        lambda: {"last_data_file": str(data_path)},
    )
    monkeypatch.setattr(
        web_app.training_manager,
        "status",
        lambda: {"active": False, "job": None},
    )
    monkeypatch.setattr(web_app, "get_symbol_progress", fake_progress)
    monkeypatch.setattr(
        web_app,
        "_attach_training_time",
        lambda row, **_kwargs: row,
    )

    overview = web_app.api_overview()

    assert overview["data_file"]["symbol"] == "601899.SH"
    assert overview["progress"]["symbol"] == "601899.SH"
    assert overview["progress"]["current_step"] == 4
    assert requested_symbols == ["601899.SH"]
