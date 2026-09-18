from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_d21_v2.backtest import BufferedTop20Backtester
from research_d21_v2.gate import (
    classify_d21_v2,
    evaluate_d21_v2_go,
)
from research_d21_v2.protocol import D21V2Config
from research_d21_v2.runner import Stage3RResearchRunner
from research_d21_v2.signals import SlowResidualEnsembleBuilder
from research_stage3.backtest import ReferenceBacktester
from research_stage4.stats import BootstrapResult


def _signal_universe(dates: pd.DatetimeIndex, *, n_codes: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = [f"{idx:06d}.SZ" for idx in range(1, n_codes + 1)]
    industries = ["SW_A" if idx < n_codes // 2 else "SW_B" for idx in range(n_codes)]
    member_rows = []
    bar_rows = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(codes):
            close = 10.0 * np.exp(0.0004 * date_idx + 0.002 * code_idx)
            member_rows.append(
                {"date": date, "code": code, "industry_code": industries[code_idx]}
            )
            bar_rows.append(
                {
                    "date": date,
                    "code": code,
                    "close_tr": close,
                    "amount": 50_000_000.0,
                    "has_quote": True,
                }
            )
    return pd.DataFrame(member_rows), pd.DataFrame(bar_rows)


def test_config_hash_is_stable_and_rejects_unfrozen_knobs() -> None:
    first = D21V2Config().fingerprint()
    second = D21V2Config().fingerprint()
    assert first == second
    assert len(first) == 64
    with pytest.raises(ValueError, match="2×K"):
        D21V2Config(top_n=20, exit_rank=30)
    with pytest.raises(ValueError, match="horizon 5"):
        D21V2Config(target_horizon=3)
    with pytest.raises(ValueError, match="Holdout"):
        D21V2Config(holdout_read=True)


def test_industry_residual_missing_when_fewer_than_five_others() -> None:
    dates = pd.bdate_range("2020-01-02", periods=80)
    panel, bars = _signal_universe(dates)
    last = dates[-1]
    sw_a = [f"{idx:06d}.SZ" for idx in range(1, 7)]
    bars = bars.copy()
    drop = sw_a[:2]
    bars.loc[(bars["date"] == last) & (bars["code"].isin(drop)), "has_quote"] = False
    bars.loc[(bars["date"] == last) & (bars["code"].isin(drop)), ["close_tr", "amount"]] = np.nan
    built = SlowResidualEnsembleBuilder().build(panel, bars, dates)
    last_a = built[(built["date"] == last) & (built["code"].isin(sw_a))]
    assert last_a["epsilon"].isna().all()
    last_b = built[(built["date"] == last) & (~built["code"].isin(sw_a))]
    assert last_b["epsilon"].notna().all()
    missing_r = built["r_1d"].isna()
    assert not (built.loc[missing_r, "epsilon"] == 0).any()


def test_residual_momentum_skips_recent_twenty_days() -> None:
    dates = pd.bdate_range("2020-01-02", periods=200)
    panel, bars = _signal_universe(dates)
    baseline = SlowResidualEnsembleBuilder().build(panel, bars, dates)
    target = "000001.SZ"
    shocked = bars.copy()
    shocked.loc[
        (shocked["code"] == target) & (shocked["date"] > dates[-21]), "close_tr"
    ] *= 8.0
    altered = SlowResidualEnsembleBuilder().build(panel, shocked, dates)
    last = dates[-1]
    base_row = baseline[(baseline["date"] == last) & (baseline["code"] == target)].iloc[0]
    alt_row = altered[(altered["date"] == last) & (altered["code"] == target)].iloc[0]
    assert base_row["RES_MOM_120_20"] == pytest.approx(alt_row["RES_MOM_120_20"], rel=1e-10)
    assert base_row["IDIO_LOW_VOL_60"] != pytest.approx(alt_row["IDIO_LOW_VOL_60"], rel=1e-6)


def test_ensemble_requires_all_three_signals() -> None:
    dates = pd.bdate_range("2020-01-02", periods=200)
    panel, bars = _signal_universe(dates)
    built = SlowResidualEnsembleBuilder().build(panel, bars, dates)
    early = built[built["date"] == dates[80]]
    assert not bool(early["valid_slow_signal"].any())
    assert early["slow_residual_ensemble"].isna().all()
    late = built[built["date"] == dates[-1]]
    assert bool(late["valid_slow_signal"].all())
    mutated = late.copy()
    mutated.loc[mutated.index[0], "RES_MOM_120_20_rank"] = np.nan
    assert mutated.loc[mutated.index[0], ["IDIO_LOW_VOL_60_rank", "RES_TREND_EFF_60_5_rank"]].notna().all()
    recomputed = mutated[
        ["IDIO_LOW_VOL_60_rank", "RES_MOM_120_20_rank", "RES_TREND_EFF_60_5_rank"]
    ].mean(axis=1, skipna=False)
    assert np.isnan(recomputed.iloc[0])


def _rank_book(dates: pd.DatetimeIndex, codes: list[str], score_by_date: dict[pd.Timestamp, dict[str, float]]):
    feature_rows = []
    bar_rows = []
    status_rows = []
    member_rows = []
    for date in dates:
        scores = score_by_date[date]
        for code in codes:
            feature_rows.append(
                {
                    "date": date,
                    "code": code,
                    "valid_slow_signal": True,
                    "median_amount_20": 100.0,
                    "slow_residual_ensemble": scores[code],
                }
            )
            bar_rows.append({"date": date, "code": code, "open_tr": 10.0, "has_quote": True})
            status_rows.append(
                {"date": date, "code": code, "can_buy_open": True, "can_sell_open": True}
            )
            member_rows.append({"date": date, "code": code, "is_member": True})
    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(bar_rows),
        pd.DataFrame(status_rows),
        pd.DataFrame(member_rows),
    )


def test_buffer_holds_ranks_six_to_ten_and_does_not_refill_from_them() -> None:
    dates = pd.bdate_range("2024-01-02", periods=24)
    codes = [f"{idx:06d}.SZ" for idx in range(1, 16)]
    score_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for date in dates:
        if date < dates[5]:
            scores = {code: float(16 - idx) for idx, code in enumerate(codes, start=1)}
        elif date < dates[10]:
            # Former Top5 move to ranks 6-10; new names take Top5.
            scores = {}
            for idx, code in enumerate(codes, start=1):
                if idx <= 5:
                    scores[code] = float(11 - idx)  # 10..6
                elif idx <= 10:
                    scores[code] = float(16 - (idx - 5))  # 15..11
                else:
                    scores[code] = float(6 - (idx - 10))
        else:
            scores = {code: float(idx) for idx, code in enumerate(codes, start=1)}
        score_by_date[date] = scores
    features, bars, status, members = _rank_book(dates, codes, score_by_date)
    cfg = D21V2Config(
        top_n=5,
        exit_rank=10,
        target_weight=0.05,
        liquidity_median_amount_20d=0.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
    )
    buffered, daily_b = BufferedTop20Backtester(cfg).run(
        features,
        bars,
        status,
        dates,
        dates,
        horizon=5,
        phase=0,
        score_column="slow_residual_ensemble",
        valid_column="valid_slow_signal",
        membership=members,
    )
    naive_features = features.rename(columns={"valid_slow_signal": "valid_signal"})
    naive, daily_n = ReferenceBacktester(cfg.as_stage3_config()).run(
        naive_features,
        bars,
        status,
        dates,
        dates,
        horizon=5,
        phase=0,
        score_column="slow_residual_ensemble",
        membership=members,
    )
    second = dates[6]
    assert daily_b.loc[daily_b["date"] == second, "holding_count"].iloc[0] == 5
    assert daily_b.loc[daily_b["date"] == second, "turnover"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert daily_n.loc[daily_n["date"] == second, "turnover"].iloc[0] > 0.2
    third = dates[11]
    assert daily_b.loc[daily_b["date"] == third, "turnover"].iloc[0] > 0.2
    assert buffered.rebalance_count >= 3
    assert naive.rebalance_count >= 3

    blocked = features.copy()
    first_signal = dates[0]
    top_new = [f"{idx:06d}.SZ" for idx in range(4, 6)]
    blocked.loc[
        (blocked["date"] == first_signal) & (blocked["code"].isin(top_new)),
        "median_amount_20",
    ] = 0.0
    cfg_liq = D21V2Config(
        top_n=5,
        exit_rank=10,
        target_weight=0.05,
        liquidity_median_amount_20d=1.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
    )
    _metrics, daily_gap = BufferedTop20Backtester(cfg_liq).run(
        blocked,
        bars,
        status,
        dates,
        dates,
        horizon=5,
        phase=0,
        score_column="slow_residual_ensemble",
        valid_column="valid_slow_signal",
        membership=members,
    )
    first_exec = dates[1]
    assert daily_gap.loc[daily_gap["date"] == first_exec, "holding_count"].iloc[0] == 3
    assert daily_gap.loc[daily_gap["date"] == first_exec, "cash_weight"].iloc[0] == pytest.approx(0.85, abs=1e-6)


def test_d18_v2_absolute_gate_and_v1_untouched() -> None:
    boot = BootstrapResult(
        observed=0.2,
        ci_low=0.05,
        ci_high=0.4,
        p_value=0.001,
        samples=2000,
        seed=20260812,
        block_days=20,
    )
    cfg = D21V2Config()
    failed = evaluate_d21_v2_go(
        candidate={
            "median_sharpe": -0.10,
            "median_annualized_return": -0.04,
            "phases_agreeing": 5,
            "folds_agreeing": 5,
            "signals_agreeing": 3,
            "concentration_ok": True,
        },
        baseline={
            "median_sharpe": -0.40,
            "median_annualized_return": -0.10,
        },
        sharpe_bootstrap=boot,
        holm_da={"reject": True, "p_holm": 0.01},
        coverage=0.99,
        config=cfg,
    )
    assert failed["passed"] is False
    assert failed["checks"]["absolute_net_sharpe"]["passed"] is False
    assert failed["checks"]["sharpe_improvement_vs_A"]["passed"] is True
    passed = classify_d21_v2({"passed": True}, coverage_ok=True)
    assert passed["statistical_gate"] == "HORIZON_FEASIBLE_5D_V2"
    assert passed["rd_agent_allowed"] is False
    assert passed["d21_v1_gate_unchanged"] == "KEEP_1D_BASELINE"
    assert passed["holdout_read"] is False


def test_stage3r_refuses_to_overwrite_v1_ledger(tmp_path) -> None:
    with pytest.raises(PermissionError, match="must not overwrite"):
        Stage3RResearchRunner().run(
            tmp_path / "snapshot",
            tmp_path / "stage3_hs300_v2_execution_ledger",
        )


def test_stage3r_runner_writes_independent_ablation_artifacts(tmp_path) -> None:
    import importlib.util
    from pathlib import Path as _Path

    helper = _Path(__file__).with_name("test_stage3_research.py")
    spec = importlib.util.spec_from_file_location("stage3_test_helpers", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _make_snapshot = module._make_snapshot
    _test_config = module._test_config

    root = _make_snapshot(tmp_path / "snapshot", dates=180, symbols=12)
    output = tmp_path / "stage3r_d21_v2_h2_5d"
    config = D21V2Config(
        top_n=5,
        exit_rank=10,
        target_weight=0.05,
        liquidity_median_amount_20d=1.0,
        idio_vol_window=8,
        idio_vol_min_obs=6,
        res_mom_long=16,
        res_mom_skip=3,
        res_mom_min_obs=8,
        trend_window=10,
        trend_skip=2,
        trend_min_obs=6,
        min_industry_others=5,
    )
    report = Stage3RResearchRunner(config, split_config=_test_config()).run(
        root,
        output,
        expected_members=12,
        required_start=None,
    )
    assert report["status"] == "STAGE3R_D21_V2_COMPLETED"
    assert report["holdout_read"] is False
    assert report["rd_agent_allowed"] is False
    daily = pd.read_parquet(output / "development_oof_portfolio_daily.parquet")
    assert set(daily["experiment"]) == {"A", "B", "C", "D"}
    assert set(daily["horizon"]) == {5}
    assert daily["phase"].max() == 4
    features = pd.read_parquet(output / "development_features.parquet")
    assert "slow_residual_ensemble" in features.columns
    assert "simple_ensemble" in features.columns
    assert (output / "stage3r_report.json").is_file()
    assert not (tmp_path / "stage3_hs300_v2_execution_ledger").exists()
