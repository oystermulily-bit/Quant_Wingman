from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.hs300_panel import (
    HS300PanelDataManager,
    SNAPSHOT_MANIFEST_VERSION,
    SnapshotManifestValidator,
)
from research_stage3.backtest import ReferenceBacktester
from research_stage3.candidate_features import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_FEATURE_SPECS,
    CANDIDATE_FEATURE_VERSION,
)
from research_stage3.labels import LabelRegistry
from research_stage3.protocol import DateSplitProtocol, Stage3Config
from research_stage3.runner import Stage3ResearchRunner
from research_stage3.signals import SimpleSignalBuilder


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_snapshot(root: Path, *, dates: int = 180, symbols: int = 12) -> Path:
    root.mkdir(parents=True)
    raw_response = root / "raw_response.json"
    raw_response.write_text('{"ok":true}', encoding="utf-8")
    calendar_dates = pd.bdate_range("2020-01-02", periods=dates)
    codes = [f"{idx:06d}.SZ" for idx in range(1, symbols + 1)]
    member_rows = []
    bar_rows = []
    status_rows = []
    for date_idx, date in enumerate(calendar_dates):
        available = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21)
        for code_idx, code in enumerate(codes):
            base = 10.0 + code_idx + 0.015 * date_idx + 0.001 * code_idx * date_idx
            member_rows.append(
                {
                    "date": date,
                    "index_code": "000300.SH",
                    "code": code,
                    "is_member": True,
                    "weight_pct": 100.0 / symbols,
                    "effective_at": date.tz_localize("Asia/Shanghai"),
                    "known_at": pd.NaT,
                    "known_at_source": "MISSING",
                    "entry_effective_date": calendar_dates[0],
                    "membership_source": "DAILY_INDEX_WEIGHT",
                    "source_request_id": "synthetic-test",
                }
            )
            bar_rows.append(
                {
                    "date": date,
                    "code": code,
                    "open_raw": base,
                    "high_raw": base * 1.01,
                    "low_raw": base * 0.99,
                    "close_raw": base * (1.0 + 0.0005 * ((code_idx % 3) - 1)),
                    "volume": 1_000_000 + code_idx,
                    "amount": 50_000_000 + code_idx * 1_000_000,
                    "preclose": base - 0.01 if date_idx else base,
                    "open_tr": base,
                    "close_tr": base * (1.0 + 0.0005 * ((code_idx % 3) - 1)),
                    "adjustment_type": "PRECLOSE_CHAIN_V1",
                    "available_at": available,
                    "available_at_source": "DERIVED_POLICY",
                    "has_quote": True,
                    "quote_missing_reason": None,
                    "revision_id": "r1",
                    "source_row_hash": f"{date_idx:032x}{code_idx:032x}",
                }
            )
            status_rows.append(
                {
                    "date": date,
                    "code": code,
                    "can_buy_open": True,
                    "can_sell_open": True,
                    "is_suspended": False,
                    "is_st": False,
                    "is_xr": False,
                    "is_wd": False,
                    "limit_up_price": base * 1.1,
                    "limit_down_price": base * 0.9,
                    "buy_block_reason": None,
                    "sell_block_reason": None,
                    "available_at": available,
                    "available_at_source": "DERIVED_POLICY",
                }
            )
    industries = pd.DataFrame(
        [
            {
                "code": code,
                "industry_system": "SW",
                "level": 1,
                "industry_code": "SW_A" if idx < symbols // 2 else "SW_B",
                "industry_name": "行业A" if idx < symbols // 2 else "行业B",
                "valid_from": calendar_dates[0],
                "valid_to": pd.NaT,
                "effective_at": calendar_dates[0].tz_localize("Asia/Shanghai"),
                "known_at": pd.Timestamp("2019-12-01", tz="Asia/Shanghai"),
                "known_at_source": "NATIVE",
            }
            for idx, code in enumerate(codes)
        ]
    )
    tables: dict[str, pd.DataFrame | dict] = {
        "raw_request_manifest": {
            "requests": [
                {
                    "request_id": "synthetic-test",
                    "method": "synthetic_fixture",
                    "parameters": {},
                    "requested_at": "2026-08-25T00:00:00Z",
                    "completed_at": "2026-08-25T00:00:01Z",
                    "response_path": raw_response.name,
                    "response_sha256": _hash(raw_response),
                    "row_count": dates * symbols,
                    "success": True,
                }
            ]
        },
        "daily_bars": pd.DataFrame(bar_rows),
        "universe_membership": pd.DataFrame(member_rows),
        "industry_membership": industries,
        "trading_status": pd.DataFrame(status_rows),
        "trading_calendar": pd.DataFrame(
            {
                "date": calendar_dates,
                "market": "CN_A",
                "is_trading_day": True,
                "session_open": "09:30:00",
                "session_close": "15:00:00",
                "is_complete_session": True,
                "available_at": [
                    date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21)
                    for date in calendar_dates
                ],
            }
        ),
        "security_master": pd.DataFrame(
            {
                "code": codes,
                "name": [f"测试{idx}" for idx in range(symbols)],
                "exchange": "SZ",
                "list_date": pd.Timestamp("2010-01-01"),
                "delist_date": pd.NaT,
                "security_type": "STOCK",
                "board": "MAIN",
                "valid_from": pd.Timestamp("2010-01-01"),
                "valid_to": pd.NaT,
                "known_at": pd.Timestamp("2010-01-01", tz="Asia/Shanghai"),
            }
        ),
        "corporate_actions": pd.DataFrame(
            {
                name: pd.Series(dtype="string")
                for name in (
                    "action_id", "code", "action_type", "announcement_at",
                    "record_date", "ex_date", "payment_date", "effective_at",
                    "cash_dividend", "stock_dividend_ratio", "rights_ratio",
                    "rights_price", "known_at", "known_at_source", "raw_payload_hash",
                )
            }
        ),
    }
    files: dict[str, dict] = {}
    for name, value in tables.items():
        if isinstance(value, dict):
            path = root / f"{name}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            row_count = None
        else:
            path = root / f"{name}.parquet"
            value.to_parquet(path, index=False)
            row_count = len(value)
        entry = {"path": path.name, "sha256": _hash(path)}
        if row_count is not None:
            entry["row_count"] = row_count
            entry["rows"] = row_count
        files[name] = entry
    manifest = {
        "manifest_version": SNAPSHOT_MANIFEST_VERSION,
        "snapshot_id": "synthetic-stage3-v1",
        "data_version": "synthetic-v1",
        "built_at": "2026-08-25T00:01:00Z",
        "sources": [{"name": "synthetic", "version": "1"}],
        "frozen": True,
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root


def _test_config() -> Stage3Config:
    return Stage3Config(
        n_folds=3,
        purge_bars=10,
        embargo_bars=2,
        holdout_min_dates=30,
        holdout_min_years=0,
        top_n=5,
        target_weight=0.10,
        liquidity_median_amount_20d=1.0,
    )


def test_snapshot_manifest_detects_tampering(tmp_path: Path) -> None:
    root = _make_snapshot(tmp_path / "snapshot", dates=70, symbols=3)
    _, report = SnapshotManifestValidator(root).validate()
    assert report.passed
    with (root / "raw_request_manifest.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    _, report = SnapshotManifestValidator(root).validate()
    assert not report.passed
    assert any(issue.code == "FILE_HASH_MISMATCH" for issue in report.issues)


def test_runner_fails_closed_when_snapshot_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "result"
    report = Stage3ResearchRunner(_test_config()).run(
        tmp_path / "missing",
        output,
        expected_members=3,
        required_start=None,
    )
    assert report["status"] == "DATA_GATE_FAILED"
    gate = json.loads((output / "data_gate_report.json").read_text(encoding="utf-8"))
    assert gate["issues"][0]["code"] == "SNAPSHOT_MANIFEST_INVALID"


def test_panel_manager_preserves_members_and_historical_industry(tmp_path: Path) -> None:
    root = _make_snapshot(tmp_path / "snapshot", dates=70, symbols=3)
    manager = HS300PanelDataManager(
        root, expected_members=3, required_start=None
    ).load()
    assert manager.report is not None and manager.report.passed
    assert manager.panel is not None
    assert manager.panel.groupby("date")["code"].nunique().eq(3).all()
    assert len(manager.panel) == 70 * 3
    assert set(manager.panel["industry_code"]) == {"SW_A", "SW_B"}


def test_label_registry_uses_t1_open_to_t_h1_open() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    panel = pd.DataFrame(
        {"date": dates, "code": "000001.SZ", "industry_code": "SW_A"}
    )
    bars = pd.DataFrame(
        {
            "date": dates,
            "code": "000001.SZ",
            "open_tr": np.arange(10.0, 20.0),
            "has_quote": True,
        }
    )
    labels = LabelRegistry().build(panel, bars, dates)
    first = labels[
        (labels["signal_date"] == dates[0]) & (labels["label_type"] == "ABSOLUTE")
    ].set_index("horizon")
    assert first.loc[1, "value"] == pytest.approx(np.log(12.0 / 11.0))
    assert first.loc[3, "value"] == pytest.approx(np.log(14.0 / 11.0))
    assert first.loc[5, "value"] == pytest.approx(np.log(16.0 / 11.0))
    assert first.loc[5, "entry_date"] == dates[1]
    assert first.loc[5, "exit_date"] == dates[6]


def _signal_frames(days: int = 60, symbols: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=days)
    panel_rows = []
    bars_rows = []
    for t, date in enumerate(dates):
        for idx in range(symbols):
            code = f"{idx:06d}.SZ"
            close = 10 + idx + 0.02 * t + 0.002 * idx * t
            panel_rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry_code": "A" if idx < symbols // 2 else "B",
                    "has_quote": True,
                    "can_buy_open": True,
                    "can_sell_open": True,
                    "weight_pct": 100.0 / symbols + 0.01 * t * (idx + 1),
                    "entry_effective_date": dates[0],
                }
            )
            bars_rows.append(
                {
                    "date": date,
                    "code": code,
                    "close_tr": close,
                    "amount": 30_000_000.0,
                    "has_quote": True,
                }
            )
    return pd.DataFrame(panel_rows), pd.DataFrame(bars_rows)


def test_industry_loo_allows_singleton_industry() -> None:
    panel, bars = _signal_frames(days=25, symbols=4)
    panel.loc[panel["code"] == "000003.SZ", "industry_code"] = "SOLO"
    out = SimpleSignalBuilder().build(panel, bars)
    solo = out[out["code"] == "000003.SZ"]
    assert solo["industry_return_1d_loo"].isna().all()
    assert solo["industry_member_count_loo"].eq(0).all()
    shared = out[out["industry_code"] == "A"]
    assert shared["industry_return_1d_loo"].notna().any()


def test_candidate_feature_registry_is_complete_and_unique() -> None:
    assert len(CANDIDATE_FEATURE_SPECS) == 8
    assert len(CANDIDATE_FEATURE_COLUMNS) == 8
    assert len(set(CANDIDATE_FEATURE_COLUMNS)) == 8
    assert {spec.name for spec in CANDIDATE_FEATURE_SPECS} == {
        "MARKET_RESIDUAL_MOMENTUM_20",
        "INDUSTRY_LOO_RESIDUAL_MOMENTUM_20",
        "STOCK_RESIDUAL_VOLATILITY_20",
        "INDUSTRY_BREADTH_LOO",
        "INDUSTRY_DISPERSION_1D_LOO",
        "TRADABILITY_CROWDING_LOO",
        "INDEX_WEIGHT_CHANGE_PCT",
        "MEMBERSHIP_AGE_DAYS",
    }


def test_candidate_feature_expressions_match_hand_calculation() -> None:
    dates = pd.bdate_range("2024-01-02", periods=25)
    slopes = {
        "000001.SZ": 0.01,
        "000002.SZ": 0.02,
        "000003.SZ": -0.01,
        "000004.SZ": 0.005,
    }
    panel_rows = []
    bars_rows = []
    for t, date in enumerate(dates):
        for idx, (code, slope) in enumerate(slopes.items()):
            panel_rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry_code": "A" if code != "000004.SZ" else "B",
                    "has_quote": True,
                    "can_buy_open": not (date == dates[-1] and code == "000003.SZ"),
                    "can_sell_open": True,
                    "weight_pct": 20.0 + idx + 0.1 * t,
                    "entry_effective_date": dates[0],
                }
            )
            bars_rows.append(
                {
                    "date": date,
                    "code": code,
                    "close_tr": 100.0 * np.exp(slope * t),
                    "amount": 30_000_000.0,
                    "has_quote": True,
                }
            )

    out = SimpleSignalBuilder().build(pd.DataFrame(panel_rows), pd.DataFrame(bars_rows))
    final = out[out["date"] == dates[-1]].set_index("code")
    target = final.loc["000001.SZ"]

    # MOM20 values are 20 * slope.  The market LOO peers are 0.4, -0.2, 0.1;
    # the industry LOO peers are 0.4 and -0.2.
    assert target["MOM_20"] == pytest.approx(0.2, abs=1e-12)
    assert target["market_momentum_20_loo"] == pytest.approx(0.1, abs=1e-12)
    assert target["market_residual_momentum_20"] == pytest.approx(0.1, abs=1e-12)
    assert target["industry_momentum_20_loo"] == pytest.approx(0.1, abs=1e-12)
    assert target["industry_loo_residual_momentum_20"] == pytest.approx(0.1, abs=1e-12)

    # The two industry peers have returns 0.02 and -0.01.  Their positive
    # fraction is 1/2 and population standard deviation is 0.015.
    assert target["industry_breadth_loo"] == pytest.approx(0.5, abs=1e-12)
    assert target["industry_dispersion_1d_loo"] == pytest.approx(0.015, abs=1e-12)
    assert target["tradability_crowding_loo"] == pytest.approx(0.5, abs=1e-12)

    # Constant exponential slopes give a constant industry residual and zero
    # residual volatility once 20 observations are available.
    assert target["stock_residual_volatility_20"] == pytest.approx(0.0, abs=1e-12)
    assert target["index_weight_change_pct"] == pytest.approx(0.1, abs=1e-12)
    assert target["membership_age_days"] == float((dates[-1] - dates[0]).days)
    assert target["candidate_feature_version"] == CANDIDATE_FEATURE_VERSION


def test_loo_excludes_missing_self_without_dropping_a_valid_peer() -> None:
    panel, bars = _signal_frames(days=25, symbols=4)
    final_date = panel["date"].max()
    target_code = "000000.SZ"
    bars.loc[
        (bars["date"] == final_date) & (bars["code"] == target_code),
        ["close_tr", "has_quote"],
    ] = [np.nan, False]
    panel.loc[
        (panel["date"] == final_date) & (panel["code"] == target_code),
        "has_quote",
    ] = False
    out = SimpleSignalBuilder().build(panel, bars)
    target = out[(out["date"] == final_date) & (out["code"] == target_code)].iloc[0]
    # Industry A has one valid peer after the target's return becomes missing.
    assert target["industry_member_count_loo"] == 1
    assert np.isfinite(target["industry_return_1d_loo"])


def test_label_registry_allows_singleton_industry() -> None:
    dates = pd.bdate_range("2024-01-02", periods=12)
    codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    panel_rows = []
    bars_rows = []
    for t, date in enumerate(dates):
        for idx, code in enumerate(codes):
            panel_rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry_code": "SOLO" if code == "000004.SZ" else "A",
                }
            )
            bars_rows.append(
                {
                    "date": date,
                    "code": code,
                    "open_tr": 10.0 + idx + 0.1 * t,
                    "has_quote": True,
                }
            )
    labels = LabelRegistry().build(
        pd.DataFrame(panel_rows),
        pd.DataFrame(bars_rows),
        dates,
    )
    solo = labels[
        (labels["code"] == "000004.SZ") & (labels["label_type"] == "INDUSTRY_EXCESS")
    ]
    assert solo["value"].isna().all()
    shared = labels[
        (labels["code"] == "000001.SZ") & (labels["label_type"] == "INDUSTRY_EXCESS")
    ]
    assert shared["value"].notna().any()


def test_simple_signals_are_prefix_invariant_and_future_sentinel_safe() -> None:
    panel, bars = _signal_frames()
    cutoff = panel["date"].drop_duplicates().iloc[44]
    prefix_panel = panel[panel["date"] <= cutoff]
    prefix_bars = bars[bars["date"] <= cutoff]
    prefix = SimpleSignalBuilder().build(prefix_panel, prefix_bars)
    mutated = bars.copy()
    mutated.loc[mutated["date"] > cutoff, "close_tr"] *= 1_000_000
    mutated_panel = panel.copy()
    future = mutated_panel["date"] > cutoff
    mutated_panel.loc[future, "weight_pct"] *= 100
    mutated_panel.loc[future, ["can_buy_open", "can_sell_open"]] = False
    full = SimpleSignalBuilder().build(mutated_panel, mutated)
    columns = [
        "MOM_20", "REV_5", "LOW_VOL_20", "simple_ensemble",
        "market_return_1d", "industry_return_1d_loo", "stock_residual_1d",
        *CANDIDATE_FEATURE_COLUMNS,
    ]
    actual = full[full["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix[columns], actual[columns], check_exact=True)


def test_industry_context_is_leave_one_out() -> None:
    panel, bars = _signal_frames(days=40, symbols=12)
    baseline = SimpleSignalBuilder().build(panel, bars)
    target_code = "000000.SZ"
    final_date = panel["date"].max()
    changed = bars.copy()
    changed.loc[
        (changed["code"] == target_code) & (changed["date"] == final_date), "close_tr"
    ] *= 5
    altered = SimpleSignalBuilder().build(panel, changed)
    left = baseline.loc[
        (baseline["code"] == target_code) & (baseline["date"] == final_date),
        "industry_return_1d_loo",
    ].iloc[0]
    right = altered.loc[
        (altered["code"] == target_code) & (altered["date"] == final_date),
        "industry_return_1d_loo",
    ].iloc[0]
    assert left == pytest.approx(right, abs=1e-15)


def test_date_split_reserves_requested_purge_and_hidden_tail() -> None:
    dates = pd.bdate_range("2020-01-02", periods=200)
    plan = DateSplitProtocol(_test_config()).build(dates)
    assert plan.holdout_date_count == 30
    assert max(plan.development_dates) < plan.holdout_start
    assert len(plan.folds) == 3
    for fold in plan.folds:
        assert len(fold.purge_dates) == 10
        assert set(fold.train_dates).isdisjoint(fold.validation_dates)
        assert max(fold.train_dates) < min(fold.validation_dates)
    forbidden = pd.DataFrame({"date": [plan.holdout_start]})
    with pytest.raises(PermissionError):
        plan.assert_panel_dates(forbidden)


def test_directional_cost_and_blocked_buy_are_audited() -> None:
    dates = pd.bdate_range("2024-01-02", periods=35)
    codes = ["000001.SZ", "000002.SZ"]
    features = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "valid_signal": True,
                "median_amount_20": 100.0,
                "simple_ensemble": 1.0 if code == codes[0] else 0.0,
            }
            for date in dates
            for code in codes
        ]
    )
    bars = pd.DataFrame(
        [
            {"date": date, "code": code, "open_tr": 10.0, "has_quote": True}
            for date in dates
            for code in codes
        ]
    )
    status = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "can_buy_open": not (date == dates[11] and code == codes[0]),
                "can_sell_open": True,
            }
            for date in dates
            for code in codes
        ]
    )
    cfg = Stage3Config(
        top_n=1,
        target_weight=0.5,
        liquidity_median_amount_20d=0,
        holdout_min_dates=1,
        holdout_min_years=0,
    )
    metrics, daily = ReferenceBacktester(cfg).run(
        features,
        bars,
        status,
        dates,
        dates[10:25],
        horizon=1,
        phase=0,
    )
    assert metrics.blocked_buys == 1
    assert metrics.transaction_cost > 0
    assert metrics.net_total_return < 0
    assert daily["transaction_cost"].sum() == pytest.approx(metrics.transaction_cost)


def test_exited_constituent_sells_on_next_open() -> None:
    dates = pd.bdate_range("2024-01-02", periods=12)
    held = "000001.SZ"
    other = "000002.SZ"
    member_rows = []
    feature_rows = []
    bar_rows = []
    status_rows = []
    for date in dates:
        for code in (held, other):
            is_member = not (code == held and date >= dates[5])
            if is_member:
                member_rows.append({"date": date, "code": code, "is_member": True})
                feature_rows.append(
                    {
                        "date": date,
                        "code": code,
                        "valid_signal": True,
                        "median_amount_20": 100.0,
                        "simple_ensemble": 1.0 if code == held else 0.0,
                    }
                )
            bar_rows.append(
                {"date": date, "code": code, "open_tr": 10.0, "has_quote": True}
            )
            status_rows.append(
                {
                    "date": date,
                    "code": code,
                    "can_buy_open": True,
                    "can_sell_open": True,
                }
            )
    cfg = Stage3Config(
        top_n=1,
        target_weight=0.5,
        liquidity_median_amount_20d=0,
        holdout_min_dates=1,
        holdout_min_years=0,
    )
    metrics, daily = ReferenceBacktester(cfg).run(
        pd.DataFrame(feature_rows),
        pd.DataFrame(bar_rows),
        pd.DataFrame(status_rows),
        dates,
        dates[:10],
        horizon=5,
        phase=1,
        membership=pd.DataFrame(member_rows),
    )
    assert metrics.universe_exit_sells >= 1
    assert metrics.universe_exit_pending == 0
    held_after_exit = daily[daily["date"] >= dates[5]]
    assert held_after_exit["holding_count"].max() <= 1


def test_missing_quotes_catch_up_gap_on_resume() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    code = "000001.SZ"
    opens = [10.0, 10.0, np.nan, np.nan, 12.0, 12.0, 12.0, 12.0]
    features = pd.DataFrame(
        {
            "date": dates,
            "code": code,
            "valid_signal": True,
            "median_amount_20": 100.0,
            "simple_ensemble": 1.0,
        }
    )
    bars = pd.DataFrame(
        {
            "date": dates,
            "code": code,
            "open_tr": opens,
            "has_quote": [np.isfinite(value) for value in opens],
        }
    )
    status = pd.DataFrame(
        {
            "date": dates,
            "code": code,
            "can_buy_open": True,
            "can_sell_open": True,
        }
    )
    cfg = Stage3Config(
        top_n=1,
        target_weight=1.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        liquidity_median_amount_20d=0,
        holdout_min_dates=1,
        holdout_min_years=0,
    )
    metrics, daily = ReferenceBacktester(cfg).run(
        features,
        bars,
        status,
        dates,
        dates[:6],
        horizon=1,
        phase=0,
    )
    assert metrics.missing_valuation_intervals >= 1
    assert daily["nav"].iloc[-1] == pytest.approx(1.2, rel=1e-9)


def test_stage3_runner_writes_development_only_outputs(tmp_path: Path) -> None:
    root = _make_snapshot(tmp_path / "snapshot", dates=180, symbols=12)
    output = tmp_path / "result"
    report = Stage3ResearchRunner(_test_config()).run(
        root,
        output,
        expected_members=12,
        required_start=None,
    )
    assert report["status"] == "STAGE3_MINIMUM_LOOP_COMPLETED"
    assert report["statistical_gate"] == "PENDING_STAGE4_FEASIBILITY_DECISION"
    assert (output / "development_features.parquet").is_file()
    assert (output / "development_labels.parquet").is_file()
    assert not any("holdout" in path.name.lower() for path in output.glob("*.parquet"))
    features = pd.read_parquet(output / "development_features.parquet")
    assert set(CANDIDATE_FEATURE_COLUMNS).issubset(features.columns)
    assert features["candidate_feature_version"].eq(CANDIDATE_FEATURE_VERSION).all()
    assert pd.to_datetime(features["date"]).max() < pd.Timestamp(
        report["split"]["holdout_start"]
    )
