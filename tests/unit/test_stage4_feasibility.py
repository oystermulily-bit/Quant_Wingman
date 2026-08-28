from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_stage3.protocol import Stage3Config
from research_stage4.gate import Stage4Thresholds, classify_gate, evaluate_horizon_go
from research_stage4.runner import Stage4FeasibilityRunner
from research_stage4.stats import BootstrapResult, holm_adjust, moving_block_bootstrap


def test_holm_rejects_smaller_p_first() -> None:
    adjusted = holm_adjust({"3d_vs_1d": 0.01, "5d_vs_1d": 0.04}, alpha=0.05)
    assert adjusted["3d_vs_1d"]["reject"] is True
    assert adjusted["5d_vs_1d"]["p_holm"] >= adjusted["5d_vs_1d"]["p_value"]
    assert adjusted["5d_vs_1d"]["reject"] is True
    weak = holm_adjust({"3d_vs_1d": 0.01, "5d_vs_1d": 0.06}, alpha=0.05)
    assert weak["3d_vs_1d"]["reject"] is True
    assert weak["5d_vs_1d"]["reject"] is False


def test_moving_block_bootstrap_is_seed_deterministic() -> None:
    rng = np.random.default_rng(1)
    series = rng.normal(0.002, 0.01, size=400)

    def mean_stat(values: np.ndarray) -> float:
        return float(np.mean(values))

    first = moving_block_bootstrap(
        series, statistic=mean_stat, block_days=20, samples=80, seed=20260812
    )
    second = moving_block_bootstrap(
        series, statistic=mean_stat, block_days=20, samples=80, seed=20260812
    )
    assert first.ci_low == second.ci_low
    assert first.p_value == second.p_value
    assert first.observed == pytest.approx(float(series.mean()))


def test_go_fails_when_sharpe_gain_is_below_frozen_threshold() -> None:
    boot = BootstrapResult(
        observed=0.05,
        ci_low=0.01,
        ci_high=0.12,
        p_value=0.001,
        samples=2000,
        seed=20260812,
        block_days=20,
    )
    result = evaluate_horizon_go(
        horizon=3,
        candidate={
            "median_sharpe": 0.40,
            "median_annualized_return": 0.10,
            "median_max_drawdown": 0.12,
            "worst_phase_sharpe": 0.35,
            "phases_agreeing": 3,
            "folds_agreeing": 5,
            "signals_agreeing": 3,
            "concentration_ok": True,
        },
        baseline={
            "median_sharpe": 0.30,
            "median_annualized_return": 0.07,
            "median_max_drawdown": 0.15,
        },
        sharpe_bootstrap=boot,
        holm={"reject": True, "p_holm": 0.01},
    )
    assert result["passed"] is False
    assert result["checks"]["sharpe_improvement"]["passed"] is False
    assert result["thresholds"]["min_sharpe_improvement"] == 0.15


def test_classify_gate_keeps_1d_when_both_fail() -> None:
    gate = classify_gate({3: {"passed": False}, 5: {"passed": False}})
    assert gate["statistical_gate"] == "KEEP_1D_BASELINE"
    assert gate["allowed_horizons"] == [1]
    assert gate["rd_agent_allowed"] is False
    assert gate["holdout_read"] is False


def _write_stage3_artifacts(root: Path, *, holdout_leak: bool = False, strong_3d: bool = True) -> None:
    rng = np.random.default_rng(20260812)
    dates = pd.bdate_range("2018-01-02", periods=520)
    holdout_start = pd.Timestamp("2024-08-26")
    rows = []
    for horizon, mean in ((1, 0.00015), (3, 0.0020 if strong_3d else 0.00016), (5, 0.00010)):
        for phase in range(horizon):
            noise = rng.normal(0.0, 0.008, size=len(dates))
            for date_idx, (date, shock) in enumerate(zip(dates, noise)):
                rows.append(
                    {
                        "date": date,
                        "fold_id": min(date_idx // 104, 4),
                        "horizon": horizon,
                        "phase": phase,
                        "net_return": mean + shock,
                        "gross_return": mean + shock + 0.00005,
                    }
                )
    if holdout_leak:
        rows.append(
            {
                "date": holdout_start,
                "fold_id": 4,
                "horizon": 1,
                "phase": 0,
                "net_return": 0.0,
                "gross_return": 0.0,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "development_oof_portfolio_daily.parquet", index=False)

    codes = [f"{idx:06d}.SZ" for idx in range(1, 16)]
    feature_rows = []
    label_rows = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(codes):
            mom = 0.1 * code_idx + 0.001 * date_idx
            feature_rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry_code": "SW_A" if code_idx < 8 else "SW_B",
                    "MOM_20": mom,
                    "REV_5": -0.02 * code_idx,
                    "LOW_VOL_20": 0.01 * (15 - code_idx),
                    "simple_ensemble": mom,
                    "valid_signal": True,
                }
            )
            for horizon in (1, 3, 5):
                label_rows.append(
                    {
                        "signal_date": date,
                        "code": code,
                        "horizon": horizon,
                        "label_type": "ABSOLUTE",
                        "value": mom * (1.4 if horizon == 3 else 1.0 if horizon == 1 else 0.7),
                    }
                )
    pd.DataFrame(feature_rows).to_parquet(root / "development_features.parquet", index=False)
    pd.DataFrame(label_rows).to_parquet(root / "development_labels.parquet", index=False)

    def horizon_block(annual: float, rank_ic: float) -> dict:
        return {
            "median_annualized_return": annual,
            "median_net_sharpe": annual * 2,
            "rank_ic": {"rank_ic": rank_ic, "dates": 40},
            "phases": [
                {
                    "sharpe": annual * 2,
                    "annualized_return": annual,
                    "gross_total_return": annual,
                    "net_total_return": annual * 0.9,
                    "cost_sensitivity": {
                        "0.0": {
                            "sharpe": annual * 2.1,
                            "annualized_return": annual * 1.05,
                            "gross_total_return": annual,
                            "net_total_return": annual,
                        },
                        "1.0": {
                            "sharpe": annual * 2,
                            "annualized_return": annual,
                            "gross_total_return": annual,
                            "net_total_return": annual * 0.9,
                        },
                        "1.5": {
                            "sharpe": annual * 1.8,
                            "annualized_return": annual * 0.9,
                            "gross_total_return": annual,
                            "net_total_return": annual * 0.8,
                        },
                    },
                }
            ],
        }

    folds = []
    for fold_id in range(5):
        validation_start = dates[fold_id * 104]
        validation_end = dates[-1] if fold_id == 4 else dates[(fold_id + 1) * 104 - 1]
        folds.append(
            {
                "fold_id": fold_id,
                "validation_start": validation_start.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "horizons": {
                    "1": horizon_block(0.04, 0.02),
                    "3": horizon_block(0.09 if strong_3d else 0.041, 0.08 if strong_3d else 0.021),
                    "5": horizon_block(0.03, 0.01),
                },
            }
        )
    (root / "stage3_report.json").write_text(
        json.dumps(
            {
                "protocol": "w1ngman_stage3_minimum_loop_v1",
                "status": "STAGE3_MINIMUM_LOOP_COMPLETED",
                "snapshot_id": "synthetic",
                "split": {"holdout_start": "2024-08-26", "holdout_date_count": 483},
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )


def test_stage4_fails_closed_on_holdout_leak(tmp_path: Path) -> None:
    stage3 = tmp_path / "stage3"
    stage3.mkdir()
    _write_stage3_artifacts(stage3, holdout_leak=True)
    report = Stage4FeasibilityRunner(
        Stage3Config(bootstrap_samples=20),
        Stage4Thresholds(),
    ).run(
        snapshot_dir=tmp_path / "unused",
        stage3_dir=stage3,
        output_dir=tmp_path / "stage4",
        bootstrap_samples=20,
    )
    assert report["status"] == "RESEARCH_GATE_FAILED"
    assert report["system_status"] == "MODEL_NOT_VALIDATED"


def test_stage4_keep_1d_when_horizons_do_not_clear_go(tmp_path: Path) -> None:
    stage3 = tmp_path / "stage3"
    stage3.mkdir()
    _write_stage3_artifacts(stage3, strong_3d=False)
    report = Stage4FeasibilityRunner(
        Stage3Config(bootstrap_samples=40),
        Stage4Thresholds(),
    ).run(
        snapshot_dir=tmp_path / "unused",
        stage3_dir=stage3,
        output_dir=tmp_path / "stage4",
        bootstrap_samples=40,
    )
    assert report["statistical_gate"] == "KEEP_1D_BASELINE"
    assert report["allowed_horizons"] == [1]
    assert report["rd_agent_allowed"] is False
    assert report["go"]["3"]["checks"]["sharpe_improvement"]["threshold"] == 0.15
    assert Path(tmp_path / "stage4" / "stage4_report.json").is_file()


def test_stage4_robustness_excludes_development_non_oof_rows(tmp_path: Path) -> None:
    stage3 = tmp_path / "stage3"
    stage3.mkdir()
    _write_stage3_artifacts(stage3, strong_3d=False)
    runner = Stage4FeasibilityRunner(Stage3Config(bootstrap_samples=20))
    baseline = runner.run(
        snapshot_dir=tmp_path / "unused",
        stage3_dir=stage3,
        output_dir=tmp_path / "stage4_a",
        bootstrap_samples=20,
    )

    features = pd.read_parquet(stage3 / "development_features.parquet")
    labels = pd.read_parquet(stage3 / "development_labels.parquet")
    extra_date = pd.Timestamp("2017-12-29")
    extra_features = features.iloc[:15].copy()
    extra_features["date"] = extra_date
    extra_features[list(("MOM_20", "REV_5", "LOW_VOL_20", "simple_ensemble"))] = 1e12
    extra_labels = labels.iloc[:45].copy()
    extra_labels["signal_date"] = extra_date
    extra_labels["value"] = -1e12
    pd.concat([features, extra_features], ignore_index=True).to_parquet(
        stage3 / "development_features.parquet", index=False
    )
    pd.concat([labels, extra_labels], ignore_index=True).to_parquet(
        stage3 / "development_labels.parquet", index=False
    )

    repeated = runner.run(
        snapshot_dir=tmp_path / "unused",
        stage3_dir=stage3,
        output_dir=tmp_path / "stage4_b",
        bootstrap_samples=20,
    )

    assert repeated["oof_scope"]["oof_only"] is True
    assert repeated["oof_scope"]["feature_rows_excluded_non_oof"] == 15
    assert repeated["oof_scope"]["label_rows_excluded_non_oof"] == 45
    assert repeated["horizons"]["3"]["signal_rank_ic_deltas"] == baseline["horizons"]["3"][
        "signal_rank_ic_deltas"
    ]
    assert repeated["horizons"]["3"]["concentration_detail"] == baseline["horizons"]["3"][
        "concentration_detail"
    ]


def test_stage4_fails_closed_when_oof_fold_provenance_is_invalid(tmp_path: Path) -> None:
    stage3 = tmp_path / "stage3"
    stage3.mkdir()
    _write_stage3_artifacts(stage3)
    daily_path = stage3 / "development_oof_portfolio_daily.parquet"
    daily = pd.read_parquet(daily_path)
    daily.loc[daily.index[0], "fold_id"] = 99
    daily.to_parquet(daily_path, index=False)

    report = Stage4FeasibilityRunner(Stage3Config(bootstrap_samples=20)).run(
        snapshot_dir=tmp_path / "unused",
        stage3_dir=stage3,
        output_dir=tmp_path / "stage4",
        bootstrap_samples=20,
    )

    assert report["status"] == "RESEARCH_GATE_FAILED"
    assert "OOF provenance validation failed" in report["reason"]
