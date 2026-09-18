"""Stage 3R: D21-v2 Development OOF. Does not overwrite D21-v1 artifacts."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.hs300_panel import HS300PanelDataManager
from research_stage3.backtest import ReferenceBacktester
from research_stage3.labels import LabelRegistry
from research_stage3.protocol import DateSplitProtocol, SplitPlan, Stage3Config
from research_stage3.runner import (
    _assert_frozen_holdout,
    _research_trading_dates,
)
from research_stage3.signals import SimpleSignalBuilder

from .backtest import BufferedTop20Backtester
from .protocol import (
    EXPERIMENTS,
    STAGE3R_PROTOCOL_VERSION,
    D21V2Config,
)
from .signals import SlowResidualEnsembleBuilder


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank_ic_by_date(frame: pd.DataFrame, score: str) -> dict[str, float | int | None]:
    values: list[float] = []
    for _date, group in frame.groupby("signal_date", observed=True):
        valid = group[[score, "value"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 10 or valid.nunique().min() < 2:
            continue
        corr = valid[score].rank(method="average").corr(valid["value"].rank(method="average"))
        if np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return {"dates": 0, "rank_ic": None, "rank_icir": None, "positive_rate": None}
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {
        "dates": len(values),
        "rank_ic": float(array.mean()),
        "rank_icir": float(array.mean() / std) if std > 1e-12 else None,
        "positive_rate": float((array > 0).mean()),
    }


def merge_feature_frames(simple: pd.DataFrame, slow: pd.DataFrame) -> pd.DataFrame:
    slow_cols = [
        "date",
        "code",
        "epsilon",
        "IDIO_LOW_VOL_60",
        "RES_MOM_120_20",
        "RES_TREND_EFF_60_5",
        "IDIO_LOW_VOL_60_rank",
        "RES_MOM_120_20_rank",
        "RES_TREND_EFF_60_5_rank",
        "slow_residual_ensemble",
        "valid_slow_signal",
    ]
    return simple.merge(slow[slow_cols], on=["date", "code"], how="left", validate="one_to_one")


class Stage3RResearchRunner:
    """Independent D21-v2 loop. Holdout stays sealed. RD-Agent stays off."""

    experiments = EXPERIMENTS
    completion_status = "STAGE3R_D21_V2_COMPLETED"

    def build_slow(self, dev_panel, dev_bars, trading_dates):
        return SlowResidualEnsembleBuilder(self.config).build(dev_panel, dev_bars, trading_dates)

    def __init__(
        self,
        config: D21V2Config | None = None,
        split_config: Stage3Config | None = None,
    ) -> None:
        self.config = config or D21V2Config()
        # Production uses the D21-v1 split object so holdout.lock stays sealed.
        self.split_config = split_config or Stage3Config()

    def run(
        self,
        snapshot_dir: str | Path,
        output_dir: str | Path,
        *,
        expected_members: int = 300,
        required_start: str | None = "2014-01-02",
    ) -> dict[str, Any]:
        output = Path(output_dir)
        if output.name == "stage3_hs300_v2_execution_ledger":
            raise PermissionError("D21-v2 must not overwrite D21-v1 Stage 3 artifacts")
        output.mkdir(parents=True, exist_ok=True)
        manager = HS300PanelDataManager(
            snapshot_dir,
            expected_members=expected_members,
            required_start=required_start,
        ).load(raise_on_gate_failure=False)
        assert manager.report is not None
        _atomic_json(output / "data_gate_report.json", manager.report.to_dict())
        if not manager.report.passed:
            report = {
                "protocol": self.config.stage3r_protocol,
                "status": "DATA_GATE_FAILED",
                "statistical_gate": "NOT_RUN",
                "reason": "正式快照未通过数据门禁；未计算任何 D21-v2 研究结果",
                "holdout_read": False,
                "rd_agent_allowed": False,
            }
            _atomic_json(output / "stage3r_report.json", report)
            return report

        assert manager.panel is not None
        assert manager.daily_bars is not None
        assert manager.trading_status is not None
        snapshot = Path(snapshot_dir)
        # Same chronological split as D21-v1 so Holdout.lock remains the sealed object.
        split: SplitPlan = DateSplitProtocol(self.split_config).build(
            _research_trading_dates(snapshot, manager)
        )
        try:
            _assert_frozen_holdout(snapshot, split)
        except PermissionError as exc:
            report = {
                "protocol": self.config.stage3r_protocol,
                "status": "SPLIT_PROTOCOL_MISMATCH",
                "statistical_gate": "NOT_RUN",
                "reason": str(exc),
                "holdout_read": False,
            }
            _atomic_json(output / "stage3r_report.json", report)
            return report
        _atomic_json(output / "split_plan.json", split.to_dict())
        print(
            f"[stage3r] hypothesis={self.config.hypothesis_id} "
            f"development_dates={len(split.development_dates)} "
            f"holdout_start={split.holdout_start.date().isoformat()}",
            flush=True,
        )

        dev_panel = manager.development_panel(split.development_dates)
        dev_bars = manager.bars_for_dates(split.development_dates)
        dev_status = manager.trading_status[
            manager.trading_status["date"].isin(split.development_dates)
        ].copy()
        exec_bars = manager.execution_bars_for_dates(split.development_dates)
        exec_status = manager.execution_status_for_dates(split.development_dates)
        split.assert_panel_dates(dev_panel)
        split.assert_panel_dates(dev_bars)
        split.assert_panel_dates(dev_status)
        split.assert_panel_dates(exec_bars)
        split.assert_panel_dates(exec_status)

        print("[stage3r] building SIMPLE_ENSEMBLE_V1", flush=True)
        simple = SimpleSignalBuilder().build(dev_panel, dev_bars)
        print("[stage3r] building SLOW_RESIDUAL_ENSEMBLE_V1", flush=True)
        slow = self.build_slow(
            dev_panel,
            dev_bars,
            pd.DatetimeIndex(split.development_dates),
        )
        features = merge_feature_frames(simple, slow)
        print("[stage3r] building H=5 labels", flush=True)
        labels = LabelRegistry(
            horizons=(self.config.target_horizon,),
            execution_lag_bars=self.config.execution_lag_bars,
        ).build(dev_panel, dev_bars, pd.DatetimeIndex(split.development_dates), dev_status)
        features.to_parquet(output / "development_features.parquet", index=False)
        simple.to_parquet(output / "development_features_simple.parquet", index=False)
        slow.to_parquet(output / "development_features_slow.parquet", index=False)
        labels.to_parquet(output / "development_labels.parquet", index=False)

        absolute = labels[labels["label_type"] == "ABSOLUTE"][
            ["signal_date", "code", "horizon", "value"]
        ]
        joined = features.merge(
            absolute,
            left_on=["date", "code"],
            right_on=["signal_date", "code"],
            how="inner",
        )

        naive = ReferenceBacktester(self.config.as_stage3_config())
        buffered = BufferedTop20Backtester(self.config)
        membership = dev_panel[["date", "code"]]
        fold_reports: list[dict[str, Any]] = []
        daily_outputs: list[pd.DataFrame] = []
        horizon = self.config.target_horizon

        for fold in split.folds:
            validation_dates = pd.DatetimeIndex(fold.validation_dates)
            print(
                f"[stage3r] fold={fold.fold_id} "
                f"{validation_dates[0].date().isoformat()}->"
                f"{validation_dates[-1].date().isoformat()}",
                flush=True,
            )
            fold_payload: dict[str, Any] = {
                "fold_id": fold.fold_id,
                "validation_start": validation_dates[0].date().isoformat(),
                "validation_end": validation_dates[-1].date().isoformat(),
                "experiments": {},
            }
            for spec in self.experiments:
                score = spec["score_column"]
                valid_column = spec["valid_column"]
                ic_frame = joined[
                    (joined["horizon"] == horizon)
                    & joined["signal_date"].isin(validation_dates)
                ]
                phase_metrics: list[dict[str, Any]] = []
                print(
                    f"[stage3r] fold={fold.fold_id} experiment={spec['experiment_id']} "
                    f"{spec['signal_version']}+{spec['portfolio_version']}",
                    flush=True,
                )
                for phase in range(horizon):
                    metrics, daily = self._run_book(
                        naive=naive,
                        buffered=buffered,
                        spec=spec,
                        features=features,
                        exec_bars=exec_bars,
                        exec_status=exec_status,
                        trading_dates=split.development_dates,
                        signal_dates=validation_dates,
                        membership=membership,
                        phase=phase,
                        cost_multiplier=1.0,
                    )
                    metric_payload = metrics.to_dict()
                    if "exit_pending" in daily.attrs:
                        metric_payload["exit_pending"] = int(daily.attrs["exit_pending"])
                    sensitivity: dict[str, dict[str, Any]] = {"1.0": dict(metric_payload)}
                    for multiplier in self.config.cost_multipliers:
                        if float(multiplier) == 1.0:
                            continue
                        stressed, _ = self._run_book(
                            naive=naive,
                            buffered=buffered,
                            spec=spec,
                            features=features,
                            exec_bars=exec_bars,
                            exec_status=exec_status,
                            trading_dates=split.development_dates,
                            signal_dates=validation_dates,
                            membership=membership,
                            phase=phase,
                            cost_multiplier=float(multiplier),
                        )
                        sensitivity[f"{float(multiplier):.1f}"] = stressed.to_dict()
                    metric_payload["cost_sensitivity"] = sensitivity
                    phase_metrics.append(metric_payload)
                    daily = daily.copy()
                    daily["fold_id"] = fold.fold_id
                    daily["horizon"] = horizon
                    daily["phase"] = phase
                    daily["experiment"] = spec["experiment_id"]
                    daily["signal_version"] = spec["signal_version"]
                    daily["portfolio_version"] = spec["portfolio_version"]
                    daily_outputs.append(daily)
                fold_payload["experiments"][spec["experiment_id"]] = {
                    "signal_version": spec["signal_version"],
                    "portfolio_version": spec["portfolio_version"],
                    "rank_ic": _rank_ic_by_date(ic_frame, score),
                    "phases": phase_metrics,
                    "median_net_sharpe": float(
                        np.median([row["sharpe"] for row in phase_metrics])
                    ),
                    "median_annualized_return": float(
                        np.median([row["annualized_return"] for row in phase_metrics])
                    ),
                }
            fold_reports.append(fold_payload)

        if daily_outputs:
            pd.concat(daily_outputs, ignore_index=True).to_parquet(
                output / "development_oof_portfolio_daily.parquet", index=False
            )
        oof_dates = pd.DatetimeIndex(split.validation_dates)
        oof_features = features[features["date"].isin(oof_dates)]
        coverage = (
            float(oof_features["valid_slow_signal"].fillna(False).mean())
            if len(oof_features)
            else 0.0
        )
        report = {
            "protocol": self.config.stage3r_protocol,
            "status": self.completion_status,
            "hypothesis_id": self.config.hypothesis_id,
            "development_adaptive": self.config.development_adaptive,
            "data_gate": manager.report.status,
            "statistical_gate": "PENDING_STAGE4R_FEASIBILITY_DECISION",
            "snapshot_id": manager.report.snapshot_id,
            "config": asdict(self.config),
            "config_hash": self.config.fingerprint(),
            "d21_v1_split_hash": split.config_hash,
            "signal_version": self.config.signal_version,
            "portfolio_version": self.config.portfolio_version,
            "target_horizon": self.config.target_horizon,
            "split": split.to_dict(),
            "development_feature_rows": len(features),
            "development_label_rows": len(labels),
            "oof_slow_coverage": coverage,
            "experiments": [dict(spec) for spec in self.experiments],
            "folds": fold_reports,
            "restrictions": [
                "D21-v1 KEEP_1D_BASELINE 证据未覆盖",
                "RD-Agent未启动",
                "Holdout未读取、未生成标签、未参与报告",
                "仅研究 H=5；未同时搜索 H=3",
                "development_adaptive=true",
                "未将本轮 Development 解释为生产模型有效",
            ],
            "holdout_read": False,
            "rd_agent_allowed": False,
            "system_status": "MODEL_NOT_VALIDATED",
        }
        _atomic_json(output / "stage3r_report.json", report)
        return report

    def _run_book(
        self,
        *,
        naive: ReferenceBacktester,
        buffered: BufferedTop20Backtester,
        spec: dict[str, str],
        features: pd.DataFrame,
        exec_bars: pd.DataFrame,
        exec_status: pd.DataFrame,
        trading_dates: tuple,
        signal_dates: pd.DatetimeIndex,
        membership: pd.DataFrame,
        phase: int,
        cost_multiplier: float,
    ):
        score = spec["score_column"]
        valid_column = spec["valid_column"]
        book_features = features
        if valid_column != "valid_signal":
            book_features = features.copy()
            book_features["valid_signal"] = book_features[valid_column]
        if spec["execution"] == "naive":
            return naive.run(
                book_features,
                exec_bars,
                exec_status,
                trading_dates,
                signal_dates,
                horizon=self.config.target_horizon,
                phase=phase,
                cost_multiplier=cost_multiplier,
                score_column=score,
                membership=membership,
            )
        return buffered.run(
            features,
            exec_bars,
            exec_status,
            trading_dates,
            signal_dates,
            horizon=self.config.target_horizon,
            phase=phase,
            cost_multiplier=cost_multiplier,
            score_column=score,
            valid_column=valid_column,
            membership=membership,
        )
