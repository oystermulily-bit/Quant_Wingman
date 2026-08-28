"""Orchestrate the frozen stage-3 minimum research loop."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.hs300_panel import HS300PanelDataManager

from .backtest import ReferenceBacktester
from .labels import LabelRegistry
from .protocol import DateSplitProtocol, SplitPlan, Stage3Config
from .signals import SIGNAL_VERSION, SimpleSignalBuilder


STAGE3_PROTOCOL_VERSION = "w1ngman_stage3_minimum_loop_v1"


def _research_trading_dates(snapshot_dir: Path, manager: HS300PanelDataManager) -> pd.DatetimeIndex:
    """Build splits on the full research calendar, including sealed Holdout dates.

    v2 panels drop Holdout prices from standardized tables. Using only
    manager.calendar would silently carve a second Holdout out of Development.
    The calendar file still lists every research session and does not contain
    prices.
    """
    for relative in (
        Path("standardized") / "trading_calendar.parquet",
        Path("trading_calendar.parquet"),
    ):
        path = snapshot_dir / relative
        if not path.is_file():
            continue
        calendar = pd.read_parquet(path)
        if "date" not in calendar.columns:
            continue
        trading = calendar.loc[
            calendar.get("is_trading_day", True).fillna(False).astype(bool)
            & calendar.get("is_complete_session", True).fillna(False).astype(bool),
            "date",
        ]
        dates = pd.DatetimeIndex(pd.to_datetime(trading).dt.normalize().unique()).sort_values()
        if len(dates):
            return dates
    if manager.calendar is None:
        raise RuntimeError("no research trading calendar")
    return pd.DatetimeIndex(manager.calendar)


def _assert_frozen_holdout(snapshot_dir: Path, split: SplitPlan) -> None:
    lock_path = snapshot_dir / "splits" / "holdout.lock.json"
    if not lock_path.is_file():
        return
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_hash = lock.get("holdout_date_hash")
    expected_start = lock.get("holdout_start")
    if expected_hash and expected_hash != split.holdout_date_hash:
        raise PermissionError(
            "rebuilt split Holdout hash does not match frozen holdout.lock.json"
        )
    if expected_start and str(expected_start) != split.holdout_start.date().isoformat():
        raise PermissionError(
            "rebuilt split Holdout start does not match frozen holdout.lock.json"
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank_ic_by_date(frame: pd.DataFrame) -> dict[str, float | int | None]:
    values: list[float] = []
    for _date, group in frame.groupby("signal_date", observed=True):
        valid = group[["simple_ensemble", "value"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(valid) < 10 or valid.nunique().min() < 2:
            continue
        factor_rank = valid["simple_ensemble"].rank(method="average")
        label_rank = valid["value"].rank(method="average")
        corr = factor_rank.corr(label_rank)
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


class Stage3ResearchRunner:
    """Run stage 3 only; it cannot call RD-Agent or read a release holdout."""

    def __init__(self, config: Stage3Config | None = None) -> None:
        self.config = config or Stage3Config()

    def run(
        self,
        snapshot_dir: str | Path,
        output_dir: str | Path,
        *,
        expected_members: int = 300,
        required_start: str | None = "2014-01-02",
    ) -> dict[str, Any]:
        output = Path(output_dir)
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
                "protocol": STAGE3_PROTOCOL_VERSION,
                "status": "DATA_GATE_FAILED",
                "statistical_gate": "NOT_RUN",
                "portfolio_gate": "NOT_RUN",
                "product_gate": "NOT_RUN",
                "reason": "正式快照未通过数据门禁；未计算任何研究结果",
            }
            _atomic_json(output / "stage3_report.json", report)
            return report

        assert manager.panel is not None
        assert manager.daily_bars is not None
        assert manager.trading_status is not None
        assert manager.calendar is not None
        snapshot = Path(snapshot_dir)
        split: SplitPlan = DateSplitProtocol(self.config).build(
            _research_trading_dates(snapshot, manager)
        )
        try:
            _assert_frozen_holdout(snapshot, split)
        except PermissionError as exc:
            report = {
                "protocol": STAGE3_PROTOCOL_VERSION,
                "status": "SPLIT_PROTOCOL_MISMATCH",
                "statistical_gate": "NOT_RUN",
                "portfolio_gate": "NOT_RUN",
                "product_gate": "NOT_RUN",
                "reason": str(exc),
            }
            _atomic_json(output / "stage3_report.json", report)
            return report
        _atomic_json(output / "split_plan.json", split.to_dict())
        print(
            f"[stage3] data_gate={manager.report.status} "
            f"development_dates={len(split.development_dates)} "
            f"holdout_start={split.holdout_start.date().isoformat()}",
            flush=True,
        )

        dev_panel = manager.development_panel(split.development_dates)
        dev_bars = manager.bars_for_dates(split.development_dates)
        dev_status = manager.trading_status[
            manager.trading_status["date"].isin(split.development_dates)
        ].copy()
        split.assert_panel_dates(dev_panel)
        split.assert_panel_dates(dev_bars)
        split.assert_panel_dates(dev_status)

        print("[stage3] building development features", flush=True)
        features = SimpleSignalBuilder().build(dev_panel, dev_bars)
        print("[stage3] building development labels", flush=True)
        labels = LabelRegistry(
            horizons=self.config.horizons,
            execution_lag_bars=self.config.execution_lag_bars,
        ).build(dev_panel, dev_bars, pd.DatetimeIndex(split.development_dates), dev_status)
        # Persist only Development artifacts. No Holdout row is materialised here.
        print("[stage3] writing development parquet", flush=True)
        features.to_parquet(output / "development_features.parquet", index=False)
        labels.to_parquet(output / "development_labels.parquet", index=False)

        absolute = labels[labels["label_type"] == "ABSOLUTE"][
            ["signal_date", "code", "horizon", "value"]
        ]
        joined = features[["date", "code", "simple_ensemble"]].merge(
            absolute,
            left_on=["date", "code"],
            right_on=["signal_date", "code"],
            how="inner",
            validate="one_to_many",
        )

        backtester = ReferenceBacktester(self.config)
        fold_reports: list[dict[str, Any]] = []
        daily_outputs: list[pd.DataFrame] = []
        for fold in split.folds:
            validation_dates = pd.DatetimeIndex(fold.validation_dates)
            print(
                f"[stage3] fold={fold.fold_id} "
                f"{validation_dates[0].date().isoformat()}->"
                f"{validation_dates[-1].date().isoformat()}",
                flush=True,
            )
            fold_payload: dict[str, Any] = {
                "fold_id": fold.fold_id,
                "validation_start": validation_dates[0].date().isoformat(),
                "validation_end": validation_dates[-1].date().isoformat(),
                "horizons": {},
            }
            for horizon in self.config.horizons:
                ic_frame = joined[
                    (joined["horizon"] == horizon)
                    & joined["signal_date"].isin(validation_dates)
                ]
                phase_metrics: list[dict[str, Any]] = []
                for phase in range(horizon):
                    metrics, daily = backtester.run(
                        features,
                        dev_bars,
                        dev_status,
                        split.development_dates,
                        validation_dates,
                        horizon=horizon,
                        phase=phase,
                    )
                    metric_payload = metrics.to_dict()
                    sensitivity: dict[str, dict[str, Any]] = {
                        "1.0": dict(metric_payload),
                    }
                    for multiplier in (0.0, 1.5):
                        stressed, _ = backtester.run(
                            features,
                            dev_bars,
                            dev_status,
                            split.development_dates,
                            validation_dates,
                            horizon=horizon,
                            phase=phase,
                            cost_multiplier=multiplier,
                        )
                        sensitivity[f"{multiplier:.1f}"] = stressed.to_dict()
                    metric_payload["cost_sensitivity"] = sensitivity
                    phase_metrics.append(metric_payload)
                    daily["fold_id"] = fold.fold_id
                    daily["horizon"] = horizon
                    daily["phase"] = phase
                    daily_outputs.append(daily)
                fold_payload["horizons"][str(horizon)] = {
                    "rank_ic": _rank_ic_by_date(ic_frame),
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
        report = {
            "protocol": STAGE3_PROTOCOL_VERSION,
            "status": "STAGE3_MINIMUM_LOOP_COMPLETED",
            "data_gate": manager.report.status,
            "statistical_gate": "PENDING_STAGE4_FEASIBILITY_DECISION",
            "portfolio_gate": "REFERENCE_BACKTEST_ONLY",
            "product_gate": "NOT_RUN",
            "snapshot_id": manager.report.snapshot_id,
            "config": asdict(self.config),
            "config_hash": self.config.fingerprint(),
            "signal_version": SIGNAL_VERSION,
            "split": split.to_dict(),
            "development_feature_rows": len(features),
            "development_label_rows": len(labels),
            "folds": fold_reports,
            "restrictions": [
                "RD-Agent未启动",
                "Holdout未读取、未生成标签、未参与报告",
                "未生成生产建议或网页/API输出",
                "阶段4完成Bootstrap/Holm及GO门槛前不得宣布3日或5日更优",
            ],
        }
        _atomic_json(output / "stage3_report.json", report)
        return report
