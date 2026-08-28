"""Stage-4 feasibility runner over frozen stage-3 Development OOF artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_stage3.protocol import Stage3Config

from .gate import Stage4Thresholds, classify_gate, evaluate_horizon_go
from .stats import holm_adjust, paired_horizon_bootstrap, performance_from_returns

STAGE4_PROTOCOL_VERSION = "w1ngman_stage4_feasibility_oof_only_v2"
SIGNAL_COLUMNS = ("MOM_20", "REV_5", "LOW_VOL_20")
REQUIRED_STAGE3_FILES = (
    "development_oof_portfolio_daily.parquet",
    "stage3_report.json",
    "development_features.parquet",
    "development_labels.parquet",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank_ic(frame: pd.DataFrame, score: str, value: str = "value") -> float | None:
    values: list[float] = []
    for _date, group in frame.groupby("signal_date", observed=True):
        valid = group[[score, value]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 10 or valid.nunique().min() < 2:
            continue
        corr = valid[score].rank(method="average").corr(valid[value].rank(method="average"))
        if np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return None
    return float(np.mean(values))


def _median_phase(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def _signed_direction(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _restrict_to_oof(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    stage3: dict[str, Any],
    holdout_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate Stage-3 fold provenance and keep only validation-date rows.

    Stage-3 persists full Development features/labels for reproducibility. They
    are not all OOF observations. Stage-4 robustness tests must therefore join
    only dates proven to belong to a Stage-3 validation fold.
    """
    required_daily = {"date", "fold_id", "horizon", "phase", "net_return"}
    missing = sorted(required_daily - set(daily.columns))
    if missing:
        raise ValueError(f"OOF daily artifact missing columns: {missing}")
    if daily.empty or daily["fold_id"].isna().any():
        raise ValueError("OOF daily artifact is empty or has missing fold_id")

    fold_ranges: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for fold in stage3.get("folds", []):
        if "validation_start" not in fold or "validation_end" not in fold:
            continue
        fold_ranges[int(fold["fold_id"])] = (
            pd.Timestamp(fold["validation_start"]).normalize(),
            pd.Timestamp(fold["validation_end"]).normalize(),
        )
    if not fold_ranges:
        raise ValueError("Stage-3 report has no auditable validation fold ranges")

    date_fold = daily[["date", "fold_id"]].drop_duplicates().copy()
    if date_fold.groupby("date")["fold_id"].nunique().gt(1).any():
        raise ValueError("an OOF date is assigned to more than one validation fold")
    used_folds = {int(value) for value in date_fold["fold_id"].unique()}
    if used_folds != set(fold_ranges):
        raise ValueError(
            "dated OOF fold set does not match Stage-3 report: "
            f"daily={sorted(used_folds)}, report={sorted(fold_ranges)}"
        )
    for row in date_fold.itertuples(index=False):
        fold_id = int(row.fold_id)
        if fold_id not in fold_ranges:
            raise ValueError(f"OOF row references unknown fold_id={fold_id}")
        start, end = fold_ranges[fold_id]
        if not start <= pd.Timestamp(row.date) <= end:
            raise ValueError(
                f"OOF date {pd.Timestamp(row.date).date()} falls outside fold {fold_id} "
                f"validation range {start.date()}..{end.date()}"
            )

    oof_dates = pd.DatetimeIndex(sorted(date_fold["date"].unique()))
    if (oof_dates >= holdout_start).any():
        raise ValueError("Stage-3 OOF dates overlap Holdout")

    features = features.copy()
    labels = labels.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"]).dt.normalize()
    if (features["date"] >= holdout_start).any() or (
        labels["signal_date"] >= holdout_start
    ).any():
        raise ValueError("Development feature/label artifact contains Holdout dates")

    feature_rows_before = len(features)
    label_rows_before = len(labels)
    features = features[features["date"].isin(oof_dates)].copy()
    labels = labels[labels["signal_date"].isin(oof_dates)].copy()
    if features.empty or labels.empty:
        raise ValueError("no feature/label rows remain after OOF date restriction")
    return daily, features, labels, {
        "oof_only": True,
        "oof_date_count": int(len(oof_dates)),
        "oof_start": oof_dates[0].date().isoformat(),
        "oof_end": oof_dates[-1].date().isoformat(),
        "feature_rows_used": int(len(features)),
        "feature_rows_excluded_non_oof": int(feature_rows_before - len(features)),
        "label_rows_used": int(len(labels)),
        "label_rows_excluded_non_oof": int(label_rows_before - len(labels)),
    }


def _series_by_phase(
    daily: pd.DataFrame,
    horizon: int,
    field: str = "net_return",
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    subset = daily.loc[daily["horizon"] == horizon].copy()
    if field not in subset.columns:
        raise ValueError(f"OOF daily artifact missing column: {field}")
    subset["date"] = pd.to_datetime(subset["date"]).dt.normalize()
    dates = None
    phases: dict[int, np.ndarray] = {}
    for phase, block in subset.groupby("phase", observed=True):
        folded = (
            block.groupby("date", sort=True)[field]
            .mean()
            .sort_index()
        )
        if dates is None:
            dates = folded.index.to_numpy()
        else:
            folded = folded.reindex(pd.DatetimeIndex(dates))
        phases[int(phase)] = folded.to_numpy(dtype=float)
    if dates is None:
        raise ValueError(f"no OOF daily rows for horizon={horizon}")
    return dates, phases


def _horizon_metrics(
    daily: pd.DataFrame,
    horizon: int,
    *,
    annual_days: int,
    baseline_sharpe: float,
) -> dict[str, Any]:
    _dates, phases = _series_by_phase(daily, horizon)
    phase_rows = []
    for phase, series in sorted(phases.items()):
        perf = performance_from_returns(series, annual_days=annual_days)
        phase_rows.append({"phase": phase, **perf})
    median_sharpe = _median_phase(phase_rows, "sharpe")
    median_sign = _signed_direction(median_sharpe - baseline_sharpe)
    if horizon == 1:
        agreeing = len(phase_rows)
    else:
        agreeing = sum(
            1
            for row in phase_rows
            if _signed_direction(row["sharpe"] - baseline_sharpe) == median_sign
        )
    return {
        "horizon": horizon,
        "phases": phase_rows,
        "median_sharpe": median_sharpe,
        "median_annualized_return": _median_phase(phase_rows, "annualized_return"),
        "median_max_drawdown": _median_phase(phase_rows, "max_drawdown"),
        "worst_phase_sharpe": min(row["sharpe"] for row in phase_rows),
        "phases_agreeing": agreeing,
    }


def _oof_cost_attribution(
    daily: pd.DataFrame,
    horizon: int,
    *,
    annual_days: int,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Disclose gross alpha vs cost reduction vs net. Does not enter GO checks."""
    if "gross_return" not in daily.columns:
        return {"available": False, "reason": "OOF daily artifact has no gross_return"}
    _dates, net_phases = _series_by_phase(daily, horizon, field="net_return")
    _gross_dates, gross_phases = _series_by_phase(daily, horizon, field="gross_return")
    del _dates, _gross_dates
    phase_rows: list[dict[str, Any]] = []
    for phase in sorted(net_phases):
        net = performance_from_returns(net_phases[phase], annual_days=annual_days)
        gross = performance_from_returns(gross_phases[phase], annual_days=annual_days)
        phase_rows.append(
            {
                "phase": phase,
                "gross_sharpe": gross["sharpe"],
                "net_sharpe": net["sharpe"],
                "gross_annualized_return": gross["annualized_return"],
                "net_annualized_return": net["annualized_return"],
                "cost_drag_annualized": gross["annualized_return"] - net["annualized_return"],
                "cost_drag_sharpe": gross["sharpe"] - net["sharpe"],
            }
        )
    median_gross_ann = _median_phase(phase_rows, "gross_annualized_return")
    median_net_ann = _median_phase(phase_rows, "net_annualized_return")
    median_gross_sharpe = _median_phase(phase_rows, "gross_sharpe")
    median_net_sharpe = _median_phase(phase_rows, "net_sharpe")
    cost_drag_ann = median_gross_ann - median_net_ann
    payload: dict[str, Any] = {
        "available": True,
        "source": "development_oof_portfolio_daily.parquet cost x1",
        "median_gross_sharpe": median_gross_sharpe,
        "median_net_sharpe": median_net_sharpe,
        "median_gross_annualized_return": median_gross_ann,
        "median_net_annualized_return": median_net_ann,
        "median_cost_drag_annualized": cost_drag_ann,
        "median_cost_drag_sharpe": median_gross_sharpe - median_net_sharpe,
        "phases": phase_rows,
        "enters_go": False,
        "note": (
            "Disclosure only. GO uses net Sharpe and net annualized return at cost x1. "
            "Thresholds are not changed by this split."
        ),
    }
    turnover_field = "turnover" if "turnover" in daily.columns else None
    if turnover_field:
        subset = daily.loc[daily["horizon"] == horizon]
        phase_turnover = [
            float(block[turnover_field].mean())
            for _, block in subset.groupby("phase", observed=True)
        ]
        payload["median_mean_daily_turnover"] = float(np.median(phase_turnover)) if phase_turnover else None
    if baseline is not None:
        gross_delta = median_gross_ann - float(baseline["median_gross_annualized_return"])
        net_delta = median_net_ann - float(baseline["median_net_annualized_return"])
        cost_reduction = float(baseline["median_cost_drag_annualized"]) - cost_drag_ann
        payload["vs_1d"] = {
            "gross_alpha_annualized": gross_delta,
            "cost_reduction_annualized": cost_reduction,
            "net_annualized": net_delta,
            "identity_residual": abs((gross_delta + cost_reduction) - net_delta),
            "note": "net = gross_alpha + cost_reduction; positive cost_reduction means lower cost drag than 1d",
        }
    return payload


def _fold_agreement(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int,
    *,
    overall_sign: int,
    annual_days: int,
) -> int:
    """Count fold agreement using only dated OOF rows, never report summaries."""
    agreeing = 0
    absolute = labels[labels["label_type"] == "ABSOLUTE"]
    for fold_id, fold_daily in daily.groupby("fold_id", observed=True):
        fold_dates = pd.DatetimeIndex(fold_daily["date"].drop_duplicates())
        candidate = fold_daily[fold_daily["horizon"] == horizon]
        baseline = fold_daily[fold_daily["horizon"] == 1]
        if candidate.empty or baseline.empty:
            continue
        cand_series = candidate.groupby("date")["net_return"].median().sort_index()
        base_series = baseline.groupby("date")["net_return"].median().sort_index()
        aligned = pd.concat(
            [cand_series.rename("candidate"), base_series.rename("baseline")], axis=1
        ).dropna()
        if len(aligned) < 2:
            continue
        net_delta = float(
            performance_from_returns(aligned["candidate"], annual_days=annual_days)[
                "annualized_return"
            ]
            - performance_from_returns(aligned["baseline"], annual_days=annual_days)[
                "annualized_return"
            ]
        )
        fold_features = features[features["date"].isin(fold_dates)]
        fold_labels = absolute[absolute["signal_date"].isin(fold_dates)]
        joined = fold_features.merge(
            fold_labels[["signal_date", "code", "horizon", "value"]],
            left_on=["date", "code"],
            right_on=["signal_date", "code"],
            how="inner",
        )
        rank_c = _rank_ic(joined[joined["horizon"] == horizon], "simple_ensemble")
        rank_b = _rank_ic(joined[joined["horizon"] == 1], "simple_ensemble")
        if rank_c is None or rank_b is None:
            continue
        rank_delta = float(rank_c - rank_b)
        if overall_sign == 0:
            continue
        if np.sign(net_delta) == overall_sign and np.sign(rank_delta) == overall_sign:
            agreeing += 1
    return agreeing


def _cost_breakdown(stage3: dict[str, Any], horizon: int) -> dict[str, Any]:
    by_mult: dict[str, list[dict[str, float]]] = {"0.0": [], "1.0": [], "1.5": []}
    for fold in stage3.get("folds", []):
        for phase in fold["horizons"][str(horizon)]["phases"]:
            sensitivity = phase.get("cost_sensitivity", {})
            for key in by_mult:
                payload = sensitivity.get(key, phase if key == "1.0" else None)
                if payload is None:
                    continue
                by_mult[key].append(
                    {
                        "sharpe": float(payload["sharpe"]),
                        "annualized_return": float(payload["annualized_return"]),
                        "gross_total_return": float(payload["gross_total_return"]),
                        "net_total_return": float(payload["net_total_return"]),
                    }
                )
    summary = {}
    for key, rows in by_mult.items():
        if not rows:
            summary[key] = None
            continue
        summary[key] = {
            "median_sharpe": float(np.median([row["sharpe"] for row in rows])),
            "median_annualized_return": float(
                np.median([row["annualized_return"] for row in rows])
            ),
        }
    one = summary.get("1.0") or {}
    return {
        "by_multiplier": summary,
        "gross_minus_net_hint": None
        if not one
        else "gross/net split is disclosed per fold in stage3_report; medians use cost x1",
        "median_net_sharpe_x1": one.get("median_sharpe"),
    }


def _signal_agreement(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int,
    *,
    ensemble_sign: int,
) -> tuple[int, dict[str, float | None]]:
    absolute = labels[labels["label_type"] == "ABSOLUTE"][
        ["signal_date", "code", "horizon", "value"]
    ]
    joined = features.merge(
        absolute,
        left_on=["date", "code"],
        right_on=["signal_date", "code"],
        how="inner",
    )
    deltas: dict[str, float | None] = {}
    agreeing = 0
    for column in SIGNAL_COLUMNS:
        ic_h = _rank_ic(joined[joined["horizon"] == horizon], column)
        ic_1 = _rank_ic(joined[joined["horizon"] == 1], column)
        if ic_h is None or ic_1 is None:
            deltas[column] = None
            continue
        delta = float(ic_h - ic_1)
        deltas[column] = delta
        if ensemble_sign == 0:
            continue
        if np.sign(delta) == ensemble_sign:
            agreeing += 1
    return agreeing, deltas


def _concentration_ok(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int,
    *,
    annual_days: int,
    ensemble_sign: int,
) -> tuple[bool, dict[str, Any]]:
    dates, phases = _series_by_phase(daily, horizon)
    baseline_dates, baseline_phases = _series_by_phase(daily, 1)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "candidate": np.median(np.vstack(list(phases.values())), axis=0),
        }
    )
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(baseline_dates),
            "baseline": baseline_phases[0],
        }
    )
    merged = frame.merge(base, on="date", how="inner").dropna()
    merged["year"] = merged["date"].dt.year
    year_ok = 0
    years = sorted(merged["year"].unique())
    year_deltas = {}
    for year in years:
        keep = merged[merged["year"] != year]
        if len(keep) < 40:
            continue
        cand = performance_from_returns(keep["candidate"], annual_days=annual_days)["sharpe"]
        base_s = performance_from_returns(keep["baseline"], annual_days=annual_days)["sharpe"]
        delta = float(cand - base_s)
        year_deltas[int(year)] = delta
        if ensemble_sign == 0 or np.sign(delta) == ensemble_sign:
            year_ok += 1
    year_pass = year_ok >= min(2, max(1, len(year_deltas) - 1)) and len(year_deltas) >= 2

    absolute = labels[labels["label_type"] == "ABSOLUTE"]
    joined = features.merge(
        absolute[absolute["horizon"] == horizon][["signal_date", "code", "value"]],
        left_on=["date", "code"],
        right_on=["signal_date", "code"],
        how="inner",
    )
    joined_1 = features.merge(
        absolute[absolute["horizon"] == 1][["signal_date", "code", "value"]],
        left_on=["date", "code"],
        right_on=["signal_date", "code"],
        how="inner",
    )
    industry_ok = True
    industry_detail: dict[str, float | None] = {}
    if "industry_code" in joined.columns:
        industries = [code for code in joined["industry_code"].dropna().unique()]
        full_h = _rank_ic(joined, "simple_ensemble")
        full_1 = _rank_ic(joined_1, "simple_ensemble")
        if full_h is not None and full_1 is not None:
            full_delta = float(full_h - full_1)
            flips = 0
            for industry in industries:
                sub_h = _rank_ic(joined[joined["industry_code"] != industry], "simple_ensemble")
                sub_1 = _rank_ic(joined_1[joined_1["industry_code"] != industry], "simple_ensemble")
                if sub_h is None or sub_1 is None:
                    continue
                delta = float(sub_h - sub_1)
                industry_detail[str(industry)] = delta
                if np.sign(full_delta) != 0 and np.sign(delta) != np.sign(full_delta):
                    flips += 1
            industry_ok = flips == 0 or (flips / max(len(industry_detail), 1) < 0.5)

    stock_ok = True
    if "code" in joined.columns and ensemble_sign != 0:
        contrib = (
            joined.assign(
                contrib=joined["simple_ensemble"] * pd.to_numeric(joined["value"], errors="coerce")
            )
            .groupby("code")["contrib"]
            .mean()
            .sort_values(ascending=False)
        )
        drop = set(contrib.head(max(1, int(np.ceil(0.05 * len(contrib))))).index)
        sub_h = _rank_ic(joined[~joined["code"].isin(drop)], "simple_ensemble")
        sub_1 = _rank_ic(joined_1[~joined_1["code"].isin(drop)], "simple_ensemble")
        if sub_h is not None and sub_1 is not None:
            stock_ok = bool(
                np.sign(float(sub_h - sub_1)) == ensemble_sign
                or abs(float(sub_h - sub_1)) < 1e-12
            )

    detail = {
        "year_deltas": year_deltas,
        "years_agreeing": year_ok,
        "industry_leave_one_out": industry_detail,
        "stock_drop_top5pct_ok": stock_ok,
    }
    return bool(year_pass and industry_ok and stock_ok), detail


class Stage4FeasibilityRunner:
    def __init__(
        self,
        config: Stage3Config | None = None,
        thresholds: Stage4Thresholds | None = None,
    ) -> None:
        self.config = config or Stage3Config()
        self.thresholds = thresholds or Stage4Thresholds()

    def run(
        self,
        *,
        snapshot_dir: str | Path,
        stage3_dir: str | Path,
        output_dir: str | Path,
        expected_members: int = 300,
        required_start: str | None = "2014-01-02",
        bootstrap_samples: int | None = None,
    ) -> dict[str, Any]:
        stage3_path = Path(stage3_dir)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        daily_path = stage3_path / "development_oof_portfolio_daily.parquet"
        report_path = stage3_path / "stage3_report.json"
        feature_path = stage3_path / "development_features.parquet"
        label_path = stage3_path / "development_labels.parquet"
        missing = [
            name
            for name in REQUIRED_STAGE3_FILES
            if not (stage3_path / name).is_file()
        ]
        if missing:
            payload = {
                "protocol": STAGE4_PROTOCOL_VERSION,
                "status": "STAGE3_ARTIFACTS_MISSING",
                "statistical_gate": "INSUFFICIENT_EVIDENCE",
                "reason": f"missing Stage-3 artifacts: {missing}",
                "system_status": "MODEL_NOT_VALIDATED",
                "rd_agent_allowed": False,
                "holdout_read": False,
            }
            _atomic_json(output / "stage4_report.json", payload)
            return payload
        stage3 = json.loads(report_path.read_text(encoding="utf-8"))
        if stage3.get("status") != "STAGE3_MINIMUM_LOOP_COMPLETED":
            payload = {
                "protocol": STAGE4_PROTOCOL_VERSION,
                "status": "STAGE3_ARTIFACTS_MISSING",
                "statistical_gate": "INSUFFICIENT_EVIDENCE",
                "reason": stage3.get("status"),
            }
            _atomic_json(output / "stage4_report.json", payload)
            return payload

        daily = pd.read_parquet(daily_path)
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        holdout_start = pd.Timestamp(stage3["split"]["holdout_start"]).normalize()
        if (daily["date"] >= holdout_start).any():
            payload = {
                "protocol": STAGE4_PROTOCOL_VERSION,
                "status": "RESEARCH_GATE_FAILED",
                "statistical_gate": "RESEARCH_GATE_FAILED",
                "reason": "stage3 daily OOF contains Holdout dates",
                "system_status": "MODEL_NOT_VALIDATED",
            }
            _atomic_json(output / "stage4_report.json", payload)
            return payload

        features = pd.read_parquet(feature_path)
        labels = pd.read_parquet(label_path)
        try:
            daily, features, labels, oof_scope = _restrict_to_oof(
                daily, features, labels, stage3, holdout_start
            )
        except (KeyError, TypeError, ValueError) as exc:
            payload = {
                "protocol": STAGE4_PROTOCOL_VERSION,
                "status": "RESEARCH_GATE_FAILED",
                "statistical_gate": "RESEARCH_GATE_FAILED",
                "reason": f"OOF provenance validation failed: {exc}",
                "system_status": "MODEL_NOT_VALIDATED",
                "holdout_read": False,
            }
            _atomic_json(output / "stage4_report.json", payload)
            return payload
        samples = int(bootstrap_samples or self.config.bootstrap_samples)
        annual = self.config.annual_trading_days
        baseline = _horizon_metrics(daily, 1, annual_days=annual, baseline_sharpe=0.0)
        baseline_cost = _oof_cost_attribution(daily, 1, annual_days=annual)
        horizon_payloads: dict[int, dict[str, Any]] = {}
        bootstrap_payloads: dict[str, Any] = {}
        p_values: dict[str, float] = {}

        base_dates, base_phases = _series_by_phase(daily, 1)
        for horizon in (3, 5):
            cand_dates, cand_phases = _series_by_phase(daily, horizon)
            boot = paired_horizon_bootstrap(
                cand_phases,
                cand_dates,
                base_phases[0],
                base_dates,
                metric="sharpe",
                annual_days=annual,
                block_days=self.config.bootstrap_block_days,
                samples=samples,
                seed=self.config.random_seed,
            )
            key = f"{horizon}d_vs_1d"
            bootstrap_payloads[key] = {
                "observed": boot.observed,
                "ci_low": boot.ci_low,
                "ci_high": boot.ci_high,
                "p_value": boot.p_value,
                "samples": boot.samples,
                "seed": boot.seed,
                "block_days": boot.block_days,
            }
            p_values[key] = boot.p_value
            metrics = _horizon_metrics(
                daily, horizon, annual_days=annual, baseline_sharpe=baseline["median_sharpe"]
            )
            overall_sign = int(np.sign(metrics["median_sharpe"] - baseline["median_sharpe"]))
            metrics["folds_agreeing"] = _fold_agreement(
                daily,
                features,
                labels,
                horizon,
                overall_sign=overall_sign,
                annual_days=annual,
            )
            agreeing, signal_deltas = _signal_agreement(
                features, labels, horizon, ensemble_sign=overall_sign
            )
            metrics["signals_agreeing"] = agreeing
            metrics["signal_rank_ic_deltas"] = signal_deltas
            concentration_ok, concentration_detail = _concentration_ok(
                daily,
                features,
                labels,
                horizon,
                annual_days=annual,
                ensemble_sign=overall_sign,
            )
            metrics["concentration_ok"] = concentration_ok
            metrics["concentration_detail"] = concentration_detail
            metrics["cost_sensitivity"] = _cost_breakdown(stage3, horizon)
            metrics["cost_attribution"] = _oof_cost_attribution(
                daily,
                horizon,
                annual_days=annual,
                baseline=baseline_cost if baseline_cost.get("available") else None,
            )
            horizon_payloads[horizon] = metrics
            horizon_payloads[horizon]["_bootstrap"] = boot

        holm = holm_adjust(p_values, alpha=self.thresholds.family_alpha)
        go_results: dict[int, dict[str, Any]] = {}
        for horizon in (3, 5):
            boot: Any = horizon_payloads[horizon].pop("_bootstrap")
            go_results[horizon] = evaluate_horizon_go(
                horizon=horizon,
                candidate=horizon_payloads[horizon],
                baseline=baseline,
                sharpe_bootstrap=boot,
                holm=holm[f"{horizon}d_vs_1d"],
                thresholds=self.thresholds,
            )

        gate = classify_gate(go_results)
        report = {
            "protocol": STAGE4_PROTOCOL_VERSION,
            "status": gate["statistical_gate"],
            "stage3_protocol": stage3.get("protocol"),
            "snapshot_id": stage3.get("snapshot_id"),
            "snapshot_dir": str(snapshot_dir),
            "config_hash": self.config.fingerprint(),
            "thresholds": {
                "min_sharpe_improvement": self.thresholds.min_sharpe_improvement,
                "min_annualized_improvement": self.thresholds.min_annualized_improvement,
                "family_alpha": self.thresholds.family_alpha,
                "bootstrap_block_days": self.config.bootstrap_block_days,
                "bootstrap_samples": samples,
                "random_seed": self.config.random_seed,
            },
            "baseline_1d": {key: value for key, value in baseline.items() if key != "phases"}
            | {"phases": baseline["phases"], "cost_attribution": baseline_cost},
            "horizons": {str(key): value for key, value in horizon_payloads.items()},
            "bootstrap": bootstrap_payloads,
            "holm": holm,
            "oof_scope": oof_scope,
            "go": {str(key): value for key, value in go_results.items()},
            "restrictions": [
                "Holdout未读取、未计算表现",
                "所有稳健性、集中度与信号一致性统计仅使用Development OOF日期",
                "毛Alpha/成本节约/净改善仅为披露，不进入GO判定、不修改预注册阈值",
                "阶段4不自动重跑阶段3；缺失产物时失败关闭",
                "RD-Agent仅在 RESEARCH_GATE_PASSED 后才允许启动",
                "不得根据本报告修改预注册GO阈值",
                "本门禁不等于 PRODUCTION_ELIGIBLE",
            ],
            **gate,
        }
        _atomic_json(output / "stage4_report.json", report)
        return report
