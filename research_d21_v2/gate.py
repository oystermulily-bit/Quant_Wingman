"""D21-v2 Stage 4R: absolute D18-v2 plus ablation contrasts. Holdout unread."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_stage4.runner import _rank_ic, _restrict_to_oof
from research_stage4.stats import (
    BootstrapResult,
    holm_adjust,
    performance_from_returns,
    _moving_block_indices,
)

from .protocol import (
    EXPERIMENTS,
    GATE_CANDIDATE,
    HOLM_CONTRASTS,
    NEW_SIGNAL_COLUMNS,
    RELATIVE_BASELINE,
    STAGE4R_PROTOCOL_VERSION,
    D21V2Config,
)


REQUIRED_STAGE3R_FILES = (
    "development_oof_portfolio_daily.parquet",
    "stage3r_report.json",
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


def _median_phase(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def _signed_direction(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _series_by_phase(
    daily: pd.DataFrame,
    experiment: str,
    *,
    field: str = "net_return",
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    subset = daily.loc[daily["experiment"] == experiment].copy()
    if subset.empty:
        raise ValueError(f"no OOF daily rows for experiment={experiment}")
    dates = None
    phases: dict[int, np.ndarray] = {}
    for phase, block in subset.groupby("phase", observed=True):
        folded = block.groupby("date", sort=True)[field].mean().sort_index()
        if dates is None:
            dates = folded.index.to_numpy()
        else:
            folded = folded.reindex(pd.DatetimeIndex(dates))
        phases[int(phase)] = folded.to_numpy(dtype=float)
    if dates is None:
        raise ValueError(f"no OOF phases for experiment={experiment}")
    return dates, phases


def _experiment_metrics(
    daily: pd.DataFrame,
    experiment: str,
    *,
    annual_days: int,
    baseline_sharpe: float,
) -> dict[str, Any]:
    _dates, phases = _series_by_phase(daily, experiment)
    phase_rows = []
    for phase, series in sorted(phases.items()):
        perf = performance_from_returns(series, annual_days=annual_days)
        phase_rows.append({"phase": phase, **perf})
    median_sharpe = _median_phase(phase_rows, "sharpe")
    median_sign = _signed_direction(median_sharpe - baseline_sharpe)
    agreeing = sum(
        1
        for row in phase_rows
        if _signed_direction(row["sharpe"] - baseline_sharpe) == median_sign
        or median_sign == 0
    )
    return {
        "experiment": experiment,
        "phases": phase_rows,
        "median_sharpe": median_sharpe,
        "median_annualized_return": _median_phase(phase_rows, "annualized_return"),
        "median_max_drawdown": _median_phase(phase_rows, "max_drawdown"),
        "worst_phase_sharpe": min(row["sharpe"] for row in phase_rows),
        "phases_agreeing": agreeing,
    }


def paired_ablation_bootstrap(
    candidate_phases: dict[int, np.ndarray],
    baseline_phases: dict[int, np.ndarray],
    dates: np.ndarray,
    *,
    metric: str,
    annual_days: int,
    block_days: int,
    samples: int,
    seed: int,
) -> BootstrapResult:
    frame = pd.DataFrame({"date": pd.to_datetime(dates).normalize()})
    for phase, values in candidate_phases.items():
        frame[f"c{phase}"] = values
    for phase, values in baseline_phases.items():
        frame[f"b{phase}"] = values
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    if frame.empty:
        raise ValueError("no overlapping OOF dates for ablation bootstrap")

    def _delta(table: pd.DataFrame) -> float:
        cand = [
            performance_from_returns(table[f"c{phase}"], annual_days=annual_days)[metric]
            for phase in candidate_phases
        ]
        base = [
            performance_from_returns(table[f"b{phase}"], annual_days=annual_days)[metric]
            for phase in baseline_phases
        ]
        return float(np.median(cand) - np.median(base))

    observed = _delta(frame)
    rng = np.random.default_rng(seed)
    n = len(frame)
    draws = np.empty(samples, dtype=float)
    values = frame.reset_index(drop=True)
    for idx in range(samples):
        index = _moving_block_indices(n, block_days, rng)
        draws[idx] = _delta(values.iloc[index])
    return BootstrapResult(
        observed=observed,
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        p_value=float((1.0 + np.sum(draws <= 0.0)) / (1.0 + samples)),
        samples=samples,
        seed=seed,
        block_days=block_days,
    )


def evaluate_d21_v2_go(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    sharpe_bootstrap: BootstrapResult,
    holm_da: dict[str, Any],
    coverage: float,
    config: D21V2Config,
) -> dict[str, Any]:
    sharpe_delta = float(candidate["median_sharpe"] - baseline["median_sharpe"])
    annual_delta = float(
        candidate["median_annualized_return"] - baseline["median_annualized_return"]
    )
    checks = {
        "oof_coverage": {
            "passed": coverage >= config.min_oof_coverage,
            "observed": coverage,
            "threshold": config.min_oof_coverage,
        },
        "absolute_net_sharpe": {
            "passed": float(candidate["median_sharpe"]) > config.min_net_sharpe,
            "observed": candidate["median_sharpe"],
            "threshold": config.min_net_sharpe,
        },
        "absolute_net_annualized": {
            "passed": float(candidate["median_annualized_return"]) > config.min_net_annualized,
            "observed": candidate["median_annualized_return"],
            "threshold": config.min_net_annualized,
        },
        "sharpe_improvement_vs_A": {
            "passed": sharpe_delta >= config.min_sharpe_improvement,
            "observed": sharpe_delta,
            "threshold": config.min_sharpe_improvement,
        },
        "annualized_improvement_vs_A": {
            "passed": annual_delta >= config.min_annualized_improvement,
            "observed": annual_delta,
            "threshold": config.min_annualized_improvement,
        },
        "bootstrap_ci_lower_positive": {
            "passed": sharpe_bootstrap.ci_low > 0.0,
            "observed": sharpe_bootstrap.ci_low,
            "threshold": 0.0,
        },
        "holm_D_minus_A": {
            "passed": bool(holm_da.get("reject")),
            "observed": holm_da.get("p_holm"),
            "threshold": config.family_alpha,
        },
        "phase_direction": {
            "passed": int(candidate["phases_agreeing"]) >= config.min_phase_agree,
            "observed": candidate["phases_agreeing"],
            "threshold": config.min_phase_agree,
        },
        "fold_direction": {
            "passed": int(candidate["folds_agreeing"]) >= config.min_fold_agree,
            "observed": candidate["folds_agreeing"],
            "threshold": config.min_fold_agree,
        },
        "signal_direction": {
            "passed": int(candidate["signals_agreeing"]) >= config.min_signal_agree,
            "observed": candidate["signals_agreeing"],
            "threshold": config.min_signal_agree,
        },
        "not_single_year_or_group": {
            "passed": bool(candidate["concentration_ok"]),
            "observed": candidate.get("concentration_detail"),
            "threshold": "year/industry/stock leave-one-group vs experiment A",
        },
    }
    return {
        "candidate": GATE_CANDIDATE,
        "baseline": RELATIVE_BASELINE,
        "passed": all(item["passed"] for item in checks.values()),
        "sharpe_delta": sharpe_delta,
        "annualized_delta": annual_delta,
        "checks": checks,
        "thresholds": {
            "min_net_sharpe": config.min_net_sharpe,
            "min_net_annualized": config.min_net_annualized,
            "min_sharpe_improvement": config.min_sharpe_improvement,
            "min_annualized_improvement": config.min_annualized_improvement,
            "min_oof_coverage": config.min_oof_coverage,
            "family_alpha": config.family_alpha,
        },
    }


def classify_d21_v2(go: dict[str, Any], *, coverage_ok: bool) -> dict[str, Any]:
    if not coverage_ok:
        status = "INSUFFICIENT_EVIDENCE"
        sub = "INSUFFICIENT_EVIDENCE"
    elif go["passed"]:
        status = "HORIZON_FEASIBLE_5D_V2"
        sub = "HORIZON_FEASIBLE_5D_V2"
    else:
        status = "D21_V2_FAILED"
        sub = "KEEP_D21_V1_EVIDENCE"
    return {
        "statistical_gate": status,
        "horizon_conclusion": sub,
        "allowed_horizons": [5] if go.get("passed") and coverage_ok else [],
        "system_status": "MODEL_NOT_VALIDATED",
        "rd_agent_allowed": False,
        "holdout_read": False,
        "d21_v1_gate_unchanged": "KEEP_1D_BASELINE",
        "production_eligible": False,
    }


def _fold_agreement(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    overall_sign: int,
    annual_days: int,
) -> int:
    agreeing = 0
    absolute = labels[labels["label_type"] == "ABSOLUTE"]
    for _fold_id, fold_daily in daily.groupby("fold_id", observed=True):
        fold_dates = pd.DatetimeIndex(fold_daily["date"].drop_duplicates())
        cand = fold_daily[fold_daily["experiment"] == GATE_CANDIDATE]
        base = fold_daily[fold_daily["experiment"] == RELATIVE_BASELINE]
        if cand.empty or base.empty:
            continue
        cand_series = cand.groupby("date")["net_return"].median().sort_index()
        base_series = base.groupby("date")["net_return"].median().sort_index()
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
        rank_c = _rank_ic(joined, "slow_residual_ensemble")
        rank_b = _rank_ic(joined, "simple_ensemble")
        if rank_c is None or rank_b is None:
            continue
        rank_delta = float(rank_c - rank_b)
        if overall_sign == 0:
            continue
        if np.sign(net_delta) == overall_sign and np.sign(rank_delta) == overall_sign:
            agreeing += 1
    return agreeing


def _signal_agreement(
    features: pd.DataFrame,
    labels: pd.DataFrame,
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
    simple_ic = _rank_ic(joined, "simple_ensemble")
    deltas: dict[str, float | None] = {}
    agreeing = 0
    for column in NEW_SIGNAL_COLUMNS:
        ic_new = _rank_ic(joined, column)
        if ic_new is None or simple_ic is None:
            deltas[column] = None
            continue
        delta = float(ic_new - simple_ic)
        deltas[column] = delta
        if delta > 0:
            agreeing += 1
    deltas["SIMPLE_ENSEMBLE_V1"] = simple_ic
    return agreeing, deltas


def _concentration_ok(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    annual_days: int,
    ensemble_sign: int,
) -> tuple[bool, dict[str, Any]]:
    dates, phases = _series_by_phase(daily, GATE_CANDIDATE)
    base_dates, base_phases = _series_by_phase(daily, RELATIVE_BASELINE)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "candidate": np.median(np.vstack(list(phases.values())), axis=0),
        }
    )
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(base_dates),
            "baseline": np.median(np.vstack(list(base_phases.values())), axis=0),
        }
    )
    merged = frame.merge(base, on="date", how="inner").dropna()
    merged["year"] = merged["date"].dt.year
    year_ok = 0
    year_deltas: dict[int, float] = {}
    for year in sorted(merged["year"].unique()):
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
        absolute[["signal_date", "code", "value"]],
        left_on=["date", "code"],
        right_on=["signal_date", "code"],
        how="inner",
    )
    industry_ok = True
    industry_detail: dict[str, float | None] = {}
    if "industry_code" in joined.columns:
        full_h = _rank_ic(joined, "slow_residual_ensemble")
        full_1 = _rank_ic(joined, "simple_ensemble")
        if full_h is not None and full_1 is not None:
            full_delta = float(full_h - full_1)
            flips = 0
            industries = [code for code in joined["industry_code"].dropna().unique()]
            for industry in industries:
                sub = joined[joined["industry_code"] != industry]
                sub_h = _rank_ic(sub, "slow_residual_ensemble")
                sub_1 = _rank_ic(sub, "simple_ensemble")
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
                contrib=joined["slow_residual_ensemble"]
                * pd.to_numeric(joined["value"], errors="coerce")
            )
            .groupby("code")["contrib"]
            .mean()
            .sort_values(ascending=False)
        )
        drop = set(contrib.head(max(1, int(np.ceil(0.05 * len(contrib))))).index)
        sub = joined[~joined["code"].isin(drop)]
        sub_h = _rank_ic(sub, "slow_residual_ensemble")
        sub_1 = _rank_ic(sub, "simple_ensemble")
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


def _cost_attribution(
    daily: pd.DataFrame,
    experiment: str,
    *,
    annual_days: int,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if "gross_return" not in daily.columns:
        return {"available": False}
    _dates, net_phases = _series_by_phase(daily, experiment, field="net_return")
    _gross, gross_phases = _series_by_phase(daily, experiment, field="gross_return")
    del _dates, _gross
    phase_rows = []
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
            }
        )
    payload = {
        "available": True,
        "median_gross_sharpe": _median_phase(phase_rows, "gross_sharpe"),
        "median_net_sharpe": _median_phase(phase_rows, "net_sharpe"),
        "median_gross_annualized_return": _median_phase(phase_rows, "gross_annualized_return"),
        "median_net_annualized_return": _median_phase(phase_rows, "net_annualized_return"),
        "median_cost_drag_annualized": _median_phase(phase_rows, "cost_drag_annualized"),
        "phases": phase_rows,
        "enters_go": False,
    }
    if baseline is not None and baseline.get("available"):
        payload["vs_A"] = {
            "gross_alpha_annualized": payload["median_gross_annualized_return"]
            - float(baseline["median_gross_annualized_return"]),
            "cost_reduction_annualized": float(baseline["median_cost_drag_annualized"])
            - payload["median_cost_drag_annualized"],
            "net_annualized": payload["median_net_annualized_return"]
            - float(baseline["median_net_annualized_return"]),
        }
    return payload


class Stage4RFeasibilityRunner:
    expected_stage3_status = "STAGE3R_D21_V2_COMPLETED"
    use_full_oof_coverage = False

    @staticmethod
    def classify(go, coverage_ok):
        return classify_d21_v2(go, coverage_ok=coverage_ok)

    def __init__(self, config: D21V2Config | None = None) -> None:
        self.config = config or D21V2Config()

    def run(
        self,
        *,
        snapshot_dir: str | Path,
        stage3r_dir: str | Path,
        output_dir: str | Path,
        bootstrap_samples: int | None = None,
    ) -> dict[str, Any]:
        del snapshot_dir
        stage3_path = Path(stage3r_dir)
        output = Path(output_dir)
        if output.name == "stage4_hs300_v2_execution_ledger":
            raise PermissionError("D21-v2 must not overwrite D21-v1 Stage 4 artifacts")
        output.mkdir(parents=True, exist_ok=True)
        missing = [name for name in REQUIRED_STAGE3R_FILES if not (stage3_path / name).is_file()]
        if missing:
            payload = {
                "protocol": self.config.stage4r_protocol,
                "status": "STAGE3R_ARTIFACTS_MISSING",
                "statistical_gate": "INSUFFICIENT_EVIDENCE",
                "reason": f"missing Stage-3R artifacts: {missing}",
                "system_status": "MODEL_NOT_VALIDATED",
                "rd_agent_allowed": False,
                "holdout_read": False,
            }
            _atomic_json(output / "stage4r_report.json", payload)
            return payload

        stage3 = json.loads((stage3_path / "stage3r_report.json").read_text(encoding="utf-8"))
        if stage3.get("status") != self.expected_stage3_status:
            payload = {
                "protocol": self.config.stage4r_protocol,
                "status": "STAGE3R_ARTIFACTS_MISSING",
                "statistical_gate": "INSUFFICIENT_EVIDENCE",
                "reason": stage3.get("status"),
                "holdout_read": False,
            }
            _atomic_json(output / "stage4r_report.json", payload)
            return payload

        daily = pd.read_parquet(stage3_path / "development_oof_portfolio_daily.parquet")
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        holdout_start = pd.Timestamp(stage3["split"]["holdout_start"]).normalize()
        if (daily["date"] >= holdout_start).any():
            payload = {
                "protocol": self.config.stage4r_protocol,
                "status": "RESEARCH_GATE_FAILED",
                "statistical_gate": "RESEARCH_GATE_FAILED",
                "reason": "stage3r daily OOF contains Holdout dates",
                "holdout_read": False,
            }
            _atomic_json(output / "stage4r_report.json", payload)
            return payload

        features = pd.read_parquet(stage3_path / "development_features.parquet")
        labels = pd.read_parquet(stage3_path / "development_labels.parquet")
        full_oof_coverage = None
        if self.use_full_oof_coverage:
            masks = [features.date.between(f['validation_start'], f['validation_end'])
                     for f in stage3['split']['folds']]
            full_oof = features.loc[np.logical_or.reduce(masks)]
            if len(full_oof) != 460500 or not full_oof.groupby('date').size().eq(300).all():
                raise ValueError('V3 original OOF member-day denominator changed')
            full_oof_coverage = float(full_oof.valid_slow_signal.fillna(False).mean())
        try:
            daily, features, labels, oof_scope = _restrict_to_oof(
                daily, features, labels, stage3, holdout_start
            )
        except (KeyError, TypeError, ValueError) as exc:
            payload = {
                "protocol": self.config.stage4r_protocol,
                "status": "RESEARCH_GATE_FAILED",
                "statistical_gate": "RESEARCH_GATE_FAILED",
                "reason": f"OOF provenance validation failed: {exc}",
                "holdout_read": False,
            }
            _atomic_json(output / "stage4r_report.json", payload)
            return payload

        if "valid_slow_signal" in features.columns:
            coverage = float(features["valid_slow_signal"].fillna(False).mean())
        else:
            coverage = 0.0
        if full_oof_coverage is not None:
            coverage = full_oof_coverage
        coverage_ok = coverage >= self.config.min_oof_coverage
        if not coverage_ok:
            signal_coverage = {
                column: float(features[column].notna().mean())
                if column in features.columns
                else None
                for column in (*NEW_SIGNAL_COLUMNS, "simple_ensemble")
            }
            payload = {
                "protocol": self.config.stage4r_protocol,
                "status": "INSUFFICIENT_EVIDENCE",
                "statistical_gate": "INSUFFICIENT_EVIDENCE",
                "reason": (
                    f"OOF slow-signal coverage {coverage:.4f} < "
                    f"{self.config.min_oof_coverage}"
                ),
                "oof_slow_coverage": coverage,
                "signal_coverage": signal_coverage,
                "go_not_evaluated": True,
                "horizon_conclusion": "INSUFFICIENT_EVIDENCE",
                "note": (
                    "Coverage failure is fail-closed. A–D net metrics are not used "
                    "to relax windows, Top20/Top40, or the 95% rule."
                ),
                "system_status": "MODEL_NOT_VALIDATED",
                "rd_agent_allowed": False,
                "holdout_read": False,
                "d21_v1_gate_unchanged": "KEEP_1D_BASELINE",
                "production_eligible": False,
            }
            _atomic_json(output / "stage4r_report.json", payload)
            return payload

        samples = int(bootstrap_samples or self.config.bootstrap_samples)
        annual = self.config.annual_trading_days
        experiment_metrics = {
            spec["experiment_id"]: _experiment_metrics(
                daily,
                spec["experiment_id"],
                annual_days=annual,
                baseline_sharpe=0.0,
            )
            for spec in EXPERIMENTS
        }
        baseline = experiment_metrics[RELATIVE_BASELINE]
        candidate = experiment_metrics[GATE_CANDIDATE]
        for experiment_id, metrics in experiment_metrics.items():
            if experiment_id == RELATIVE_BASELINE:
                metrics["phases_agreeing"] = len(metrics["phases"])
                continue
            signed = _experiment_metrics(
                daily,
                experiment_id,
                annual_days=annual,
                baseline_sharpe=baseline["median_sharpe"],
            )
            metrics["phases_agreeing"] = signed["phases_agreeing"]

        overall_sign = int(np.sign(candidate["median_sharpe"] - baseline["median_sharpe"]))
        candidate["folds_agreeing"] = _fold_agreement(
            daily,
            features,
            labels,
            overall_sign=overall_sign,
            annual_days=annual,
        )
        agreeing, signal_deltas = _signal_agreement(features, labels)
        candidate["signals_agreeing"] = agreeing
        candidate["signal_rank_ic_deltas"] = signal_deltas
        concentration_ok, concentration_detail = _concentration_ok(
            daily,
            features,
            labels,
            annual_days=annual,
            ensemble_sign=overall_sign,
        )
        candidate["concentration_ok"] = concentration_ok
        candidate["concentration_detail"] = concentration_detail

        dates_a, phases_a = _series_by_phase(daily, "A")
        bootstrap_payloads: dict[str, Any] = {}
        p_values: dict[str, float] = {}
        for name, left, right in HOLM_CONTRASTS:
            left_dates, left_phases = _series_by_phase(daily, left)
            right_dates, right_phases = _series_by_phase(daily, right)
            aligned_left = pd.DataFrame(
                {"date": pd.to_datetime(left_dates)}
                | {f"p{phase}": values for phase, values in left_phases.items()}
            )
            aligned_right = pd.DataFrame(
                {"date": pd.to_datetime(right_dates)}
                | {f"q{phase}": values for phase, values in right_phases.items()}
            )
            merged = aligned_left.merge(aligned_right, on="date", how="inner").dropna()
            boot = paired_ablation_bootstrap(
                {phase: merged[f"p{phase}"].to_numpy(dtype=float) for phase in left_phases},
                {phase: merged[f"q{phase}"].to_numpy(dtype=float) for phase in right_phases},
                merged["date"].to_numpy(),
                metric="sharpe",
                annual_days=annual,
                block_days=self.config.bootstrap_block_days,
                samples=samples,
                seed=self.config.random_seed,
            )
            bootstrap_payloads[name] = {
                "observed": boot.observed,
                "ci_low": boot.ci_low,
                "ci_high": boot.ci_high,
                "p_value": boot.p_value,
                "samples": boot.samples,
                "seed": boot.seed,
                "block_days": boot.block_days,
            }
            p_values[name] = boot.p_value
            if name == "D_minus_A":
                da_boot = boot
        del dates_a, phases_a
        holm = holm_adjust(p_values, alpha=self.config.family_alpha)

        go = evaluate_d21_v2_go(
            candidate=candidate,
            baseline=baseline,
            sharpe_bootstrap=da_boot,
            holm_da=holm["D_minus_A"],
            coverage=coverage,
            config=self.config,
        )
        gate = self.classify(go, coverage_ok=True)
        cost_a = _cost_attribution(daily, "A", annual_days=annual)
        contrasts = {
            "B_minus_A": "new features on naive Top20",
            "C_minus_A": "buffer only",
            "D_minus_C": "new features given buffer",
            "D_minus_A": "final net improvement",
        }
        report = {
            "protocol": self.config.stage4r_protocol,
            "status": gate["statistical_gate"],
            "stage3r_protocol": stage3.get("protocol"),
            "hypothesis_id": stage3.get("hypothesis_id"),
            "development_adaptive": True,
            "config_hash": self.config.fingerprint(),
            "oof_slow_coverage": coverage,
            "experiments": {
                key: {
                    field: value
                    for field, value in metrics.items()
                    if field != "phases"
                }
                | {"phases": metrics["phases"]}
                for key, metrics in experiment_metrics.items()
            },
            "cost_attribution": {
                spec["experiment_id"]: _cost_attribution(
                    daily,
                    spec["experiment_id"],
                    annual_days=annual,
                    baseline=cost_a if spec["experiment_id"] != "A" else None,
                )
                for spec in EXPERIMENTS
            },
            "bootstrap": bootstrap_payloads,
            "holm": holm,
            "contrasts": contrasts,
            "oof_scope": oof_scope,
            "go": go,
            "restrictions": [
                "Holdout未读取、未计算表现",
                "对照是实验 A（同为 5 日 NAIVE_TOP20），不是 D21-v1 的 1 日",
                "A–D 全部进入 Holm；门禁候选只有 D",
                "毛Alpha/成本节约仅为披露，不修改冻结门槛",
                "D21-v1 KEEP_1D_BASELINE 不得覆盖",
                "即使 HORIZON_FEASIBLE_5D_V2 也不得启动 RD-Agent 或阶段5",
                "不得把 v2 Development 通过解释成生产有效",
            ],
            **gate,
        }
        _atomic_json(output / "stage4r_report.json", report)
        return report
