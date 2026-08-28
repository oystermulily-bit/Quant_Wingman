"""Pre-registered GO evaluation. Thresholds are frozen and not result-dependent."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .stats import BootstrapResult


@dataclass(frozen=True)
class Stage4Thresholds:
    min_sharpe_improvement: float = 0.15
    min_annualized_improvement: float = 0.02
    family_alpha: float = 0.05
    worst_phase_sharpe_gap: float = 0.25
    max_drawdown_ratio: float = 1.2
    min_phase_agree_3: int = 2
    min_phase_agree_5: int = 3
    min_fold_agree: int = 3
    min_signal_agree: int = 2
    min_year_agree: int = 2


def _sign(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def evaluate_horizon_go(
    *,
    horizon: int,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    sharpe_bootstrap: BootstrapResult,
    holm: dict[str, Any],
    thresholds: Stage4Thresholds | None = None,
) -> dict[str, Any]:
    cfg = thresholds or Stage4Thresholds()
    sharpe_delta = float(candidate["median_sharpe"] - baseline["median_sharpe"])
    annual_delta = float(
        candidate["median_annualized_return"] - baseline["median_annualized_return"]
    )
    required_phases = cfg.min_phase_agree_3 if horizon == 3 else cfg.min_phase_agree_5
    checks = {
        "sharpe_improvement": {
            "passed": sharpe_delta >= cfg.min_sharpe_improvement,
            "observed": sharpe_delta,
            "threshold": cfg.min_sharpe_improvement,
        },
        "annualized_improvement": {
            "passed": annual_delta >= cfg.min_annualized_improvement,
            "observed": annual_delta,
            "threshold": cfg.min_annualized_improvement,
        },
        "bootstrap_ci_lower_positive": {
            "passed": sharpe_bootstrap.ci_low > 0.0,
            "observed": sharpe_bootstrap.ci_low,
            "threshold": 0.0,
        },
        "holm_family": {
            "passed": bool(holm.get("reject")),
            "observed": holm.get("p_holm"),
            "threshold": cfg.family_alpha,
        },
        "phase_direction": {
            "passed": int(candidate["phases_agreeing"]) >= required_phases,
            "observed": candidate["phases_agreeing"],
            "threshold": required_phases,
        },
        "worst_phase_floor": {
            "passed": float(candidate["worst_phase_sharpe"])
            >= float(baseline["median_sharpe"]) - cfg.worst_phase_sharpe_gap,
            "observed": candidate["worst_phase_sharpe"] - baseline["median_sharpe"],
            "threshold": -cfg.worst_phase_sharpe_gap,
        },
        "fold_direction": {
            "passed": int(candidate["folds_agreeing"]) >= cfg.min_fold_agree,
            "observed": candidate["folds_agreeing"],
            "threshold": cfg.min_fold_agree,
        },
        "drawdown_bound": {
            "passed": float(candidate["median_max_drawdown"])
            <= float(baseline["median_max_drawdown"]) * cfg.max_drawdown_ratio + 1e-12,
            "observed": candidate["median_max_drawdown"],
            "threshold": baseline["median_max_drawdown"] * cfg.max_drawdown_ratio,
        },
        "signal_direction": {
            "passed": int(candidate["signals_agreeing"]) >= cfg.min_signal_agree,
            "observed": candidate["signals_agreeing"],
            "threshold": cfg.min_signal_agree,
        },
        "not_single_year_or_group": {
            "passed": bool(candidate["concentration_ok"]),
            "observed": candidate.get("concentration_detail"),
            "threshold": "year/industry/stock leave-one-group",
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "horizon": horizon,
        "passed": passed,
        "sharpe_delta": sharpe_delta,
        "annualized_delta": annual_delta,
        "checks": checks,
        "thresholds": asdict(cfg),
    }


def classify_gate(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    passed_3 = bool(results.get(3, {}).get("passed"))
    passed_5 = bool(results.get(5, {}).get("passed"))
    if passed_3 and passed_5:
        status = "RESEARCH_GATE_PASSED"
        sub = "HORIZON_FEASIBLE_BOTH"
        allowed = [3, 5]
    elif passed_3:
        status = "RESEARCH_GATE_PASSED"
        sub = "HORIZON_FEASIBLE_3D"
        allowed = [3]
    elif passed_5:
        status = "RESEARCH_GATE_PASSED"
        sub = "HORIZON_FEASIBLE_5D"
        allowed = [5]
    else:
        status = "KEEP_1D_BASELINE"
        sub = "KEEP_1D_BASELINE"
        allowed = [1]
    return {
        "statistical_gate": status,
        "horizon_conclusion": sub,
        "allowed_horizons": allowed,
        "system_status": "MODEL_NOT_VALIDATED",
        "rd_agent_allowed": status == "RESEARCH_GATE_PASSED",
        "holdout_read": False,
    }
