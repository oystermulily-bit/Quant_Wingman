"""Frozen uncertainty estimates for stage 4."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def performance_from_returns(
    returns: np.ndarray | pd.Series,
    *,
    annual_days: int,
) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "observations": 0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }
    total = float(np.prod(1.0 + values) - 1.0)
    years = values.size / annual_days
    annual = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else -1.0
    std = float(values.std(ddof=0))
    sharpe = float(values.mean() / std * np.sqrt(annual_days)) if std > 1e-12 else 0.0
    downside = values[values < 0.0]
    downside_std = float(downside.std(ddof=0)) if downside.size else 0.0
    sortino = (
        float(values.mean() / downside_std * np.sqrt(annual_days)) if downside_std > 1e-12 else 0.0
    )
    equity = np.cumprod(1.0 + values)
    drawdown = 1.0 - equity / np.maximum.accumulate(equity)
    max_dd = float(drawdown.max()) if drawdown.size else 0.0
    calmar = float(annual / max_dd) if max_dd > 1e-12 else 0.0
    return {
        "observations": int(values.size),
        "total_return": total,
        "annualized_return": annual,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def _moving_block_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=int)
    width = min(max(int(block), 1), length)
    n_blocks = int(np.ceil(length / width))
    max_start = length - width + 1
    starts = rng.integers(0, max_start, size=n_blocks)
    stacked = np.concatenate([np.arange(start, start + width) for start in starts])
    return stacked[:length]


@dataclass(frozen=True)
class BootstrapResult:
    observed: float
    ci_low: float
    ci_high: float
    p_value: float
    samples: int
    seed: int
    block_days: int


def moving_block_bootstrap(
    delta_series: np.ndarray,
    *,
    statistic,
    block_days: int,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapResult:
    """One-sided p-value that the statistic is > 0, plus a two-sided percentile CI."""
    series = np.asarray(delta_series, dtype=float)
    observed = float(statistic(series))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for idx in range(samples):
        index = _moving_block_indices(len(series), block_days, rng)
        draws[idx] = float(statistic(series[index]))
    lower = float(np.quantile(draws, alpha / 2.0))
    upper = float(np.quantile(draws, 1.0 - alpha / 2.0))
    p_value = float((1.0 + np.sum(draws <= 0.0)) / (1.0 + samples))
    return BootstrapResult(
        observed=observed,
        ci_low=lower,
        ci_high=upper,
        p_value=p_value,
        samples=samples,
        seed=seed,
        block_days=block_days,
    )


def paired_horizon_bootstrap(
    candidate_phases: dict[int, np.ndarray],
    candidate_dates: np.ndarray,
    baseline: np.ndarray,
    baseline_dates: np.ndarray,
    *,
    metric: str,
    annual_days: int,
    block_days: int,
    samples: int,
    seed: int,
) -> BootstrapResult:
    """Apply the same moving blocks to every phase and the 1-day baseline."""
    aligned = _align_phase_matrix(
        candidate_phases, candidate_dates, baseline, baseline_dates
    )
    observed = _median_metric_delta(aligned, metric=metric, annual_days=annual_days)
    rng = np.random.default_rng(seed)
    n = aligned["baseline"].shape[0]
    draws = np.empty(samples, dtype=float)
    for idx in range(samples):
        index = _moving_block_indices(n, block_days, rng)
        resampled = {
            "baseline": aligned["baseline"][index],
            "phases": {phase: values[index] for phase, values in aligned["phases"].items()},
        }
        draws[idx] = _median_metric_delta(resampled, metric=metric, annual_days=annual_days)
    lower = float(np.quantile(draws, 0.025))
    upper = float(np.quantile(draws, 0.975))
    p_value = float((1.0 + np.sum(draws <= 0.0)) / (1.0 + samples))
    return BootstrapResult(
        observed=observed,
        ci_low=lower,
        ci_high=upper,
        p_value=p_value,
        samples=samples,
        seed=seed,
        block_days=block_days,
    )


def _align_phase_matrix(
    candidate_phases: dict[int, np.ndarray],
    candidate_dates: np.ndarray,
    baseline: np.ndarray,
    baseline_dates: np.ndarray,
) -> dict:
    frame = pd.DataFrame(
        {"date": pd.to_datetime(candidate_dates).normalize()}
        | {f"p{phase}": values for phase, values in candidate_phases.items()}
    )
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(baseline_dates).normalize(),
            "baseline": baseline,
        }
    )
    frame = frame.merge(base, on="date", how="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    if frame.empty:
        raise ValueError("no overlapping OOF dates for paired bootstrap")
    return {
        "dates": frame["date"].to_numpy(),
        "baseline": frame["baseline"].to_numpy(dtype=float),
        "phases": {
            phase: frame[f"p{phase}"].to_numpy(dtype=float)
            for phase in candidate_phases
        },
    }


def _median_metric_delta(aligned: dict, *, metric: str, annual_days: int) -> float:
    baseline = performance_from_returns(aligned["baseline"], annual_days=annual_days)[metric]
    values = [
        performance_from_returns(series, annual_days=annual_days)[metric]
        for series in aligned["phases"].values()
    ]
    return float(np.median(values) - baseline)


def holm_adjust(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, dict]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    family = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        raw = min(1.0, float(p_value) * (family - rank))
        running = max(running, raw)
        adjusted[name] = running
    return {
        name: {
            "p_value": float(p_values[name]),
            "p_holm": adjusted[name],
            "reject": adjusted[name] < alpha,
            "alpha": alpha,
        }
        for name in p_values
    }
