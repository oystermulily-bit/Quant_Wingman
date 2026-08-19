from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from .config import ResearchConfig
from .schemas import FoldMetrics, ValidationResult


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1))
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with exact tie handling, equivalent to Spearman ranks."""
    values = np.asarray(values).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1))
    if x.size < 3:
        return 0.0
    return _pearson(_ranks(x), _ranks(y))


def _period_ics(factor: np.ndarray, target: np.ndarray, rank: bool) -> list[float]:
    n, t = factor.shape
    corr = _rank_ic if rank else _pearson
    if n > 1:
        values = [corr(factor[:, idx], target[:, idx]) for idx in range(t)]
    else:
        window = max(20, min(252, t // 8 or 20))
        values = [
            corr(factor[0, start : start + window], target[0, start : start + window])
            for start in range(0, max(1, t - window + 1), window)
        ]
    return [value for value in values if math.isfinite(value)]


def _mean_ir(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, mean / (std + 1e-12)


def _bootstrap_ci(values: list[float], config: ResearchConfig) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(config.random_seed)
    block = max(1, min(config.bootstrap_block_size, array.size))
    means = []
    starts = np.arange(max(1, array.size - block + 1))
    block_count = math.ceil(array.size / block)
    for _ in range(config.bootstrap_samples):
        sample = np.concatenate([
            array[start : start + block]
            for start in rng.choice(starts, size=block_count, replace=True)
        ])[: array.size]
        means.append(float(sample.mean()))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _portfolio_metrics(
    factor: np.ndarray, target: np.ndarray, cost: float, periods_per_year: int
) -> dict[str, float]:
    position = np.tanh(np.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0))
    previous = np.zeros_like(position)
    previous[:, 1:] = position[:, :-1]
    turnover = np.abs(position - previous)
    pnl = position * target - turnover * cost
    portfolio = np.nanmean(pnl, axis=0)
    mean = float(np.mean(portfolio)) if portfolio.size else 0.0
    std = float(np.std(portfolio)) if portfolio.size else 0.0
    downside = portfolio[portfolio < 0]
    downside_std = float(np.std(downside)) if downside.size else 0.0
    cumulative = np.cumsum(portfolio)
    drawdown = np.maximum.accumulate(cumulative) - cumulative if cumulative.size else np.array([0.0])
    max_drawdown = float(np.max(drawdown))
    annual = mean * periods_per_year
    return {
        "net_return": float(np.sum(portfolio)),
        "annual_return": annual,
        "net_sharpe": mean / (std + 1e-12) * math.sqrt(periods_per_year),
        "sortino": mean / (downside_std + 1e-12) * math.sqrt(periods_per_year),
        "calmar": annual / (max_drawdown + 1e-12),
        "max_drawdown": max_drawdown,
        "turnover": float(np.mean(turnover)),
        "long_exposure": float(np.mean(np.clip(position, 0, None))),
        "short_exposure": float(np.mean(np.clip(-position, 0, None))),
    }


class ValidationUnit:
    def __init__(
        self,
        config: ResearchConfig,
        periods_per_year: int,
        *,
        timestamps: torch.Tensor | None = None,
        symbols: list[str] | None = None,
        close: torch.Tensor | None = None,
    ):
        self.config = config
        self.periods_per_year = max(1, int(periods_per_year))
        self.timestamps = timestamps
        self.symbols = list(symbols or [])
        self.close = close

    @staticmethod
    def correlations(candidate: torch.Tensor, sota_matrix: torch.Tensor | None) -> list[float]:
        if sota_matrix is None:
            return []
        x = candidate.detach().cpu().numpy().reshape(-1)
        values = []
        for index in range(sota_matrix.shape[1]):
            y = sota_matrix[:, index, :].detach().cpu().numpy().reshape(-1)
            values.append(_pearson(x, y))
        return values

    def evaluate(
        self,
        factor: torch.Tensor,
        target: torch.Tensor,
        folds: list[dict],
        sota_matrix: torch.Tensor | None = None,
        invalid_mask: torch.Tensor | None = None,
    ) -> ValidationResult:
        f = factor.detach().cpu().to(torch.float64).numpy()
        y = target.detach().cpu().to(torch.float64).numpy()
        if f.ndim != 2 or y.ndim != 2 or f.shape[0] != y.shape[0]:
            raise ValueError("factor and target must both have shape [N,T]")
        if f.shape[1] != y.shape[1]:
            common = min(f.shape[1], y.shape[1])
            f, y = f[:, :common], y[:, :common]
        finite = np.isfinite(f) & np.isfinite(y)
        if invalid_mask is not None:
            invalid = invalid_mask.detach().cpu().numpy().astype(bool)
            invalid = invalid[:, : f.shape[1]]
            finite &= ~invalid
        else:
            invalid = ~np.isfinite(f)
        coverage = float(finite.mean()) if finite.size else 0.0
        invalid_fraction = float(invalid.mean()) if invalid.size else 0.0
        f = np.where(finite, f, np.nan)
        y = np.where(finite, y, np.nan)
        failure = None
        valid = True
        if coverage < self.config.min_coverage:
            valid, failure = False, "数据覆盖率不足"
        elif np.nanstd(f) < 1e-8:
            valid, failure = False, "公式退化为常数"

        fold_rows: list[FoldMetrics] = []
        all_ic: list[float] = []
        all_rank_ic: list[float] = []
        oos_parts_f: list[np.ndarray] = []
        oos_parts_y: list[np.ndarray] = []
        for index, fold in enumerate(folds):
            start, end = int(fold["val_start"]), int(fold["val_end"])
            ff, yy = f[:, start:end], y[:, start:end]
            ics = _period_ics(ff, yy, rank=False)
            ranks = _period_ics(ff, yy, rank=True)
            ic = float(np.mean(ics)) if ics else 0.0
            rank_ic = float(np.mean(ranks)) if ranks else 0.0
            pm = _portfolio_metrics(ff, yy, self.config.transaction_cost, self.periods_per_year)
            fold_rows.append(FoldMetrics(index + 1, ic, rank_ic, pm["net_sharpe"], pm["turnover"]))
            all_ic.extend(ics)
            all_rank_ic.extend(ranks)
            oos_parts_f.append(ff)
            oos_parts_y.append(yy)

        if oos_parts_f:
            oos_f, oos_y = np.concatenate(oos_parts_f, axis=1), np.concatenate(oos_parts_y, axis=1)
        else:
            oos_f, oos_y = f, y
        ic, icir = _mean_ir(all_ic)
        rank_ic, rank_icir = _mean_ir(all_rank_ic)
        direction = 0.0
        if all_rank_ic:
            sign = 1.0 if rank_ic >= 0 else -1.0
            direction = float(np.mean(np.asarray(all_rank_ic) * sign > 0))
        ci_low, ci_high = _bootstrap_ci(all_rank_ic, self.config)
        portfolio = _portfolio_metrics(
            oos_f, oos_y, self.config.transaction_cost, self.periods_per_year
        )
        yearly, regimes, symbols = self._segment_metrics(f, y, folds)
        segment_values = [*yearly.values(), *regimes.values(), *symbols.values()]
        if segment_values:
            main_sign = 1.0 if rank_ic >= 0 else -1.0
            segment_direction = float(
                np.mean(np.asarray(segment_values) * main_sign > 0)
            )
        else:
            segment_direction = 1.0
        group_returns, group_monotonicity = self._group_metrics(oos_f, oos_y)
        correlations = self.correlations(factor, sota_matrix)
        max_corr = max((abs(value) for value in correlations), default=0.0)
        return ValidationResult(
            valid=valid, coverage=coverage, ic=ic, icir=icir,
            rank_ic=rank_ic, rank_icir=rank_icir, direction_ratio=direction,
            bootstrap_ci_low=ci_low, bootstrap_ci_high=ci_high,
            max_correlation=max_corr, fold_results=fold_rows,
            failure_reason=failure, invalid_fraction=invalid_fraction,
            yearly_rank_ic=yearly, regime_rank_ic=regimes,
            symbol_rank_ic=symbols, group_returns=group_returns,
            group_monotonicity=group_monotonicity,
            segment_direction_ratio=segment_direction, **portfolio,
        )

    @staticmethod
    def _group_metrics(
        factor: np.ndarray, target: np.ndarray, groups: int = 5
    ) -> tuple[list[float], float]:
        x, y = _finite_pair(factor.reshape(-1), target.reshape(-1))
        if x.size < groups * 5 or np.std(x) < 1e-12:
            return [], 0.0
        ranks = _ranks(x)
        bins = np.minimum(groups - 1, (ranks * groups / x.size).astype(int))
        means = [float(np.mean(y[bins == group])) for group in range(groups)]
        monotonicity = abs(_rank_ic(np.arange(groups), np.asarray(means)))
        return means, float(monotonicity)

    def _segment_metrics(
        self, factor: np.ndarray, target: np.ndarray, folds: list[dict]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        oos = np.zeros(factor.shape, dtype=bool)
        for fold in folds:
            oos[:, int(fold["val_start"]) : int(fold["val_end"])] = True

        yearly: dict[str, float] = {}
        if self.timestamps is not None:
            ts = self.timestamps.detach().cpu().numpy()[:, : factor.shape[1]]
            years = ts.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
            for year in np.unique(years[oos]):
                mask = oos & (years == year)
                if mask.sum() >= 20:
                    yearly[str(int(year))] = _rank_ic(factor[mask], target[mask])

        symbols: dict[str, float] = {}
        for index in range(factor.shape[0]):
            mask = oos[index]
            if mask.sum() >= 20:
                name = self.symbols[index] if index < len(self.symbols) else str(index)
                symbols[name] = _rank_ic(factor[index, mask], target[index, mask])

        regimes: dict[str, float] = {}
        if self.close is not None:
            close = self.close.detach().cpu().numpy()[:, : factor.shape[1]]
            returns = np.zeros_like(close, dtype=np.float64)
            returns[:, 1:] = np.diff(np.log(np.clip(close, 1e-12, None)), axis=1)
            volatility = np.full_like(returns, np.nan)
            window = 20
            for end in range(window, returns.shape[1]):
                volatility[:, end] = np.std(returns[:, end - window : end], axis=1)
            observed = volatility[oos & np.isfinite(volatility)]
            if observed.size >= 30:
                low, high = np.quantile(observed, [1 / 3, 2 / 3])
                labels = {
                    "low": volatility <= low,
                    "mid": (volatility > low) & (volatility < high),
                    "high": volatility >= high,
                }
                for name, regime_mask in labels.items():
                    mask = oos & regime_mask
                    if mask.sum() >= 20:
                        regimes[name] = _rank_ic(factor[mask], target[mask])
        return yearly, regimes, symbols
