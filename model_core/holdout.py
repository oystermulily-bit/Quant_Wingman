"""Strict chronological holdout support for formula research.

The development view exposes only the first 90% of the observations to the
research engine.  The final 10% stays in the full data manager and is evaluated
once, after the winning formula has been frozen.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from strategy_manager.signal import compute_target_positions_stateless

from .backtest import MT5Backtest, estimate_periods_per_year
from .features import MT5FeatureEngineer
from .vm import StackVM


HOLDOUT_PROTOCOL = "chronological_holdout_10_v1"


@dataclass(frozen=True)
class HoldoutSpec:
    fraction: float
    split_index: int
    development_bars: int
    holdout_bars: int
    total_bars: int
    protocol: str = HOLDOUT_PROTOCOL


class DevelopmentDataView:
    """Cloned prefix-only view passed to the formula research engine."""

    def __init__(self, full_data_manager, fraction: float = 0.10):
        if not 0.0 < fraction < 0.5:
            raise ValueError("holdout fraction must be between 0 and 0.5")

        total = int(full_data_manager.target_ret.shape[1])
        holdout_bars = max(3, int(math.ceil(total * fraction)))
        split = total - holdout_bars
        if split < 10:
            raise ValueError(
                f"not enough development data after holdout: total={total}, split={split}"
            )

        self.spec = HoldoutSpec(
            fraction=float(fraction),
            split_index=split,
            development_bars=split,
            holdout_bars=total - split,
            total_bars=total,
        )
        self.symbols = list(getattr(full_data_manager, "symbols", []))
        self.raw_dict = {
            name: value[:, :split].clone()
            for name, value in full_data_manager.raw_dict.items()
        }
        if "open" in self.raw_dict:
            # Recompute from the prefix rather than slicing full-data features.
            # This remains safe even if a future feature is accidentally non-causal.
            self.feat_tensor = MT5FeatureEngineer.compute_features(self.raw_dict)

            # Recompute labels from prefix-only opens.  The two labels adjacent
            # to the boundary cannot use holdout prices.
            open_values = self.raw_dict["open"]
            self.target_ret = torch.zeros_like(open_values, dtype=torch.float32)
            if split >= 3:
                self.target_ret[:, :-2] = torch.log(
                    open_values[:, 2:] / open_values[:, 1:-1].clamp(min=1e-12)
                )
        else:
            # Minimal synthetic managers used by unit tests may expose tensors
            # without raw OHLCV.  Clone their prefix; production managers always
            # take the stricter recomputation branch above.
            self.feat_tensor = full_data_manager.feat_tensor[:, :, :split].clone()
            self.target_ret = full_data_manager.target_ret[:, :split].clone()
            self.target_ret[:, -2:] = 0.0

        self.data_protocol = self.spec.protocol
        self.holdout_fraction = self.spec.fraction
        self.holdout_split_index = self.spec.split_index
        self.full_bars = self.spec.total_bars

    @property
    def bar_time(self) -> torch.Tensor:
        time_values = self.raw_dict.get("time")
        if time_values is None:
            return torch.zeros(len(self.symbols) or 1, dtype=torch.int64)
        return time_values[:, -1].long()


def evaluate_holdout(
    full_data_manager,
    formula: list[int],
    spec: HoldoutSpec,
    *,
    cost_rate: float | None = None,
) -> dict:
    """Evaluate a frozen formula on the untouched suffix without selecting on it."""
    return evaluate_holdout_snapshot(
        full_data_manager, [formula], spec, cost_rate=cost_rate
    )


def evaluate_holdout_snapshot(
    full_data_manager,
    formulas: list[list[int]],
    spec: HoldoutSpec,
    *,
    cost_rate: float | None = None,
) -> dict:
    """Evaluate the equally weighted, fully frozen SOTA formula snapshot."""
    vm = StackVM()
    factors = []
    for formula in formulas:
        factor = vm.execute(formula, full_data_manager.feat_tensor)
        if factor is None:
            raise ValueError("frozen SOTA formula could not be executed on full data")
        factors.append(factor)
    if not factors:
        raise ValueError("frozen SOTA snapshot is empty")
    factor_full = torch.stack(factors, dim=0).mean(dim=0)

    # The final two labels are boundary padding rather than observable returns.
    start = spec.split_index
    end = spec.total_bars - 2
    if end <= start:
        raise ValueError("holdout contains no evaluable forward-return labels")

    factor = factor_full[:, start:end]
    target = full_data_manager.target_ret[:, start:end]
    position = compute_target_positions_stateless(factor)

    previous = torch.zeros_like(position)
    previous[:, 1:] = position[:, :-1]
    turnover = (position - previous).abs()

    bt = MT5Backtest(cost_rate=cost_rate)
    raw_time = full_data_manager.raw_dict.get("time")
    if raw_time is not None:
        bt.periods_per_year = estimate_periods_per_year(raw_time[:, start:end])
    pnl = position * target - turnover * bt.cost_rate
    portfolio_pnl = pnl.mean(dim=0)

    total_log_pnl = float(portfolio_pnl.sum().item())
    mean = portfolio_pnl.mean()
    std = portfolio_pnl.std(unbiased=False)
    sharpe = float((mean / (std + 1e-8) * math.sqrt(bt.periods_per_year)).item())
    sortino = float(bt._sortino(pnl).item())
    calmar = float(bt._calmar(pnl).item())
    cumulative = torch.cumsum(portfolio_pnl, dim=0)
    peak = torch.cummax(cumulative, dim=0).values
    max_drawdown = float((peak - cumulative).max().item())

    try:
        simple_return_equivalent = math.expm1(total_log_pnl)
    except OverflowError:
        simple_return_equivalent = math.inf if total_log_pnl > 0 else -1.0

    return {
        "status": "frozen_sota_evaluated_once",
        "selection_use": False,
        "protocol": spec.protocol,
        "split": asdict(spec),
        "evaluable_bars": int(end - start),
        "formula_count": len(formulas),
        "cost_rate": float(bt.cost_rate),
        "periods_per_year": int(bt.periods_per_year),
        "total_log_pnl": total_log_pnl,
        "simple_return_equivalent": simple_return_equivalent,
        "annualized_log_return": float((mean * bt.periods_per_year).item()),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_log": max_drawdown,
        "mean_abs_exposure": float(position.abs().mean().item()),
        "mean_turnover": float(turnover.mean().item()),
    }


def save_holdout_report(symbol: str, formula: list[int], result: dict) -> Path:
    """Atomically persist the independent report separately from selection score."""
    path = Path("strategies") / f"holdout_{symbol}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"symbol": symbol, "formula": list(formula), **result}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path
