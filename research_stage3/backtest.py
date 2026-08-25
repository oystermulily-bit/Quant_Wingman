"""Deterministic Top-20 long-only reference backtest with directional costs."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .protocol import Stage3Config


@dataclass(frozen=True)
class BacktestMetrics:
    horizon: int
    phase: int
    cost_multiplier: float
    observations: int
    rebalance_count: int
    gross_total_return: float
    net_total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    transaction_cost: float
    average_cash_weight: float
    average_holding_count: float
    blocked_buys: int
    blocked_sells: int
    missing_valuation_intervals: int

    def to_dict(self) -> dict:
        return asdict(self)


def _performance_metrics(returns: pd.Series, annual_days: int) -> tuple[float, float, float, float]:
    values = returns.dropna().astype(float)
    if values.empty:
        return 0.0, 0.0, 0.0, 0.0
    total = float(np.prod(1.0 + values) - 1.0)
    years = len(values) / annual_days
    annual = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else -1.0
    std = float(values.std(ddof=0))
    sharpe = float(values.mean() / std * math.sqrt(annual_days)) if std > 1e-12 else 0.0
    equity = (1.0 + values).cumprod()
    drawdown = 1.0 - equity / equity.cummax()
    return total, annual, sharpe, float(drawdown.max())


class ReferenceBacktester:
    """Simulate the frozen SIMPLE_ENSEMBLE_V1 reference portfolio."""

    def __init__(self, config: Stage3Config | None = None) -> None:
        self.config = config or Stage3Config()

    def run(
        self,
        features: pd.DataFrame,
        daily_bars: pd.DataFrame,
        trading_status: pd.DataFrame,
        trading_dates: Iterable[pd.Timestamp],
        signal_dates: Iterable[pd.Timestamp],
        *,
        horizon: int,
        phase: int,
        cost_multiplier: float = 1.0,
    ) -> tuple[BacktestMetrics, pd.DataFrame]:
        if horizon not in self.config.horizons:
            raise ValueError(f"unsupported horizon {horizon}")
        if not 0 <= phase < horizon:
            raise ValueError("phase must be in [0, horizon)")
        dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize()
        dates = dates.drop_duplicates().sort_values()
        allowed_signals = set(pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize())
        feature = features.copy()
        feature["date"] = pd.to_datetime(feature["date"]).dt.normalize()
        feature_by_date = {date: block for date, block in feature.groupby("date", observed=True)}

        bars = daily_bars[["date", "code", "open_tr", "has_quote"]].copy()
        bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
        bars.loc[~bars["has_quote"].fillna(False).astype(bool), "open_tr"] = np.nan
        open_prices = bars.pivot(index="date", columns="code", values="open_tr").reindex(dates)
        simple_returns = open_prices.shift(-1) / open_prices - 1.0

        status = trading_status.copy()
        status["date"] = pd.to_datetime(status["date"]).dt.normalize()
        status = status.set_index(["date", "code"])
        targets: dict[pd.Timestamp, list[tuple[str, float]]] = {}
        date_position = {date: idx for idx, date in enumerate(dates)}
        for signal_date in sorted(allowed_signals):
            position = date_position.get(signal_date)
            if position is None or position % horizon != phase or position + 1 >= len(dates):
                continue
            block = feature_by_date.get(signal_date)
            if block is None:
                continue
            eligible = block.loc[
                block["valid_signal"].fillna(False).astype(bool)
                & block["median_amount_20"].ge(self.config.liquidity_median_amount_20d)
            ].sort_values(["simple_ensemble", "code"], ascending=[False, True], kind="stable")
            selected = eligible.head(self.config.top_n)
            targets[dates[position + 1]] = [
                (str(row.code), float(row.simple_ensemble))
                for row in selected.itertuples()
            ]

        if not allowed_signals:
            raise ValueError("signal_dates cannot be empty")
        start_signal = min(allowed_signals)
        end_signal = max(allowed_signals)
        active_dates = dates[(dates > start_signal) & (dates <= end_signal)]
        if len(active_dates) < 2:
            raise ValueError("validation window has no executable return interval")

        cash = 1.0
        holdings: dict[str, float] = {}
        gross_nav = 1.0
        previous_nav = 1.0
        total_cost = 0.0
        total_turnover = 0.0
        blocked_buys = blocked_sells = missing_valuation = rebalance_count = 0
        records: list[dict] = []

        def allowed(date: pd.Timestamp, code: str, column: str) -> bool:
            try:
                value = status.loc[(date, code), column]
                if isinstance(value, pd.Series):
                    return bool(value.iloc[0])
                return bool(value)
            except KeyError:
                return False

        for date in active_dates[:-1]:
            nav_before = cash + sum(holdings.values())
            day_cost = 0.0
            day_turnover = 0.0
            if date in targets:
                rebalance_count += 1
                ranked = targets[date]
                target_values = {
                    code: nav_before * self.config.target_weight for code, _ in ranked
                }
                for code in sorted(set(holdings) | set(target_values)):
                    current = holdings.get(code, 0.0)
                    desired = target_values.get(code, 0.0)
                    sell = max(0.0, current - desired)
                    if sell <= 1e-15:
                        continue
                    if not allowed(date, code, "can_sell_open"):
                        blocked_sells += 1
                        continue
                    fee = sell * self.config.sell_cost_rate * cost_multiplier
                    holdings[code] = current - sell
                    cash += sell - fee
                    day_cost += fee
                    day_turnover += sell
                for code, _score in ranked:
                    current = holdings.get(code, 0.0)
                    desired = target_values[code]
                    buy = max(0.0, desired - current)
                    if buy <= 1e-15:
                        continue
                    if not allowed(date, code, "can_buy_open"):
                        blocked_buys += 1
                        continue
                    rate = self.config.buy_cost_rate * cost_multiplier
                    affordable = cash / (1.0 + rate) if rate >= 0 else cash
                    executed = min(buy, max(0.0, affordable))
                    if executed <= 1e-15:
                        continue
                    fee = executed * rate
                    holdings[code] = current + executed
                    cash -= executed + fee
                    day_cost += fee
                    day_turnover += executed
                holdings = {code: value for code, value in holdings.items() if value > 1e-12}

            gross_day_gain = 0.0
            for code, value in list(holdings.items()):
                asset_return = simple_returns.at[date, code] if code in simple_returns else np.nan
                if not np.isfinite(asset_return):
                    # Explicit valuation carry during suspension/missing quote. This
                    # does not create an OHLCV row and is counted in the audit trail.
                    asset_return = 0.0
                    missing_valuation += 1
                gain = value * float(asset_return)
                holdings[code] = value + gain
                gross_day_gain += gain
            gross_nav += gross_day_gain
            nav_after = cash + sum(holdings.values())
            net_return = nav_after / previous_nav - 1.0 if previous_nav > 0 else 0.0
            gross_return = gross_day_gain / previous_nav if previous_nav > 0 else 0.0
            previous_nav = nav_after
            total_cost += day_cost
            total_turnover += day_turnover / nav_before if nav_before > 0 else 0.0
            records.append(
                {
                    "date": date,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "nav": nav_after,
                    "cash_weight": cash / nav_after if nav_after > 0 else 1.0,
                    "holding_count": len(holdings),
                    "turnover": day_turnover / nav_before if nav_before > 0 else 0.0,
                    "transaction_cost": day_cost,
                }
            )

        daily = pd.DataFrame(records)
        gross_total, _, _, _ = _performance_metrics(
            daily["gross_return"], self.config.annual_trading_days
        )
        net_total, annual, sharpe, max_drawdown = _performance_metrics(
            daily["net_return"], self.config.annual_trading_days
        )
        metrics = BacktestMetrics(
            horizon=horizon,
            phase=phase,
            cost_multiplier=float(cost_multiplier),
            observations=len(daily),
            rebalance_count=rebalance_count,
            gross_total_return=gross_total,
            net_total_return=net_total,
            annualized_return=annual,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            turnover=total_turnover,
            transaction_cost=total_cost,
            average_cash_weight=float(daily["cash_weight"].mean()) if len(daily) else 1.0,
            average_holding_count=float(daily["holding_count"].mean()) if len(daily) else 0.0,
            blocked_buys=blocked_buys,
            blocked_sells=blocked_sells,
            missing_valuation_intervals=missing_valuation,
        )
        return metrics, daily
