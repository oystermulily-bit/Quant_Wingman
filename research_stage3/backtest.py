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
    universe_exit_sells: int
    universe_exit_pending: int

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
        self._market_key: tuple | None = None
        self._dates: pd.DatetimeIndex | None = None
        self._feature_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        self._simple_returns: pd.DataFrame | None = None
        self._open_prices: pd.DataFrame | None = None
        self._can_buy: dict[tuple[pd.Timestamp, str], bool] = {}
        self._can_sell: dict[tuple[pd.Timestamp, str], bool] = {}
        self._members_by_date: dict[pd.Timestamp, set[str]] = {}
        self._target_key: tuple | None = None
        self._targets: dict[pd.Timestamp, list[tuple[str, float]]] = {}

    def _prepare_market(
        self,
        features: pd.DataFrame,
        daily_bars: pd.DataFrame,
        trading_status: pd.DataFrame,
        trading_dates: Iterable[pd.Timestamp],
        membership: pd.DataFrame | None = None,
    ) -> pd.DatetimeIndex:
        dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize()
        dates = dates.drop_duplicates().sort_values()
        key = (
            id(features),
            id(daily_bars),
            id(trading_status),
            id(membership) if membership is not None else None,
            dates[0],
            dates[-1],
            len(dates),
        )
        if key == self._market_key and self._open_prices is not None:
            return dates

        feature = features.copy()
        feature["date"] = pd.to_datetime(feature["date"]).dt.normalize()
        feature["code"] = feature["code"].astype(str)
        self._feature_by_date = {
            date: block for date, block in feature.groupby("date", observed=True)
        }

        bars = daily_bars[["date", "code", "open_tr", "has_quote"]].copy()
        bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
        bars["code"] = bars["code"].astype(str)
        bars.loc[~bars["has_quote"].fillna(False).astype(bool), "open_tr"] = np.nan
        open_prices = bars.pivot(index="date", columns="code", values="open_tr").reindex(dates)
        self._open_prices = open_prices
        self._simple_returns = open_prices.shift(-1) / open_prices - 1.0

        status = trading_status.copy()
        status["date"] = pd.to_datetime(status["date"]).dt.normalize()
        status["code"] = status["code"].astype(str)
        status = status.set_index(["date", "code"])
        self._can_buy = {
            key: bool(value) for key, value in status["can_buy_open"].items()
        }
        self._can_sell = {
            key: bool(value) for key, value in status["can_sell_open"].items()
        }
        if membership is not None and len(membership):
            mem = membership.copy()
            mem["date"] = pd.to_datetime(mem["date"]).dt.normalize()
            mem["code"] = mem["code"].astype(str)
            if "is_member" in mem.columns:
                mem = mem.loc[mem["is_member"].fillna(False).astype(bool)]
            self._members_by_date = {
                date: set(block["code"].astype(str))
                for date, block in mem.groupby("date", observed=True)
            }
        else:
            self._members_by_date = {
                date: set(block["code"].astype(str))
                for date, block in self._feature_by_date.items()
            }
        self._dates = dates
        self._market_key = key
        self._target_key = None
        return dates

    def _open_px(self, date: pd.Timestamp, code: str) -> float:
        prices = self._open_prices
        if prices is None or code not in prices.columns or date not in prices.index:
            return float("nan")
        value = prices.at[date, code]
        return float(value) if np.isfinite(value) else float("nan")

    def _sellable(self, date: pd.Timestamp, code: str) -> bool:
        key = (date, code)
        if key in self._can_sell:
            return self._can_sell[key]
        px = self._open_px(date, code)
        return bool(np.isfinite(px) and px > 0)

    def _buyable(self, date: pd.Timestamp, code: str) -> bool:
        return self._can_buy.get((date, code), False)

    def _prepare_targets(
        self,
        dates: pd.DatetimeIndex,
        allowed_signals: set[pd.Timestamp],
        *,
        horizon: int,
        phase: int,
        score_column: str,
    ) -> dict[pd.Timestamp, list[tuple[str, float]]]:
        signal_key = (horizon, phase, score_column, frozenset(allowed_signals))
        if signal_key == self._target_key:
            return self._targets
        targets: dict[pd.Timestamp, list[tuple[str, float]]] = {}
        date_position = {date: idx for idx, date in enumerate(dates)}
        for signal_date in sorted(allowed_signals):
            position = date_position.get(signal_date)
            if position is None or position % horizon != phase or position + 1 >= len(dates):
                continue
            block = self._feature_by_date.get(signal_date)
            if block is None:
                continue
            if score_column not in block.columns:
                raise ValueError(f"features missing score column {score_column}")
            eligible = block.loc[
                block["valid_signal"].fillna(False).astype(bool)
                & block["median_amount_20"].ge(self.config.liquidity_median_amount_20d)
            ].sort_values([score_column, "code"], ascending=[False, True], kind="stable")
            selected = eligible.head(self.config.top_n)
            targets[dates[position + 1]] = [
                (str(row.code), float(getattr(row, score_column)))
                for row in selected.itertuples()
            ]
        self._targets = targets
        self._target_key = signal_key
        return targets

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
        score_column: str = "simple_ensemble",
        membership: pd.DataFrame | None = None,
    ) -> tuple[BacktestMetrics, pd.DataFrame]:
        if horizon not in self.config.horizons:
            raise ValueError(f"unsupported horizon {horizon}")
        if not 0 <= phase < horizon:
            raise ValueError("phase must be in [0, horizon)")
        dates = self._prepare_market(
            features, daily_bars, trading_status, trading_dates, membership
        )
        allowed_signals = set(pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize())
        targets = self._prepare_targets(
            dates,
            allowed_signals,
            horizon=horizon,
            phase=phase,
            score_column=score_column,
        )
        open_prices = self._open_prices
        assert open_prices is not None

        if not allowed_signals:
            raise ValueError("signal_dates cannot be empty")
        start_signal = min(allowed_signals)
        end_signal = max(allowed_signals)
        active_dates = dates[(dates > start_signal) & (dates <= end_signal)]
        if len(active_dates) < 2:
            raise ValueError("validation window has no executable return interval")

        cash = 1.0
        holdings: dict[str, float] = {}
        last_open: dict[str, float] = {}
        gross_nav = 1.0
        previous_nav = 1.0
        total_cost = 0.0
        total_turnover = 0.0
        blocked_buys = blocked_sells = missing_valuation = rebalance_count = 0
        universe_exit_sells = universe_exit_pending = 0
        records: list[dict] = []

        for date, next_date in zip(active_dates[:-1], active_dates[1:]):
            nav_before = cash + sum(holdings.values())
            day_cost = 0.0
            day_turnover = 0.0
            universe = self._members_by_date.get(date, set())
            rebalance = date in targets
            if rebalance:
                rebalance_count += 1
                ranked = [(code, score) for code, score in targets[date] if code in universe]
                target_values = {
                    code: nav_before * self.config.target_weight for code, _ in ranked
                }
            else:
                ranked = []
                target_values = {
                    code: value
                    for code, value in holdings.items()
                    if code in universe
                }

            for code in sorted(set(holdings) | set(target_values)):
                current = holdings.get(code, 0.0)
                desired = target_values.get(code, 0.0)
                sell = max(0.0, current - desired)
                if sell <= 1e-15:
                    continue
                exiting = code not in universe
                if not self._sellable(date, code):
                    blocked_sells += 1
                    if exiting:
                        universe_exit_pending += 1
                    continue
                fee = sell * self.config.sell_cost_rate * cost_multiplier
                holdings[code] = current - sell
                cash += sell - fee
                day_cost += fee
                day_turnover += sell
                if exiting:
                    universe_exit_sells += 1
            if rebalance:
                for code, _score in ranked:
                    current = holdings.get(code, 0.0)
                    desired = target_values[code]
                    buy = max(0.0, desired - current)
                    if buy <= 1e-15:
                        continue
                    if not self._buyable(date, code):
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
                px_today = self._open_px(date, code)
                if np.isfinite(px_today) and px_today > 0:
                    last_open[code] = px_today
                px_next = self._open_px(next_date, code)
                base = last_open.get(code)
                if (
                    base is not None
                    and base > 0
                    and np.isfinite(px_next)
                    and px_next > 0
                ):
                    asset_return = px_next / base - 1.0
                    last_open[code] = px_next
                else:
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
            universe_exit_sells=universe_exit_sells,
            universe_exit_pending=universe_exit_pending,
        )
        return metrics, daily

