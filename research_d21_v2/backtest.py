"""Naive 5-day Top20 reuse plus frozen BUFFERED_TOP20_5D_V1."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from research_stage3.backtest import BacktestMetrics, ReferenceBacktester

from .protocol import D21V2Config


def _finite(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else float("nan")


class BufferedTop20Backtester:
    """Top20 entry / Top40 exit. Does not learn the buffer width."""

    def __init__(self, config: D21V2Config | None = None) -> None:
        self.config = config or D21V2Config()
        self._naive = ReferenceBacktester(self.config.as_stage3_config())

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
        score_column: str = "slow_residual_ensemble",
        valid_column: str = "valid_slow_signal",
        membership: pd.DataFrame | None = None,
    ) -> tuple[BacktestMetrics, pd.DataFrame]:
        if horizon != self.config.target_horizon:
            raise ValueError("D21-v2 buffered book only studies horizon 5")
        if not 0 <= phase < horizon:
            raise ValueError("phase must be in [0, horizon)")
        dates = self._naive._prepare_market(
            features, daily_bars, trading_status, trading_dates, membership
        )
        allowed_signals = set(pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize())
        if not allowed_signals:
            raise ValueError("signal_dates cannot be empty")
        start_signal = min(allowed_signals)
        end_signal = max(allowed_signals)
        active_dates = dates[(dates > start_signal) & (dates <= end_signal)]
        if len(active_dates) < 2:
            raise ValueError("validation window has no executable return interval")

        date_position = {date: idx for idx, date in enumerate(dates)}
        cash = 1.0
        holdings: dict[str, float] = {}
        last_open: dict[str, float] = {}
        gross_nav = 1.0
        previous_nav = 1.0
        total_cost = 0.0
        total_turnover = 0.0
        blocked_buys = blocked_sells = missing_valuation = rebalance_count = 0
        universe_exit_sells = universe_exit_pending = exit_pending = 0
        records: list[dict] = []
        cfg = self.config

        for date, next_date in zip(active_dates[:-1], active_dates[1:]):
            nav_before = cash + sum(holdings.values())
            day_cost = 0.0
            day_turnover = 0.0
            universe = self._naive._members_by_date.get(date, set())
            position = date_position[date]
            signal_date = dates[position - 1] if position > 0 else None
            rebalance = (
                signal_date is not None
                and signal_date in allowed_signals
                and (position - 1) % horizon == phase
            )
            ranks: dict[str, int] = {}
            if rebalance:
                rebalance_count += 1
                target_values, ranks = self._buffered_targets(
                    signal_date=signal_date,
                    universe=universe,
                    holdings=holdings,
                    nav_before=nav_before,
                    score_column=score_column,
                    valid_column=valid_column,
                )
            else:
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
                exiting = desired <= 1e-15
                if not self._naive._sellable(date, code):
                    blocked_sells += 1
                    if exiting:
                        exit_pending += 1
                    if code not in universe:
                        universe_exit_pending += 1
                    continue
                fee = sell * cfg.sell_cost_rate * cost_multiplier
                holdings[code] = current - sell
                cash += sell - fee
                day_cost += fee
                day_turnover += sell
                if code not in universe:
                    universe_exit_sells += 1
            if rebalance:
                ordered = sorted(
                    target_values,
                    key=lambda code: (ranks.get(code, 10**9), code),
                )
                actual_count = sum(1 for value in holdings.values() if value > 1e-12)
                allow_new = actual_count <= cfg.top_n
                for code in ordered:
                    current = holdings.get(code, 0.0)
                    desired = target_values[code]
                    buy = max(0.0, desired - current)
                    if buy <= 1e-15:
                        continue
                    is_new = current <= 1e-12
                    if is_new and not allow_new:
                        continue
                    if not self._naive._buyable(date, code):
                        blocked_buys += 1
                        continue
                    rate = cfg.buy_cost_rate * cost_multiplier
                    affordable = cash / (1.0 + rate) if rate >= 0 else cash
                    executed = min(buy, max(0.0, affordable))
                    if executed <= 1e-15:
                        continue
                    fee = executed * rate
                    holdings[code] = current + executed
                    cash -= executed + fee
                    day_cost += fee
                    day_turnover += executed
                    if is_new:
                        actual_count += 1
                        if actual_count > cfg.top_n:
                            allow_new = False
            holdings = {code: value for code, value in holdings.items() if value > 1e-12}

            gross_day_gain = 0.0
            for code, value in list(holdings.items()):
                px_today = self._naive._open_px(date, code)
                if np.isfinite(px_today) and px_today > 0:
                    last_open[code] = px_today
                px_next = self._naive._open_px(next_date, code)
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
                    "exit_pending": exit_pending,
                }
            )

        daily = pd.DataFrame(records)
        from research_stage3.backtest import _performance_metrics

        gross_total, _, _, _ = _performance_metrics(
            daily["gross_return"], cfg.annual_trading_days
        )
        net_total, annual, sharpe, max_drawdown = _performance_metrics(
            daily["net_return"], cfg.annual_trading_days
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
        daily.attrs["exit_pending"] = exit_pending
        return metrics, daily

    def _buffered_targets(
        self,
        *,
        signal_date: pd.Timestamp,
        universe: set[str],
        holdings: dict[str, float],
        nav_before: float,
        score_column: str,
        valid_column: str,
    ) -> tuple[dict[str, float], dict[str, int]]:
        cfg = self.config
        block = self._naive._feature_by_date.get(signal_date)
        if block is None or block.empty:
            return {}, {}
        if score_column not in block.columns:
            raise ValueError(f"features missing score column {score_column}")
        if valid_column not in block.columns:
            raise ValueError(f"features missing valid column {valid_column}")
        eligible = block.loc[block[valid_column].fillna(False).astype(bool)].copy()
        if eligible.empty:
            return {}, {}
        eligible = eligible.sort_values(
            [score_column, "code"], ascending=[False, True], kind="stable"
        )
        eligible["entry_rank"] = np.arange(1, len(eligible) + 1)
        ranks = {
            str(row.code): int(row.entry_rank) for row in eligible.itertuples()
        }
        held = [code for code, value in holdings.items() if value > 1e-12]
        keep: list[str] = []
        for code in held:
            if code not in universe:
                continue
            rank = ranks.get(code)
            if rank is None or rank > cfg.exit_rank:
                continue
            keep.append(code)

        target_names = list(keep)
        actual_count = len(held)
        if actual_count <= cfg.top_n:
            top_entry = eligible.loc[eligible["entry_rank"] <= cfg.top_n]
            for row in top_entry.itertuples():
                if len(target_names) >= cfg.top_n:
                    break
                code = str(row.code)
                if code in target_names or code not in universe:
                    continue
                median_amount = _finite(getattr(row, "median_amount_20", np.nan))
                if (
                    not np.isfinite(median_amount)
                    or median_amount < cfg.liquidity_median_amount_20d
                ):
                    continue
                target_names.append(code)
        scale = nav_before if nav_before > 0 else 1.0
        targets = {code: scale * cfg.target_weight for code in target_names}
        if len(targets) * cfg.target_weight > 1.0 + 1e-12:
            raise RuntimeError("buffered target weights exceed 100%")
        return targets, ranks
