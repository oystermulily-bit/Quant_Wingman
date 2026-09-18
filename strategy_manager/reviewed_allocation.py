"""Reviewed positive-target allocation, uncertainty ranges and execution checks."""
from __future__ import annotations

from copy import copy
import math

import numpy as np

from strategy_manager.portfolio_models import AllocationError, RULE_VERSION, require
from strategy_manager.portfolio_optimization import PortfolioProblem, transaction_cost, STOCK_CAP, INDUSTRY_CAP, TOL
from strategy_manager.portfolio_explanations import conditional_weight_ranges, allocation_diagnostics


def _execution(problem, target, no_trade, *, reference=False):
    p = problem
    n = len(p.codes)
    if not p.known_holdings:
        return None, None, ["UNKNOWN"]*n, ["MISSING_HOLDINGS_TARGET_ONLY"], None
    changed = np.abs(target-p.current) > 1e-9
    permissions = [p.permission(c, "can_buy" if target[i] > p.current[i] else "can_sell")
                   for i, c in enumerate(p.codes)]
    status = ["NO_CHANGE" if not changed[i] else "UNKNOWN" if permissions[i] is None
              else "READY" if permissions[i] else "BLOCKED_BUY" if target[i] > p.current[i]
              else "BLOCKED_SELL" for i in range(n)]
    if no_trade:
        return p.current.copy(), 0., ["NO_TRADE"]*n, ["NO_TRADE_COST_OR_UTILITY"], None
    if any(changed[i] and permissions[i] is None for i in range(n)):
        return None, None, status, ["EXECUTION_INFORMATION_MISSING"], None
    lower, upper = p.base_lower.copy(), p.base_upper.copy()
    reasons = []
    for i, code in enumerate(p.codes):
        buy, sell = p.permission(code, "can_buy"), p.permission(code, "can_sell")
        if sell is not True:
            lower[i] = max(lower[i], p.current[i])
            upper[i] = max(upper[i], p.current[i])
        if buy is not True:
            upper[i] = min(upper[i], p.current[i])
        info = p.market.get(code, {})
        median = info.get("median_amount_20d")
        if median is None:
            upper[i] = min(upper[i], p.current[i])
            if target[i] > p.current[i]+TOL:
                status[i] = "LIQUIDITY_INFORMATION_MISSING"
        if median is not None and median < 20_000_000:
            upper[i] = min(upper[i], p.current[i])
            if target[i] > p.current[i]+TOL:
                status[i] = "BLOCKED_LIQUIDITY"
        if p.account is not None and info.get("mean_amount_20d") is not None:
            capacity = .01*info["mean_amount_20d"]/p.account
            lower[i] = max(lower[i], p.current[i]-capacity)
            upper[i] = min(upper[i], p.current[i]+capacity)
        elif p.account is not None and target[i] > p.current[i]+TOL:
            upper[i] = min(upper[i], p.current[i])
            status[i] = "LIQUIDITY_INFORMATION_MISSING"
    try:
        if p.feasible(target, lower=lower, upper=upper):
            projected = target.copy()
        else:
            projected, _ = p.solve(lower=lower, upper=upper, reference=target if reference else None)
        for i in range(n):
            if status[i] == "READY" and abs(projected[i]-target[i]) > 1e-7:
                status[i] = "PARTIAL_RISK_OR_CAPACITY_LIMIT"
    except AllocationError as exc:
        # Keep real frozen holdings; only simulate available reductions. Never
        # erase them or spend their hypothetical sale proceeds on new buys.
        projected = p.current.copy()
        for i in range(n):
            if target[i] < p.current[i] and permissions[i] is True:
                projected[i] = max(target[i], lower[i])
            elif target[i] > p.current[i] and permissions[i] is True:
                status[i] = "BUY_PAUSED_INFEASIBLE_EXECUTION"
        reasons.extend(["EXECUTION_CONSTRAINT_INFEASIBLE", exc.code])
    orders = None
    if p.account is not None:
        ready_for_lots = all(p.market.get(c, {}).get("price", 0) > 0
                             and p.market.get(c, {}).get("lot_size", 0) > 0
                             and (p.current[i] == 0 or p.holdings.get(c, {}).get("quantity") is not None)
                             for i, c in enumerate(p.codes))
        if ready_for_lots:
            orders = []
            for i, code in enumerate(p.codes):
                info = p.market[code]
                price, lot = info["price"], info["lot_size"]
                quantity = p.holdings.get(code, {}).get("quantity")
                if quantity is None and p.current[i] == 0:
                    quantity = 0
                require(type(lot) is int and lot > 0 and type(quantity) is int and quantity >= 0,
                        "INVALID_TRADE_UNIT", "股票交易单位或实际持股数量无效")
                require(abs(quantity*price/p.account-p.current[i]) < 1e-7,
                        "HOLDING_VALUATION_MISMATCH", "当前股数、参考价格、账户净值与持仓权重不一致", code=code)
                delta = projected[i]-p.current[i]
                shares = math.floor((abs(delta)*p.account/price+1e-8)/lot)*lot
                if delta < 0 and projected[i] <= TOL and p.permission(code, "can_sell") is True:
                    shares = quantity  # full exit may include a remaining odd lot
                if delta < 0:
                    shares = min(shares, info.get("sellable_quantity", quantity))
                signed = shares if delta >= 0 else -shares
                projected[i] = p.current[i]+signed*price/p.account
                if abs(projected[i]-target[i]) > TOL and status[i] == "READY":
                    status[i] = "PARTIAL_LOT_OR_SELLABLE_LIMIT"
                orders.append({"code": code, "delta_quantity": signed, "reference_price": price,
                               "projected_quantity": quantity+signed, "lot_size": lot,
                               "price_available_at": info["available_at"], "actual_order": False})
            if not p.feasible(projected, lower=np.minimum(lower, p.current), upper=np.maximum(upper, p.current)):
                # Rounding may retain more old risk than the continuous solve.
                # Drop proposed buys in a fixed order until the risk checks pass.
                for i in reversed(range(n)):
                    if projected[i] > p.current[i]+TOL:
                        projected[i] = p.current[i]
                        orders[i]["delta_quantity"] = 0
                        orders[i]["projected_quantity"] = p.holdings.get(p.codes[i], {}).get("quantity", 0)
                        status[i] = "BUY_PAUSED_ROUNDING_RISK"
                        if p.feasible(projected, lower=np.minimum(lower, p.current), upper=np.maximum(upper, p.current)):
                            break
                reasons.append("ROUNDING_RECHECK_REQUIRED")
        else:
            reasons.append("SHARE_QUANTITIES_UNAVAILABLE_WEIGHT_SIMULATION_ONLY")
    fees = transaction_cost(projected-p.current, p.costs, p.account)["total"]
    require(projected.sum()+fees <= 1+TOL, "EXECUTION_CASH_INFEASIBLE", "按实际模拟成交重算费用后资金不足")
    if any(projected[p.index[c]] < p.minimum-TOL for c in p.selected):
        reasons.append("SELECTED_POSITIVE_HOLDINGS_NOT_EXECUTABLE")
    if any(status[i] == "BLOCKED_SELL" for i in range(n)):
        reasons.append("EXECUTION_BLOCKED_OLD_HOLDINGS_RETAINED")
    if not p.feasible(projected, lower=np.zeros(n), upper=np.maximum(p.base_upper, p.current)):
        reasons.append("RETAINED_HOLDINGS_EXCEED_RISK_LIMITS")
    return projected, fees, status, reasons, orders


def build_reviewed_plan(snapshot, whitelist, request, analysis=None):
    require(snapshot.get("production_allowed") is False, "NONPRODUCTION_INPUT_REQUIRED", "本接口仅允许离线数据")
    p = PortfolioProblem(snapshot, whitelist, request, analysis)
    target, solver = p.solve()
    reasons = ["SELECTED_TARGETS_STRICTLY_POSITIVE", "RANKING_SCORE_DIFFERS_FROM_PORTFOLIO_UTILITY"]
    if p.account is None:
        reasons.append("NO_ACCOUNT_MINIMUM_COMMISSION_LOTS_AND_CAPACITY_UNCHECKED")
    if "UNKNOWN" in p.industries:
        reasons.append("UNKNOWN_INDUSTRY_SHARED_30_PERCENT_CAP")
    fixed = {}
    if p.known_holdings and p.deadband:
        fixed = {c: float(p.current[i]) for i, c in enumerate(p.codes)
                 if c in p.selected and p.minimum <= p.current[i] <= STOCK_CAP
                 and abs(target[i]-p.current[i]) < p.deadband}
        if fixed:
            try:
                target, solver = p.solve(fixed=fixed)
                reasons.append("SMALL_REBALANCES_FROZEN")
            except AllocationError:
                fixed = {}
                reasons.append("DEADBAND_SKIPPED_FOR_HARD_CONSTRAINTS")
    metrics = p.metrics(target)
    baseline = p.metrics(p.current) if p.known_holdings else None
    # A mandatory exit is a constraint, not an economic choice to keep an
    # unselected holding indefinitely because exiting costs money.
    forced_exit = p.known_holdings and any(p.current[i] > 0 and c not in p.selected for i, c in enumerate(p.codes))
    forced_risk = p.known_holdings and not p.feasible(p.current, lower=np.zeros(len(p.codes)))
    no_trade = bool(baseline and not forced_exit and not forced_risk and metrics["utility"] <= baseline["utility"]+1e-10)
    if no_trade and p.feasible(p.current):
        target = p.current.copy()
        metrics = p.metrics(target)
    low, high, interval_statuses = conditional_weight_ranges(p, target, fixed=fixed)
    diagnostics = allocation_diagnostics(p, target, fixed)
    projected, simulated_fee, statuses, execution_reasons, orders = _execution(p, target, no_trade)
    reasons.extend(execution_reasons)
    positive_execution = projected is not None and all(projected[p.index[c]] >= p.minimum-TOL for c in p.selected)
    if projected is not None and not positive_execution:
        reasons.append("REQUEST_NOT_FULLY_EXECUTED")
    selected_valid = all(target[p.index[c]] >= p.minimum-TOL and target[p.index[c]] > 0 for c in p.selected)
    require(selected_valid and p.feasible(target), "POST_SOLVE_CHECK_FAILED", "结果未通过正目标与约束复核")
    details = metrics["cost_details"]
    cash = float(1-target.sum())
    rows = {r["code"]: r for r in snapshot["rows"]}
    positions = []
    for i, code in enumerate(p.codes):
        delta = float(target[i]-p.current[i]) if p.known_holdings else None
        selected = code in p.selected
        row_reasons = []
        if not selected:
            row_reasons.append("NOT_IN_WHITELIST_EXIT_TARGET")
        if selected and abs(target[i]-p.minimum) < 1e-7:
            row_reasons.append("AT_USER_MINIMUM_WEIGHT")
        if selected and abs(target[i]-STOCK_CAP) < 1e-7:
            row_reasons.append("AT_STOCK_CAP")
        positions.append({"code": code, "score": rows.get(code, {}).get("score"), "selected": selected,
                          "industry": p.industries[i], "expected_return": float(p.mu[i]) if selected else None,
                          "return_standard_error": float(p.se[i]) if selected else None,
                          "current_weight": float(p.current[i]) if p.known_holdings else None,
                          "target_weight": float(target[i]), "weight_lower": float(max(0., low[i])),
                          "weight_upper": float(high[i]), "interval_status": interval_statuses[i], "delta_weight": delta,
                          "action": "TARGET_ONLY" if delta is None else "HOLD" if abs(delta) < 1e-9 else "BUY" if delta > 0 else "SELL",
                          "execution_status": statuses[i], "can_buy": p.permission(code, "can_buy"),
                          "can_sell": p.permission(code, "can_sell"),
                          "executable_delta_weight": float(projected[i]-p.current[i]) if projected is not None else None,
                          "projected_weight": float(projected[i]) if projected is not None else None,
                          "target_value": float(target[i]*p.account) if p.account is not None else None,
                          "delta_value": delta*p.account if delta is not None and p.account is not None else None,
                          "reason_codes": row_reasons})
    positions.sort(key=lambda r: (rows.get(r["code"], {}).get("rank") is None,
                                  rows.get(r["code"], {}).get("rank") or 0, r["code"]))
    sector = {g: float(target[ids].sum()) for g, ids in p.groups.items()}
    largest = max(sector, key=sector.get)
    blocked_stress = {}
    for blocked_side in ("can_buy", "can_sell"):
        scenario = copy(p)
        scenario.request = {**p.request, "tradability": {
            c: {**(p.request.get("tradability") or {}).get(c, {}), blocked_side: False} for c in p.codes}}
        try:
            holdings, fees, _, notes, _ = _execution(scenario, target, no_trade)
            blocked_stress["all_buys_blocked" if blocked_side == "can_buy" else "all_sells_blocked"] = {
                "projected_weights": dict(zip(p.codes, holdings.tolist())) if holdings is not None else None,
                "cash_weight": float(1-holdings.sum()-fees) if holdings is not None else None,
                "cost": fees, "annual_volatility": p.metrics(holdings)["expected_volatility"] if holdings is not None else None,
                "reason_codes": notes}
        except AllocationError as exc:
            blocked_stress["all_buys_blocked" if blocked_side == "can_buy" else "all_sells_blocked"] = {
                "projected_weights": None, "cash_weight": None, "cost": None, "annual_volatility": None,
                "reason_codes": [exc.code]}
    status = ("NO_TRADE" if no_trade and positive_execution else "NO_TRADE_REQUEST_UNSATISFIED" if no_trade
              else "EXECUTION_INCOMPLETE" if projected is not None and not positive_execution else "OFFLINE_OPTIMIZED_NOT_VALIDATED")
    return {"schema_version": "offline_portfolio_plan_v2", "rule_version": RULE_VERSION,
            "status": status, "method": "ROBUST_MEAN_VARIANCE_COST", "solver": solver,
            "weight_basis": "PRE_TRADE_ACCOUNT_EQUITY", "score_usage": "CALIBRATED_RETURN_INPUT_NOT_DIRECT_WEIGHT",
            "data_role": snapshot["data_role"], "snapshot_id": snapshot["snapshot_id"],
            "date": snapshot["date"], "fold_id": snapshot["fold_id"], "horizon": p.horizon,
            "analysis_id": analysis["analysis_id"], "analysis_source_sha256": analysis["source_sha256"],
            "calibration": analysis["calibrator"], "risk_method": analysis["risk_method"],
            "whitelist_id": whitelist["whitelist_id"], "whitelist_version": whitelist["version"],
            "production_allowed": False, "research_go": False, "holdout_read": False,
            "sota_promoted": False, "orders_submitted": False, "selected_positive_targets": selected_valid,
            "selected_positive_projected_holdings": positive_execution,
            "minimum_weight": p.minimum, "market_budget": p.budget, "volatility_cap": p.volatility_cap,
            "risk_aversion": p.risk_aversion, "stock_cap": STOCK_CAP, "industry_cap": INDUSTRY_CAP,
            "stock_weight": float(target.sum()), "cash_weight": cash,
            "cash_after_estimated_cost": max(0., cash-details["total"]) if details else None,
            "estimated_turnover": details["turnover"] if details else None,
            "estimated_cost": details["total"] if details else None,
            "estimated_buy_weight": details["buy_weight"] if details else None,
            "estimated_sell_weight": details["sell_weight"] if details else None,
            "buy_cost_rate": p.costs["buy_rate"], "sell_cost_rate": p.costs["sell_rate"], "cost_model": p.costs,
            "account_value": p.account, "holdings_provided": p.known_holdings,
            "assume_tradable": request.get("assume_tradable", False),
            "budget_mode": "FIXED_USER_BUDGET_NO_MARKET_TIMING", "no_trade_band": p.deadband,
            "projected_cash_weight": max(0., float(1-projected.sum()-simulated_fee)) if projected is not None else None,
            "simulated_execution_cost": simulated_fee, "orders_preview": orders,
            "objective": metrics, "hold_current_objective": baseline, "industry_weights": sector,
            "allocation_diagnostics": diagnostics,
            "interval_method": "CONDITIONAL_CASH_UTILITY_TOLERANCE_NOT_CONFIDENCE_INTERVAL",
            "range_utility_tolerance_bps": p.range_tolerance_bps,
            "interval_reference": "OTHER_STOCKS_FIXED_AT_TARGET_CASH_ABSORBS_CHANGE",
            "interval_respects_minimum_weight": True, "interval_jointly_selectable": False,
            "stress_checks": {"cost_1_5x": details["total"]*1.5 if details else None,
                              "target_cash_after_cost_1_5x": cash-details["total"]*1.5 if details else None,
                              "opening_round_trip_cost": float(sum(max(target[i]-p.current[i], 0.) for i in range(len(p.codes)))
                                                                *(p.costs["buy_rate"]+p.costs["sell_rate"])) if p.known_holdings else None,
                              "opening_round_trip_cost_basis": "PROPORTIONAL_RATES_EXCLUDING_FUTURE_MINIMUM_COMMISSION",
                              "largest_industry": largest, "largest_industry_down_10pct_loss": sector[largest]*.1,
                              **blocked_stress},
            "review": {"positive_targets_passed": True, "hard_constraints_passed": True,
                       "strategy_backtest_passed": False, "forward_tracking_available": True},
            "positions": positions, "reason_codes": list(dict.fromkeys(reasons)),
            "warnings": ["离线优化结果；工程复核不代表策略已通过真实市场回测。",
                         "排名分数评价股票；目标效用比较整套权重，二者不是同一得分。",
                         "建议区间固定其余目标、以现金吸收变化，限定效用损失；遵守最低目标与风险约束，不是置信区间，不能任意混合各股区间。",
                         "推荐目标均为正；执行受阻、未交易和实际零持仓会另行标记。"]}
