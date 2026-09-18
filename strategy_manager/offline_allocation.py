"""Deterministic account-weight simulations; no pricing, prediction or orders.

Scores determine the displayed ranking, not predicted returns or learned weights.
All amounts use pre-trade account equity. Fees are reserved within cash; projected
holdings + projected cash + simulated fees = 1 when execution inputs are known.
"""
from __future__ import annotations

import math
from typing import Any

STOCK_CAP = 0.05
INDUSTRY_CAP = 0.30
EPS = 1e-10


def _number(value, name, minimum=0.0, maximum=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} outside permitted range") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} outside permitted range")
    return value


def _permission(value):
    if value is not None and type(value) is not bool:
        raise ValueError("tradability must be boolean or null")
    return value


def _equal_weights(codes, industries, budget):
    """Progressive equal allocation under stock/industry caps; unused stays cash."""
    weights = dict.fromkeys(sorted(codes), 0.0)
    remaining = budget
    while remaining > EPS:
        sector = {}
        for code, weight in weights.items():
            group = industries[code]
            sector[group] = sector.get(group, 0.0) + weight
        active = [code for code in weights if weights[code] < STOCK_CAP-EPS
                  and sector[industries[code]] < INDUSTRY_CAP-EPS]
        if not active:
            break
        counts = {}
        for code in active:
            group = industries[code]
            counts[group] = counts.get(group, 0) + 1
        increment = min(remaining/len(active),
                        min(STOCK_CAP-weights[c] for c in active),
                        min((INDUSTRY_CAP-sector[g])/count for g, count in counts.items()))
        if increment <= EPS:
            break
        for code in active:
            weights[code] += increment
        remaining -= increment*len(active)
    return weights


def build_reference_plan(snapshot: dict, whitelist: dict, request: dict) -> dict[str, Any]:
    """Build a transparent offline target and conditional rebalance simulation."""
    if snapshot.get("production_allowed") is not False:
        raise ValueError("only explicitly non-production snapshots are supported")
    if snapshot.get("data_role") not in ("synthetic", "development"):
        raise ValueError("invalid offline data role")
    if whitelist.get("snapshot_id") != snapshot.get("snapshot_id"):
        raise ValueError("whitelist must reference this immutable score snapshot")
    rows = snapshot.get("rows", [])
    members = {row["code"]: row for row in rows}
    if len(members) != len(rows) or not members:
        raise ValueError("invalid score snapshot membership")
    selected = whitelist.get("codes", [])
    if (not isinstance(selected, list) or not selected or
            any(not isinstance(c, str) or c not in members for c in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("select unique stocks from the saved snapshot first")
    budget = _number(request.get("market_budget", 1.0), "market_budget")
    buy_cost = _number(request.get("buy_cost", .00076), "buy_cost", maximum=.1)
    sell_cost = _number(request.get("sell_cost", .00126), "sell_cost", maximum=.1)
    account = request.get("account_value")
    if account is not None:
        account = _number(account, "account_value", minimum=1e-8, maximum=1e16)
    assume = request.get("assume_tradable", False)
    if type(assume) is not bool:
        raise ValueError("assume_tradable must be boolean")
    supplied_holdings = request.get("holdings")
    known_holdings = supplied_holdings is not None
    if known_holdings and (not isinstance(supplied_holdings, list) or len(supplied_holdings) > 1000):
        raise ValueError("holdings must be a list or null")
    holdings = {}
    for item in supplied_holdings or []:
        if not isinstance(item, dict):
            raise ValueError("invalid holding")
        code = item.get("code")
        if not isinstance(code, str) or not code.strip() or len(code) > 80 or code in holdings:
            raise ValueError("holding codes must be nonempty and unique")
        weight = _number(item.get("current_weight"), "current_weight")
        holdings[code] = {**item, "current_weight": weight}
        for field in ("can_buy", "can_sell"):
            _permission(item.get(field))
    current = {c: h["current_weight"] for c, h in holdings.items()}
    if sum(current.values()) > 1+EPS:
        raise ValueError("current stock weights exceed account equity")
    industry_input = request.get("industries") or {}
    tradability = request.get("tradability") or {}
    if not isinstance(industry_input, dict) or not isinstance(tradability, dict):
        raise ValueError("industries and tradability must be mappings")
    universe = set(selected) | set(holdings)
    if not set(industry_input).issubset(set(members) | set(holdings)):
        raise ValueError("industry mapping contains unknown stock")
    if not set(tradability).issubset(set(members) | set(holdings)):
        raise ValueError("tradability mapping contains unknown stock")
    for flags in tradability.values():
        if not isinstance(flags, dict) or set(flags)-{"can_buy", "can_sell"}:
            raise ValueError("invalid tradability fields")
        for value in flags.values():
            _permission(value)
    industries = {}
    for code in universe:
        group = industry_input.get(code) or holdings.get(code, {}).get("industry") or "UNKNOWN"
        if not isinstance(group, str) or len(group) > 100:
            raise ValueError("invalid industry label")
        industries[code] = group.strip() or "UNKNOWN"
    eligible = []
    for code in selected:
        row = members[code]
        score = row.get("score")
        if (row.get("signal_valid") is True and not isinstance(score, bool)
                and isinstance(score, (int, float)) and math.isfinite(score)):
            eligible.append(code)

    def costs(weights):
        buys = sum(max(weights.get(c, 0)-current.get(c, 0), 0) for c in universe)
        sells = sum(max(current.get(c, 0)-weights.get(c, 0), 0) for c in universe)
        return buys, sells, buys*buy_cost+sells*sell_cost

    target = _equal_weights(eligible, industries, budget)
    reasons = ["OFFLINE_SIMULATION_ONLY", "SCORES_ARE_NOT_EXPECTED_RETURNS"]
    if any(industries[c] == "UNKNOWN" for c in universe):
        reasons.append("UNKNOWN_INDUSTRY_SHARED_30_PERCENT_CAP")
    if not eligible:
        reasons.append("NO_VALID_SELECTED_SCORE")
    if not known_holdings:
        reasons.append("MISSING_HOLDINGS_TARGET_ONLY")
    elif sum(target.values()) + costs(target)[2] > 1+EPS:
        # Find a feasible funded target, rather than spending 100% then charging
        # fees to negative cash. This does not learn a return-maximizing portfolio.
        low, high = 0.0, budget
        for _ in range(70):
            mid = (low+high)/2
            trial = _equal_weights(eligible, industries, mid)
            if sum(trial.values())+costs(trial)[2] <= 1:
                low = mid
            else:
                high = mid
        target = _equal_weights(eligible, industries, low)
        reasons.append("CASH_RESERVED_FOR_ESTIMATED_COST")
    stock_weight = sum(target.values())
    cash_weight = max(0.0, 1-stock_weight)
    buys, sells, cost = costs(target) if known_holdings else (None, None, None)
    if stock_weight < budget-EPS:
        reasons.append("UNUSED_BUDGET_REMAINS_CASH")

    def permission(code, field):
        # Explicit blocks always win over a convenient all-tradable assumption.
        options = [holdings.get(code, {}).get(field), tradability.get(code, {}).get(field)]
        if False in options:
            return False
        if True in options:
            return True
        return True if assume else None

    positions = []
    ordered = sorted(universe, key=lambda c: (members.get(c, {}).get("rank") is None,
                                              members.get(c, {}).get("rank") or 0, c))
    for code in ordered:
        weight = target.get(code, 0.0)
        held = current.get(code, 0.0) if known_holdings else None
        delta = weight-held if held is not None else None
        action = "TARGET_ONLY" if delta is None else (
            "BUY" if delta > EPS else "SELL" if delta < -EPS else "HOLD")
        flags = []
        if code not in selected:
            flags.append("NOT_IN_WHITELIST_EXIT_TARGET")
        elif code not in eligible:
            flags.append("INVALID_SCORE_NO_POSITIVE_TARGET")
        can_buy, can_sell = permission(code, "can_buy"), permission(code, "can_sell")
        allowed = can_buy if action == "BUY" else can_sell
        execution = ("UNKNOWN" if delta is None else "NO_CHANGE" if action == "HOLD" else
                     "UNKNOWN" if allowed is None else "READY" if allowed else "BLOCKED_"+action)
        positions.append({"code": code, "score": members.get(code, {}).get("score"),
                          "industry": industries[code], "selected": code in selected,
                          "current_weight": held, "target_weight": weight, "delta_weight": delta,
                          "action": action, "execution_status": execution,
                          "can_buy": can_buy, "can_sell": can_sell,
                          "executable_delta_weight": None, "projected_weight": None,
                          "target_value": weight*account if account is not None else None,
                          "delta_value": delta*account if delta is not None and account is not None else None,
                          "reason_codes": flags})
    projected_cash = actual_fee = None
    if known_holdings and not any(p["execution_status"] == "UNKNOWN" for p in positions):
        available_cash = max(0.0, 1-sum(current.values()))
        actual_fee = 0.0
        for p in positions:
            p["executable_delta_weight"] = 0.0
            p["projected_weight"] = p["current_weight"]
            if p["action"] == "SELL" and p["execution_status"] == "READY":
                amount = -p["delta_weight"]
                p["executable_delta_weight"] = -amount
                p["projected_weight"] -= amount
                available_cash += amount*(1-sell_cost)
                actual_fee += amount*sell_cost
        ready_buys = [p for p in positions if p["action"] == "BUY" and p["execution_status"] == "READY"]
        # Retained/blocked holdings consume the SAME risk budget as new buys.
        # Do not knowingly create a second exposure on top of a frozen position.
        retained_sector = {}
        for p in positions:
            retained_sector[p["industry"]] = retained_sector.get(p["industry"], 0)+p["projected_weight"]
        requested = {p["code"]: min(p["delta_weight"], max(0.0, STOCK_CAP-p["projected_weight"]))
                     for p in ready_buys}
        for sector in {p["industry"] for p in ready_buys}:
            group = [p for p in ready_buys if p["industry"] == sector]
            amount = sum(requested[p["code"]] for p in group)
            room = max(0.0, INDUSTRY_CAP-retained_sector.get(sector, 0))
            ratio = min(1.0, room/amount) if amount > EPS else 1.0
            for p in group:
                requested[p["code"]] *= ratio
        total = sum(requested.values())
        total_room = max(0.0, budget-sum(p["projected_weight"] for p in positions))
        risk_fraction = min(1.0, total_room/total) if total > EPS else 1.0
        required = total*risk_fraction*(1+buy_cost)
        fraction = min(1.0, available_cash/required) if required > EPS else 1.0
        for p in ready_buys:
            risk_amount = requested[p["code"]]*risk_fraction
            amount = risk_amount*fraction
            p["executable_delta_weight"] = amount
            p["projected_weight"] += amount
            available_cash -= amount*(1+buy_cost)
            actual_fee += amount*buy_cost
            if amount < risk_amount-EPS:
                p["execution_status"] = "PARTIAL_CASH_LIMIT"
            elif amount < p["delta_weight"]-EPS:
                p["execution_status"] = "PARTIAL_RISK_LIMIT"
        projected_cash = max(0.0, available_cash)
        if any(p["execution_status"] == "BLOCKED_SELL" for p in positions):
            reasons.append("EXECUTION_BLOCKED_OLD_HOLDINGS_RETAINED")
        if any(p["execution_status"] == "BLOCKED_BUY" for p in positions):
            reasons.append("BUY_EXECUTION_BLOCKED")
        if fraction < 1-EPS:
            reasons.append("BUYS_LIMITED_BY_AVAILABLE_CASH")
        if any(p["execution_status"] == "PARTIAL_RISK_LIMIT" for p in positions):
            reasons.append("BUYS_LIMITED_BY_RETAINED_HOLDING_RISK")
        projected_sector = {}
        for p in positions:
            projected_sector[p["industry"]] = projected_sector.get(p["industry"], 0)+p["projected_weight"]
        if (sum(p["projected_weight"] for p in positions) > budget+EPS
                or any(p["projected_weight"] > STOCK_CAP+EPS for p in positions)
                or any(w > INDUSTRY_CAP+EPS for w in projected_sector.values())):
            reasons.append("PROJECTED_HOLDINGS_EXCEED_TARGET_LIMITS_DUE_TO_BLOCKS")
    elif known_holdings:
        reasons.append("EXECUTION_INFORMATION_MISSING")
    return {
        "schema_version": "offline_portfolio_plan_v1",
        "status": "OFFLINE_ALLOCATION_SIMULATION_NOT_VALIDATED",
        "method": "CAPPED_EQUAL_WEIGHT", "weight_basis": "PRE_TRADE_ACCOUNT_EQUITY",
        "score_usage": "RANKING_ONLY_NOT_RETURN_OR_WEIGHT",
        "data_role": snapshot["data_role"], "snapshot_id": snapshot["snapshot_id"],
        "date": snapshot.get("date"), "fold_id": snapshot.get("fold_id"),
        "whitelist_id": whitelist.get("whitelist_id"), "whitelist_version": whitelist.get("version"),
        "production_allowed": False, "research_go": False, "holdout_read": False,
        "sota_promoted": False, "orders_submitted": False,
        "market_budget": budget, "stock_cap": STOCK_CAP, "industry_cap": INDUSTRY_CAP,
        "stock_weight": stock_weight, "cash_weight": cash_weight,
        "cash_after_estimated_cost": max(0.0, cash_weight-cost) if cost is not None else None,
        "estimated_turnover": buys+sells if buys is not None else None,
        "turnover_definition": "SUM_ABSOLUTE_WEIGHT_CHANGES",
        "estimated_cost": cost, "estimated_buy_weight": buys, "estimated_sell_weight": sells,
        "buy_cost_rate": buy_cost, "sell_cost_rate": sell_cost,
        "account_value": account, "holdings_provided": known_holdings,
        "assume_tradable": assume, "projected_cash_weight": projected_cash,
        "simulated_execution_cost": actual_fee,
        "positions": positions, "reason_codes": reasons,
        "warnings": ["离线配置模拟，未验证，不构成交易建议。",
                     "得分仅用于排名；配置采用受约束等权，未拟合收益、协方差或风险最优权重。",
                     "现金权重含预留费用；预计费用和成交模拟按调仓前账户净值计量。",
                     "行业与可交易信息由本次模拟输入提供，未连接真实行情或订单系统。"],
    }


def build_plan(snapshot: dict, whitelist: dict, request: dict, analysis=None) -> dict[str, Any]:
    """Public path: reviewed optimizer only; no implicit equal-weight fallback."""
    from strategy_manager.reviewed_allocation import build_reviewed_plan
    return build_reviewed_plan(snapshot, whitelist, request, analysis)
