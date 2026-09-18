"""Feasible conditional weight bands and explanations of binding constraints.

Bands vary one holding against cash, leaving every other target unchanged.
They measure a declared utility tolerance, not statistical confidence. Scalar
quadratic constraints avoid 600 additional portfolio solves for 300 stocks.
"""
from __future__ import annotations

import math
import numpy as np

from strategy_manager.portfolio_optimization import ANNUAL_DAYS, STOCK_CAP, INDUSTRY_CAP, transaction_cost


def _clip_quadratic(left, right, a, b, c):
    """Intersect [left,right] with a*x*x+b*x+c <= 0, for a >= 0."""
    if a <= 1e-18:
        if abs(b) <= 1e-18:
            return (left, right) if c <= 1e-14 else None
        boundary = -c/b
        left, right = (left, min(right, boundary)) if b > 0 else (max(left, boundary), right)
    else:
        disc = b*b-4*a*c
        if disc < 0:
            return None
        root = math.sqrt(disc)
        # The second root via the product avoids cancellation near zero.
        q = -.5*(b+math.copysign(root, b))
        roots = sorted((q/a, c/q)) if q else (0., 0.)
        left, right = max(left, roots[0]), min(right, roots[1])
    return (left, right) if left <= right else None


def conditional_weight_ranges(p, target, *, fixed=None):
    """Connected feasible bands containing target, with exact per-order fees.

Minimum commissions make fees discontinuous at no trade. Split fee regimes,
solve their quadratic inequalities separately, and never bridge an infeasible
gap. Execution/lot constraints remain in the separate execution simulation.
"""
    fixed = fixed or {}
    tolerance = p.range_tolerance_bps/10000
    total = float(target.sum())
    cov_w = p.cov@target
    variance = float(target@cov_w)
    fees = [transaction_cost([target[i]-p.current[i]], p.costs, p.account)["total"]
            if p.known_holdings else 0. for i in range(len(target))]
    total_fees = sum(fees)
    low, high, statuses = target.copy(), target.copy(), []
    for i, code in enumerate(p.codes):
        if code in fixed or p.base_lower[i] == p.base_upper[i]:
            statuses.append("FIXED_BY_DEADBAND" if code in fixed else "FIXED_BY_BOUNDS")
            continue
        center = float(target[i])
        sector_sum = float(target[p.groups[p.industries[i]]].sum())
        left = float(p.base_lower[i])
        right = min(float(p.base_upper[i]), center+p.budget-total,
                    center+INDUSTRY_CAP-sector_sum)
        right = max(center, right)  # target already passed the solver residual check
        current = float(p.current[i])
        cuts = [left, right, center]
        if p.known_holdings:
            cuts.append(current)
            if p.account and p.costs["minimum_commission"]:
                cuts.extend([current-1e-10, current+1e-10])
                commission = p.costs["commission_rate"] or 0.
                if commission:
                    distance = p.costs["minimum_commission"]/p.account/commission
                    cuts.extend([current-distance, current+distance])
        cuts = sorted(set(x for x in cuts if left <= x <= right))

        def fee(x):
            return transaction_cost([x-current], p.costs, p.account)["total"] if p.known_holdings else 0.

        def valid(x):
            d = x-center
            var = variance+2*cov_w[i]*d+p.cov[i,i]*d*d
            loss = -(p.mu[i]-p.se[i])*d+p.risk_aversion*(var-variance)+fee(x)-fees[i]
            return (var <= p.volatility_cap**2*p.horizon/ANNUAL_DAYS+1e-13
                    and total+d+total_fees-fees[i]+fee(x) <= 1+1e-12
                    and loss <= tolerance+1e-12)

        pieces = [(x, x) for x in cuts if valid(x)]
        for start, end in zip(cuts, cuts[1:]):
            if end-start < 1e-15:
                continue
            # Fees are affine within each regime; all remaining restrictions
            # are linear or convex quadratic in the change of this weight.
            x1, x2 = start+(end-start)/3, start+2*(end-start)/3
            slope = (fee(x2)-fee(x1))/(x2-x1)
            intercept = fee(x1)-slope*(x1-center)
            bounds = (start-center, end-center)
            inequalities = (
                (p.cov[i,i], 2*cov_w[i], variance-p.volatility_cap**2*p.horizon/ANNUAL_DAYS),
                (0., 1+slope, total+total_fees-fees[i]+intercept-1),
                (p.risk_aversion*p.cov[i,i],
                 -(p.mu[i]-p.se[i])+2*p.risk_aversion*cov_w[i]+slope,
                 intercept-fees[i]-tolerance),
            )
            for a, b, c in inequalities:
                bounds = _clip_quadratic(*bounds, a, b, c)
                if bounds is None:
                    break
            if bounds is None:
                continue
            a, b = center+bounds[0], center+bounds[1]
            # Exclude a fee jump's closed endpoint if the actual fee fails it.
            if not valid(a):
                a = min(b, a+1e-12)
            if not valid(b):
                b = max(a, b-1e-12)
            if a <= b and valid(a) and valid(b):
                pieces.append((a, b))
        pieces.append((center, center))
        merged = []
        for a, b in sorted(pieces):
            if merged and a <= merged[-1][1]+1e-14 and valid((a+merged[-1][1])/2):
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        for a, b in merged:
            if a <= center <= b:
                low[i], high[i] = max(left, a), min(right, b)
                break
        statuses.append("CONDITIONAL_FEASIBLE_BAND" if high[i]-low[i] > 1e-8 else "NO_FEASIBLE_WIDTH")
    return low, high, statuses


def allocation_diagnostics(p, target, fixed):
    selected = [p.index[c] for c in p.selected]
    at_cap = [p.codes[i] for i in selected if abs(target[i]-STOCK_CAP) < 1e-7]
    at_minimum = [p.codes[i] for i in selected if abs(target[i]-p.minimum) < 1e-7]
    metrics = p.metrics(target)
    budget_binding = abs(float(target.sum())-p.budget) < 1e-7
    risk_binding = abs(metrics["expected_volatility"]-p.volatility_cap) < 1e-7
    sectors = [g for g, ids in p.groups.items() if abs(float(target[ids].sum())-INDUSTRY_CAP) < 1e-7]
    bounds_locked = p.minimum == STOCK_CAP
    all_at_cap = len(at_cap) == len(selected)
    messages = []
    if bounds_locked:
        messages.append("每只最低目标与单股上限同为5%，全部目标被固定；调整收益、风险和成本无法改变权重。降低最低目标后才有优化空间。")
    elif all_at_cap:
        messages.append("全部所选股票达到单股5%上限，形成相同目标。收益和风险不同，也可能同时触及同一上限；这不是自动切回等权算法。")
    else:
        messages.append(f"{len(at_cap)}/{len(selected)}只达到5%上限，{len(at_minimum)}只达到用户最低目标；其余权重由收益、风险与成本共同决定。")
    messages.append(f"股票仓位{target.sum():.2%} / 预算上限{p.budget:.2%}；"
                    + ("预算上限正在约束配置。" if budget_binding else "预算尚未用满，继续调高预算不会增加当前最优仓位。"))
    messages.append(f"预测年化波动{metrics['expected_volatility']:.2%} / 上限{p.volatility_cap:.2%}；"
                    + ("波动上限正在约束配置。" if risk_binding else "波动上限尚未触及，继续放宽不会改变当前最优目标。"))
    if not p.known_holdings:
        messages.append("持仓未知：本次未计买卖成本；费用和调仓阈值无法影响目标。")
    elif not np.any(p.current):
        messages.append("当前空仓：调仓阈值不冻结首次建仓；卖出费率在没有卖出时不产生费用。")
    if fixed:
        messages.append(f"{len(fixed)}只旧仓由小额调仓阈值固定；这些股票的建议区间随之固定。")
    release = None
    # Local first release threshold is valid for this smooth all-cap corner;
    # do not claim it for arbitrary holdings, fee jumps or other active limits.
    if (all_at_cap and not bounds_locked and not budget_binding and not risk_binding
            and not sectors and not np.any(p.current) and not p.costs['minimum_commission']):
        denominator = 2*(p.cov@target)[selected]
        net = (p.mu-p.se)[selected]-(p.costs['buy_rate'] if p.known_holdings else 0.)
        if np.all(denominator > 0) and np.all(net > 0):
            release = float(np.min(net/denominator))
            messages.append(f"保持其余输入不变，风险厌恶系数约超过{release:.2f}后，才可能有股票退出5%上限（局部边际估计）。")
    return {"all_targets_fixed_by_bounds": bounds_locked, "all_selected_at_stock_cap": all_at_cap,
            "at_stock_cap": at_cap, "at_minimum": at_minimum, "budget_binding": budget_binding,
            "volatility_binding": risk_binding, "binding_industries": sectors,
            "fixed_by_deadband": list(fixed), "risk_aversion_first_cap_release": release,
            "messages": messages}
