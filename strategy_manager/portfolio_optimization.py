"""Deterministic convex portfolio targets with positive selected holdings."""
from __future__ import annotations

import math
import warnings

import numpy as np

from strategy_manager.portfolio_models import AllocationError, require

STOCK_CAP, INDUSTRY_CAP, ANNUAL_DAYS = .05, .30, 239
TOL = 2e-8


def number(value, label, lower=0., upper=1.):
    try:
        assert not isinstance(value, bool) and isinstance(value, (int, float))
        value = float(value)
        assert math.isfinite(value) and lower <= value <= upper
    except (AssertionError, ValueError, TypeError, OverflowError) as exc:
        raise AllocationError("INVALID_ALLOCATION_PARAMETER", f"{label}必须在{lower}至{upper}之间") from exc
    return value


def cost_spec(request):
    supplied = request.get("cost_model")
    if supplied is None:
        return {"buy_rate": number(request.get("buy_cost", .00076), "买入总费率", upper=.1),
                "sell_rate": number(request.get("sell_cost", .00126), "卖出总费率", upper=.1),
                "commission_rate": None, "minimum_commission": 0.,
                "mode": "COMBINED_RATE_INCLUDES_SLIPPAGE"}
    names = ("commission_rate", "transfer_rate", "buy_tax_rate", "sell_tax_rate", "slippage_rate")
    values = {k: number(supplied.get(k, 0.), k, upper=.1) for k in names}
    values["minimum_commission"] = number(supplied.get("minimum_commission", 0.), "最低佣金", upper=10000)
    values["buy_rate"] = sum(values[k] for k in ("commission_rate", "transfer_rate", "buy_tax_rate", "slippage_rate"))
    values["sell_rate"] = sum(values[k] for k in ("commission_rate", "transfer_rate", "sell_tax_rate", "slippage_rate"))
    require(max(values["buy_rate"], values["sell_rate"]) <= .1,
            "INVALID_COST_MODEL", "买卖综合费率不能超过10%")
    values["mode"] = "EXPLICIT_COMPONENTS"
    return values


def transaction_cost(delta, spec, account=None):
    delta = np.asarray(delta, dtype=float)
    absolute = np.abs(delta)
    buy = float(np.maximum(delta, 0).sum())
    sell = float(np.maximum(-delta, 0).sum())
    proportional = buy*spec["buy_rate"]+sell*spec["sell_rate"]
    extra = 0.
    if account is not None and spec["commission_rate"] is not None:
        traded = absolute[absolute > 1e-10]
        extra = float(np.maximum(0., spec["minimum_commission"]/account-traded*spec["commission_rate"]).sum())
    return {"total": proportional+extra, "proportional": proportional, "minimum_commission_extra": extra,
            "buy_weight": buy, "sell_weight": sell, "turnover": buy+sell}


class PortfolioProblem:
    """Shared frozen inputs for primary/sensitivity/execution problems."""
    def __init__(self, snapshot, whitelist, request, analysis):
        self.snapshot, self.whitelist, self.request, self.analysis = snapshot, whitelist, request, analysis
        self.selected = sorted(whitelist.get("codes", []))
        require(bool(self.selected) and len(set(self.selected)) == len(self.selected),
                "EMPTY_OR_DUPLICATE_WHITELIST", "必须保存非空且无重复的白名单")
        require(whitelist.get("snapshot_id") == snapshot.get("snapshot_id"),
                "WHITELIST_SNAPSHOT_MISMATCH", "白名单与评分快照不一致")
        rows = {r["code"]: r for r in snapshot["rows"]}
        bad = [c for c in self.selected if c not in rows or rows[c].get("signal_valid") is not True
               or not isinstance(rows[c].get("score"), (float, int)) or not math.isfinite(rows[c]["score"])]
        require(not bad, "SELECTED_SCORE_MISSING", "所选股票缺少有效评分，不能将其置零后生成成功方案", codes=bad)
        require(request.get("minimum_weight") is not None, "MINIMUM_WEIGHT_REQUIRED",
                "请设置每只所选股票的最低目标权重")
        self.minimum = number(request["minimum_weight"], "最低目标权重", 1e-6, STOCK_CAP)
        self.budget = number(request.get("market_budget", 1.), "股票总预算")
        self.volatility_cap = number(request.get("volatility_cap", .12), "年化波动上限", .001, 1.)
        self.risk_aversion = number(request.get("risk_aversion", 3.), "风险厌恶系数", .01, 100.)
        self.range_tolerance_bps = number(request.get("range_utility_tolerance_bps", 1.), "区间效用容忍度（基点）", 0., 100.)
        self.deadband = number(request.get("no_trade_band", .005), "调仓阈值", 0., .05)
        self.account = request.get("account_value")
        if self.account is not None:
            self.account = number(self.account, "账户资金", 1e-8, 1e16)
        self.known_holdings = request.get("holdings") is not None
        items = request.get("holdings") or []
        require(isinstance(items, list) and len(items) <= 1000, "INVALID_HOLDINGS", "持仓输入格式无效")
        self.holdings = {}
        for h in items:
            c = h.get("code")
            require(isinstance(c, str) and c and c not in self.holdings, "INVALID_HOLDINGS", "持仓代码不能为空或重复")
            self.holdings[c] = {**h, "current_weight": number(h.get("current_weight"), "当前权重")}
            for k in ("can_buy", "can_sell"):
                require(h.get(k) is None or type(h[k]) is bool, "INVALID_TRADABILITY", "可交易性必须是布尔值或未知")
        self.codes = sorted(set(self.selected) | set(self.holdings))
        require(set(request.get("tradability") or {}).issubset(set(self.codes)),
                "INVALID_TRADABILITY", "可交易性映射含本次方案之外的股票")
        self.index = {c: i for i, c in enumerate(self.codes)}
        self.current = np.array([self.holdings.get(c, {}).get("current_weight", 0.) for c in self.codes])
        require(self.current.sum() <= 1+TOL, "INVALID_HOLDINGS", "当前持仓合计不能超过100%")
        self.costs = cost_spec(request)
        require(analysis is not None, "ALLOCATION_EVIDENCE_REQUIRED", "当前评分包尚无收益校准和风险输入；请导入配置证据包")
        require(analysis.get("snapshot_id") == snapshot.get("snapshot_id")
                and analysis.get("fingerprints") == snapshot.get("fingerprints"),
                "ALLOCATION_EVIDENCE_MISMATCH", "收益与风险证据没有绑定当前评分快照")
        missing = [c for c in self.codes if c not in analysis["risk_codes"]]
        require(not missing, "RISK_HISTORY_MISSING", "所选股票或旧仓缺少完整风险历史", codes=missing)
        require(all(c in analysis["expected"] for c in self.selected), "RETURN_ESTIMATE_MISSING", "所选股票缺少收益校准")
        risk_lookup = {c: i for i, c in enumerate(analysis["risk_codes"])}
        idx = [risk_lookup[c] for c in self.codes]
        covariance = np.asarray(analysis["covariance"], dtype=float)
        self.cov = covariance[np.ix_(idx, idx)]
        require(np.isfinite(self.cov).all() and np.allclose(self.cov, self.cov.T, atol=1e-12)
                and np.linalg.eigvalsh(self.cov).min() >= -1e-10,
                "INVALID_RISK_MATRIX", "风险矩阵必须有限、对称、半正定")
        self.mu = np.array([analysis["expected"].get(c, {}).get("expected_return", 0.) for c in self.codes])
        self.se = np.array([analysis["expected"].get(c, {}).get("standard_error", 0.) for c in self.codes])
        require(np.isfinite(self.mu).all() and np.isfinite(self.se).all() and (self.se >= 0).all(),
                "INVALID_RETURN_ESTIMATE", "收益估计或误差无效")
        self.horizon = analysis["horizon"]
        self.market = analysis.get("market", {})
        provided = request.get("industries") or {}
        require(set(provided).issubset(set(rows) | set(self.codes)), "INVALID_INDUSTRY", "行业映射含未知股票")
        self.industries = []
        for c in self.codes:
            authoritative = self.market.get(c, {}).get("industry")
            manual = provided.get(c) or self.holdings.get(c, {}).get("industry")
            require(not authoritative or not manual or manual == authoritative,
                    "INDUSTRY_OVERRIDE_CONFLICT", "行业输入与版本化证据冲突，不能覆盖行业规避上限", code=c)
            value = authoritative or manual or "UNKNOWN"
            require(isinstance(value, str) and len(value) <= 100, "INVALID_INDUSTRY", "行业名称无效")
            self.industries.append(value.strip() or "UNKNOWN")
        self.groups = {g: np.array([i for i, industry in enumerate(self.industries) if industry == g])
                       for g in sorted(set(self.industries))}
        self.base_lower = np.array([self.minimum if c in self.selected else 0. for c in self.codes])
        self.base_upper = np.array([STOCK_CAP if c in self.selected else 0. for c in self.codes])
        require(self.base_lower.sum() <= self.budget+TOL, "MINIMUM_BUDGET_INFEASIBLE",
                "股票最低权重合计超过总预算", minimum_total=float(self.base_lower.sum()), budget=self.budget)
        for sector, group in self.groups.items():
            require(self.base_lower[group].sum() <= INDUSTRY_CAP+TOL, "MINIMUM_INDUSTRY_INFEASIBLE",
                    "所选股票的最低权重超过行业上限", industry=sector)

    def permission(self, code, field):
        supplied = self.request.get("tradability") or {}
        choices = [self.holdings.get(code, {}).get(field), supplied.get(code, {}).get(field)]
        require(all(x is None or type(x) is bool for x in choices), "INVALID_TRADABILITY", "可交易性输入无效")
        if False in choices:
            return False
        if True in choices:
            return True
        return True if self.request.get("assume_tradable") is True else None

    def metrics(self, weights, uncertainty=1.):
        weights = np.asarray(weights)
        variance = float(max(0., weights@self.cov@weights))
        cost = transaction_cost(weights-self.current, self.costs, self.account) if self.known_holdings else None
        expected = float(self.mu@weights)
        deduction = float(uncertainty*self.se@weights)
        risk = self.risk_aversion*variance
        return {"expected_return": expected, "uncertainty_deduction": deduction, "risk_penalty": risk,
                "expected_volatility": math.sqrt(variance*ANNUAL_DAYS/self.horizon),
                "transaction_cost": cost["total"] if cost else None,
                "utility": expected-deduction-risk-(cost["total"] if cost else 0.), "cost_details": cost}

    def feasible(self, weights, *, lower=None, upper=None):
        weights = np.asarray(weights)
        lo = self.base_lower if lower is None else lower
        hi = self.base_upper if upper is None else upper
        return (np.isfinite(weights).all() and np.all(weights >= lo-TOL) and np.all(weights <= hi+TOL)
                and weights.sum() <= self.budget+TOL
                and all(weights[g].sum() <= INDUSTRY_CAP+TOL for g in self.groups.values())
                and self.metrics(weights)["expected_volatility"] <= self.volatility_cap+TOL
                and weights.sum()+(self.metrics(weights)["transaction_cost"] or 0.) <= 1+TOL)

    def solve(self, *, uncertainty=1., fixed=None, lower=None, upper=None, reference=None):
        import cvxpy as cp
        lo = self.base_lower.copy() if lower is None else np.asarray(lower).copy()
        hi = self.base_upper.copy() if upper is None else np.asarray(upper).copy()
        for code, amount in (fixed or {}).items():
            i = self.index[code]
            require(lo[i]-TOL <= amount <= hi[i]+TOL, "FIXED_HOLDING_CONFLICT", "固定持仓与本次目标约束冲突", code=code)
            lo[i] = hi[i] = amount
        require(np.all(lo <= hi) and lo.sum() <= self.budget+TOL,
                "ALLOCATION_CONSTRAINT_INFEASIBLE", "正权重、持仓或预算约束无法同时满足")
        w = cp.Variable(len(self.codes))
        risk = cp.quad_form(w, cp.psd_wrap(self.cov))
        fee = (self.costs["buy_rate"]*cp.sum(cp.pos(w-self.current))
               + self.costs["sell_rate"]*cp.sum(cp.pos(self.current-w))) if self.known_holdings else 0.
        reserve = cp.Parameter(nonneg=True)
        compulsory = np.count_nonzero((lo > self.current+1e-10) | (hi < self.current-1e-10))
        reserve.value = (compulsory*self.costs["minimum_commission"]/self.account
                         if self.known_holdings and self.account is not None else 0.)
        constraints = [w >= lo, w <= hi, cp.sum(w) <= self.budget,
                       cp.sum(w)+fee+reserve <= 1,
                       risk <= self.volatility_cap**2*self.horizon/ANNUAL_DAYS]
        constraints += [cp.sum(w[group]) <= INDUSTRY_CAP for group in self.groups.values()]
        utility = (self.mu-uncertainty*self.se)@w-self.risk_aversion*risk-fee
        objective = cp.Maximize(utility) if reference is None else cp.Minimize(cp.sum_squares(w-reference))
        problem = cp.Problem(objective, constraints)
        error = None
        for attempt in range(6):
            tolerance = 1e-10 if attempt == 0 else 1e-11
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    problem.solve(solver="CLARABEL", max_iter=250, tol_gap_abs=tolerance,
                                  tol_feas=tolerance, tol_gap_rel=tolerance, max_threads=1)
                if problem.status in ("infeasible", "infeasible_inaccurate"):
                    if self.known_holdings and self.feasible(self.current, lower=lo, upper=hi) and reference is None:
                        return self.current.copy(), {"solver": "CLARABEL", "solver_status": "NO_CHANGE_FEE_BUFFER_LIMIT",
                                                     "cvxpy_version": cp.__version__, "tolerance": tolerance}
                    raise AllocationError("ALLOCATION_CONSTRAINT_INFEASIBLE", "正目标、波动、费用和行业约束无法同时满足")
                if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
                    result = np.clip(np.asarray(w.value), lo, hi)
                    if self.feasible(result, lower=lo, upper=hi):
                        return result, {"solver": "CLARABEL", "solver_status": problem.status,
                                        "cvxpy_version": cp.__version__, "tolerance": tolerance}
                    extra = transaction_cost(result-self.current, self.costs, self.account)["minimum_commission_extra"]
                    if extra > reserve.value+1e-12:
                        reserve.value = extra
            except AllocationError:
                raise
            except cp.error.SolverError as exc:
                error = type(exc).__name__
        raise AllocationError("OPTIMIZER_FAILED", "优化求解未通过残差和资金检查；未切换为等权", {"solver_error": error})
