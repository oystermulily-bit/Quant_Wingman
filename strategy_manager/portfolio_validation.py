"""Causal offline replay and predeclared comparison gates for allocations.

Callers supply point-in-time frames and subsequent opening-to-opening daily
settlements separately. No research files or Holdout are discovered here.
Synthetic and retrospective user lists can never produce a research GO.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

import numpy as np

from strategy_manager.portfolio_models import AllocationError, require, stamp, RULE_VERSION
from strategy_manager.portfolio_optimization import PortfolioProblem, ANNUAL_DAYS, transaction_cost
from strategy_manager.reviewed_allocation import build_reviewed_plan, _execution

METHODS = ("optimizer", "constrained_equal", "constrained_inverse_volatility")
COST_MULTIPLIERS = (0., 1., 1.5)


def replay(frames, request, *, method="optimizer", cost_multiplier=1., phase=0):
    require(method in METHODS and cost_multiplier in COST_MULTIPLIERS,
            "INVALID_REPLAY_METHOD", "回放方法或成本情景未登记")
    require(bool(frames), "EMPTY_REPLAY", "没有可回放日期")
    ledger, positions, cash, nav = [], {}, 1., 1.
    previous_end = previous_signal = None
    calibration_by_fold = {}
    dates_seen = set()
    research_eligible = True
    assumptions = set()
    for frame in frames:
        snapshot, white, analysis = frame["snapshot"], frame["whitelist"], frame["analysis"]
        at, h = stamp(snapshot["available_at"]), analysis["horizon"]
        require(0 <= phase < h, "INVALID_REPLAY_PHASE", "回放相位不属于模型周期")
        require(previous_signal is None or previous_signal < at, "REPLAY_TIME_OVERLAP", "历史评分必须严格按时间顺序回放")
        settlements = frame.get("settlements", [])
        require(len(settlements) == h, "INCOMPLETE_REPLAY_SETTLEMENT", "每次回放须包含完整H个逐日结算，不能只取盈利日期")
        execution_at = stamp(settlements[0]["start_at"])
        require(at < execution_at and (previous_end is None or previous_end == execution_at),
                "UNPRICED_REPLAY_GAP", "旧仓必须连续计价至本次执行开盘；不能省略两个调仓点之间的收益")
        # Targets use frozen forecasts/risk. Holdings are marked at the actual
        # execution opening, before any subsequent settlement return is known.
        require(stamp(analysis["available_at"]) <= at, "REPLAY_INPUT_LEAKAGE", "风险与校准输入晚于决策")
        fold = snapshot["fold_id"]
        calibration_hash = sha256(json.dumps(analysis["calibrator"], sort_keys=True).encode()).hexdigest()
        require(fold not in calibration_by_fold or calibration_by_fold[fold] == calibration_hash,
                "OUTER_CALIBRATOR_CHANGED", "同一外层折内不得用后续收益重调校准器")
        if fold not in calibration_by_fold:
            require(stamp(analysis["calibrator"]["frozen_at"]) < at,
                    "CALIBRATOR_NOT_FROZEN_BEFORE_OUTER", "外层评估前必须冻结收益校准器")
        calibration_by_fold[fold] = calibration_hash
        declared_selection = frame.get("selection_available_at")
        causal = declared_selection is not None and stamp(declared_selection) <= at
        research_eligible &= causal and frame.get("selection_kind") == "frozen_policy" and snapshot["data_role"] == "development"
        if not causal or frame.get("selection_kind") != "frozen_policy": assumptions.add("RETROSPECTIVE_WHITELIST_DIAGNOSTIC_ONLY")
        if snapshot["data_role"] != "development": assumptions.add("SYNTHETIC_ENGINEERING_ONLY")
        current_request = deepcopy(request)
        require(current_request.get("account_value") is None, "REPLAY_WEIGHT_LEDGER_ONLY",
                "本历史回放器按比例账本运行；真实账户股数回放须另提供逐日价格、股数与执行事件")
        current_request["holdings"] = [{"code": c, "current_weight": value/nav} for c,value in positions.items() if value > 1e-12]
        current_request["tradability"] = frame.get("tradability", {})
        if current_request.get("cost_model") is not None:
            require(current_request["cost_model"].get("minimum_commission", 0.) == 0,
                    "REPLAY_ABSOLUTE_FEE_UNSUPPORTED", "比例回放不得忽略绝对最低佣金")
            for key in ("commission_rate", "transfer_rate", "buy_tax_rate", "sell_tax_rate", "slippage_rate"):
                current_request["cost_model"][key] = current_request["cost_model"].get(key, 0.)*cost_multiplier
        else:
            current_request["buy_cost"] = current_request.get("buy_cost", .00076)*cost_multiplier
            current_request["sell_cost"] = current_request.get("sell_cost", .00126)*cost_multiplier
        problem = PortfolioProblem(snapshot, white, current_request, analysis)
        if method == "optimizer":
            plan = build_reviewed_plan(snapshot, white, current_request, analysis)
            projected = {r["code"]: r["projected_weight"] for r in plan["positions"]}
            require(all(v is not None for v in projected.values()), "REPLAY_EXECUTION_UNKNOWN", "执行信息未知，不能虚构成交回测")
            fee = plan["simulated_execution_cost"]
            execution_reasons = plan["reason_codes"]
        else:
            reference = np.zeros(len(problem.codes))
            ids = [problem.index[c] for c in problem.selected]
            base = np.ones(len(ids)) if method == "constrained_equal" else 1/np.sqrt(np.maximum(np.diag(problem.cov)[ids],1e-12))
            reference[ids] = base/base.sum()*problem.budget
            target,_ = problem.solve(reference=reference)
            weights, fee, _, execution_reasons, _ = _execution(problem,target,False,reference=True)
            require(weights is not None, "REPLAY_EXECUTION_UNKNOWN", "对照执行信息未知")
            projected = dict(zip(problem.codes,weights.tolist()))
        before_trade = nav
        positions = {c: weight*before_trade for c,weight in projected.items() if weight > 1e-12}
        cash = before_trade-sum(positions.values())-fee*before_trade
        require(cash >= -1e-9, "REPLAY_CASH_FAILURE", "回放资金不守恒")
        target_audit = {"snapshot_id": snapshot["snapshot_id"], "analysis_id": analysis["analysis_id"],
                        "whitelist_id": white["whitelist_id"], "whitelist_version": white["version"],
                        "calibration_hash": calibration_hash, "execution_reasons": execution_reasons,
                        "projected_weights": projected, "execution_at": execution_at.isoformat()}
        for j, day in enumerate(settlements):
            start, end = stamp(day["start_at"]), stamp(day["end_at"])
            require(at < start < end and (j == 0 or start == previous_end),
                    "INVALID_SETTLEMENT_TIME", "结算必须在决策后，逐日区间连续且严格有序")
            key = end.date().isoformat()
            require(key not in dates_seen, "DUPLICATE_REPLAY_DATE", "同一相位的日期不能重复结算")
            dates_seen.add(key)
            realized = day["returns"]
            require(set(positions).issubset(realized), "REPLAY_MISSING_HELD_RETURN", "旧仓或新仓缺失结算收益，不能填零")
            contributions = {}
            denominator = before_trade if j == 0 else nav
            for c,value in list(positions.items()):
                ret = realized[c]
                require(type(ret) in (int,float) and math.isfinite(ret) and -1 <= ret < 10,
                        "INVALID_SETTLEMENT_RETURN", "结算收益无效")
                sector = analysis.get("market", {}).get(c, {}).get("industry", "UNKNOWN")
                contributions[sector] = contributions.get(sector,0.)+value*ret/denominator
                positions[c] = value*(1+ret)  # a -100% delisting loss stays in NAV
            cash_return = day.get("cash_return", 0.)
            require(type(cash_return) in (int,float) and math.isfinite(cash_return) and -.1 < cash_return < .1,
                    "INVALID_CASH_RETURN", "现金日收益无效")
            cash *= 1+cash_return
            nav = cash+sum(positions.values())
            require(nav > 0, "REPLAY_BANKRUPT", "账户净值归零，不能继续计算比例")
            ledger.append({"date":key,"fold_id":fold,"phase":phase,"net_return":nav/denominator-1,
                           "net_excess_return":nav/denominator-1-cash_return,"nav":nav,
                           "stock_exposure":sum(positions.values())/nav,"cash":cash,
                           "trade_cost":fee if j==0 else 0.,"industry_contributions":contributions,
                           **target_audit})
            previous_end=end
        previous_signal=at
    return {"method":method,"cost_multiplier":cost_multiplier,"phase":phase,"ledger":ledger,
            "research_eligible":bool(research_eligible),"assumptions":sorted(assumptions),
            "timing":"FROZEN_PRIOR_SIGNAL_OPTIMIZE_AT_EXECUTION_OPEN_MARKED_HOLDINGS",
            "rule_version":RULE_VERSION,"production_allowed":False,"holdout_read":False}


def performance(rows):
    returns=np.array([r["net_return"] for r in rows],dtype=float)
    excess=np.array([r.get("net_excess_return",r["net_return"]) for r in rows],dtype=float)
    if len(returns)<2: return {"days":len(returns),"sharpe":None,"annual_return":None,"max_drawdown":None}
    curve=np.r_[1.,np.cumprod(1+returns)]
    std=excess.std(ddof=1)
    return {"days":len(returns),"sharpe":float(excess.mean()/std*np.sqrt(ANNUAL_DAYS)) if std>1e-12 else None,
            "annual_return":float(curve[-1]**(ANNUAL_DAYS/len(returns))-1),
            "max_drawdown":float(np.max(1-curve/np.maximum.accumulate(curve))),
            "mean_stock_exposure":float(np.mean([r["stock_exposure"] for r in rows])),
            "total_cost_nav_ratios":float(sum(r["trade_cost"] for r in rows))}


def compare_replays(runs, horizon, *, robustness=None):
    """All phases, five folds and cost stresses; missing evidence cannot pass.

    `robustness` holds separately rerun exclusion comparisons, never inferred
    from a favorable headline. This function returns research evidence only.
    """
    require(horizon in (1,3,5), "INVALID_HORIZON", "复核周期无效")
    by_key={(r["method"],r["cost_multiplier"],r["phase"]):r for r in runs}
    require(len(by_key)==len(runs), "DUPLICATE_REPLAY", "同一方法／成本／相位不能挑选多次试验中的赢家")
    missing=[]; summaries=[]; pairs=[]; direction=[]; phase_deltas=[]; fold_deltas={}
    reference_signatures = {}
    for run in runs:
        signature = [(row["date"],row["fold_id"],row["snapshot_id"],row["whitelist_id"],
                      row["whitelist_version"],row["analysis_id"]) for row in run["ledger"]]
        phase = run["phase"]
        require(phase not in reference_signatures or signature == reference_signatures[phase],
                "UNPAIRED_REPLAY", "所有方法和费用情景须使用同一输入、名单、折和日期")
        reference_signatures[phase] = signature
    for method in METHODS:
        for cost in COST_MULTIPLIERS:
            for phase in range(horizon):
                r=by_key.get((method,cost,phase))
                if r is None: missing.append(f"{method}/{cost}/{phase}")
                else: summaries.append({"method":method,"cost_multiplier":cost,"phase":phase,**performance(r["ledger"])})
    numeric_good=True; stress_good=True; drawdown_good=True
    for phase in range(horizon):
        candidate=by_key.get(("optimizer",1.,phase)); equal=by_key.get(("constrained_equal",1.,phase))
        if not candidate or not equal: continue
        c,b=candidate["ledger"],equal["ledger"]
        if len({x["fold_id"] for x in c}) != 5:
            missing.append(f"five_outer_folds_for_phase/{phase}")
        require([(x["date"],x["fold_id"]) for x in c]==[(x["date"],x["fold_id"]) for x in b],
                "UNPAIRED_REPLAY", "对照须使用完全相同日期、折和名单")
        require([(x["snapshot_id"],x["whitelist_id"],x["whitelist_version"],x["analysis_id"]) for x in c]
                ==[(x["snapshot_id"],x["whitelist_id"],x["whitelist_version"],x["analysis_id"]) for x in b],
                "UNPAIRED_REPLAY", "对照须绑定同一输入及白名单")
        cm,bm=performance(c),performance(b)
        if len(c)<252 or cm["sharpe"] is None or bm["sharpe"] is None:
            missing.append(f"mature_paired_dates/{phase}"); continue
        delta=cm["sharpe"]-bm["sharpe"]; phase_deltas.append(delta); direction.append(delta>0)
        numeric_good &= cm["sharpe"]>0 and cm["annual_return"]>0
        drawdown_good &= cm["max_drawdown"]<=bm["max_drawdown"]
        pairs.append((c,b))
        for fold in sorted({x["fold_id"] for x in c}):
            cf=performance([x for x in c if x["fold_id"]==fold]); bf=performance([x for x in b if x["fold_id"]==fold])
            if cf["sharpe"] is None or bf["sharpe"] is None: missing.append("degenerate_fold/"+fold)
            else: fold_deltas.setdefault(fold,[]).append(cf["sharpe"]-bf["sharpe"])
        cs=by_key.get(("optimizer",1.5,phase)); bs=by_key.get(("constrained_equal",1.5,phase))
        if cs and bs:
            csm,bsm=performance(cs["ledger"]),performance(bs["ledger"])
            stress_good &= (csm["sharpe"] is not None and bsm["sharpe"] is not None
                            and csm["annual_return"]>0 and csm["sharpe"]>=bsm["sharpe"])
    interval=None
    # Resample common calendar blocks for all overlapping phase portfolios,
    # preserving cross-phase dependence instead of counting phases as new days.
    if len(pairs)==horizon:
        common=sorted(set.intersection(*[{r["date"] for r in c} for c,b in pairs]))
        if len(common)>=252:
            arrays=[]
            for c,b in pairs:
                cm={r["date"]:r["net_excess_return"] for r in c}; bm={r["date"]:r["net_excess_return"] for r in b}
                arrays.append(np.array([[cm[d],bm[d]] for d in common]))
            rng=np.random.default_rng(20260917); distribution=[]
            for _ in range(2000):
                starts=rng.integers(0,len(common)-19,size=math.ceil(len(common)/20))
                indices=(starts[:,None]+np.arange(20)).ravel()[:len(common)]
                deltas=[]
                for arr in arrays:
                    sample=arr[indices]; std=sample.std(axis=0,ddof=1)
                    if (std<=1e-12).any(): break
                    sharpe=sample.mean(axis=0)/std*np.sqrt(ANNUAL_DAYS); deltas.append(sharpe[0]-sharpe[1])
                if len(deltas)==horizon: distribution.append(float(np.median(deltas)))
            if len(distribution)==2000: interval=np.quantile(distribution,[.025,.975]).tolist()
            else: missing.append("degenerate_bootstrap")
        else: missing.append("common_phase_dates")
    if len(fold_deltas)!=5: missing.append("five_outer_folds")
    if not robustness or not all(k in robustness for k in ("exclude_best_year", "exclude_largest_industry", "matched_risk_exposure")):
        missing.append("exclusion_reruns_and_matched_exposure")
    positive_folds=sum(np.median(values)>0 for values in fold_deltas.values())
    gates={"median_sharpe_delta_at_least_015":bool(phase_deltas and np.median(phase_deltas)>=.15),
           "positive_candidate_return_and_sharpe":bool(numeric_good and phase_deltas),
           "paired_bootstrap_lower_positive":bool(interval and interval[0]>0),
           "three_of_five_folds_positive":positive_folds>=3 and len(fold_deltas)==5,
           "majority_phases_positive":sum(direction)>horizon/2,
           "no_higher_drawdown":bool(drawdown_good and phase_deltas),
           "cost_15_stress":bool(stress_good and not any("/1.5/" in x for x in missing)),
           "exclusion_robustness":bool(robustness and all(robustness.get(k,{}).get("sharpe_delta",-1)>0
                                                       for k in ("exclude_best_year","exclude_largest_industry"))),
           "causal_development_inputs":bool(runs) and all(r.get("research_eligible") is True for r in runs),
           "execution_no_unresolved_risk":not any("RETAINED_HOLDINGS_EXCEED_RISK_LIMITS" in row.get("execution_reasons",[])
                                                 for run in runs for row in run["ledger"])}
    passed=not missing and all(gates.values())
    return {"schema_version":"portfolio_validation_v1","rule_version":RULE_VERSION,
            "status":"RESEARCH_COMPARISON_PASSED_NOT_PRODUCTION" if passed else "INSUFFICIENT_EVIDENCE" if missing else "NOT_PASSED",
            "comparison_passed":passed,"production_allowed":False,"holdout_read":False,"research_go":False,
            "summaries":summaries,"gates":gates,"missing_evidence":missing,
            "median_net_sharpe_delta":float(np.median(phase_deltas)) if phase_deltas else None,
            "paired_95pct_interval":interval,"bootstrap":{"block_days":20,"draws":2000,"seed":20260917},
            "fold_median_sharpe_deltas":{k:float(np.median(v)) for k,v in fold_deltas.items()},
            "robustness":robustness,"ledger_scope":"PROPORTIONAL_COST_WEIGHT_LEDGER_NO_LOT_FILL_BACKTEST"}
