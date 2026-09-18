"""Behavioral checks for the reviewed optimizer and execution ledger."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strategy_manager.offline_allocation import build_plan
from strategy_manager.portfolio_models import AllocationError, prepare_inputs
from strategy_manager.portfolio_optimization import transaction_cost, cost_spec


def example():
    codes = ["A", "B", "C"]
    snapshot = {"snapshot_id": "snapshot_test", "date": "2020-01-02", "fold_id": "outer_1",
                "available_at": "2020-01-02T13:00:00Z", "data_role": "synthetic", "production_allowed": False,
                "fingerprints": {k: k*64 for k in ("model", "formula", "data", "fold")},
                "rows": [{"code": c, "score": float(3-i), "rank": i+1, "signal_valid": True} for i,c in enumerate(codes)]}
    white = {"snapshot_id": snapshot["snapshot_id"], "whitelist_id": "w", "version": 1, "codes": codes}
    analysis = {"snapshot_id": snapshot["snapshot_id"], "fingerprints": snapshot["fingerprints"],
                "analysis_id": "analysis_test", "source_sha256": "a"*64, "horizon": 5,
                "risk_codes": codes.copy(), "covariance": np.diag([.02,.02,.02]).tolist(),
                "calibrator": {}, "risk_method": "TEST_FIXTURE", "expected": {
                    c: {"expected_return": mu, "standard_error": .0001} for c,mu in zip(codes,[.003,.0015,-.001])},
                "market": {c: {"industry": "sector_"+c, "median_amount_20d": 100e6, "mean_amount_20d": 100e6,
                               "price": 20., "lot_size": 100, "available_at": snapshot["available_at"]} for c in codes}}
    request = {"minimum_weight": .001, "holdings": [], "assume_tradable": True}
    return snapshot, white, request, analysis


def test_returns_risk_and_costs_determine_unequal_positive_weights():
    s,w,r,a = example()
    p = build_plan(s,w,r,a)
    rows = p["positions"]
    assert .05 > rows[0]["target_weight"] > rows[1]["target_weight"] > rows[2]["target_weight"] >= .001
    assert all(0 <= x["weight_lower"] <= x["target_weight"] <= x["weight_upper"] <= .05 for x in rows)
    assert p["stock_weight"]+p["cash_weight"] == pytest.approx(1)
    assert sum(x["projected_weight"] for x in rows)+p["projected_cash_weight"]+p["simulated_execution_cost"] == pytest.approx(1)
    higher_risk = deepcopy(a)
    higher_risk["covariance"][0][0] *= 2
    assert build_plan(s,w,r,higher_risk)["positions"][0]["target_weight"] < rows[0]["target_weight"]
    r["buy_cost"] = .0015
    assert build_plan(s,w,r,a)["positions"][0]["target_weight"] < rows[0]["target_weight"]
    assert p["review"]["strategy_backtest_passed"] is False


@pytest.mark.parametrize("edit,code", [
    (lambda s,w,r,a: r.pop("minimum_weight"), "MINIMUM_WEIGHT_REQUIRED"),
    (lambda s,w,r,a: r.update(minimum_weight=0), "INVALID_ALLOCATION_PARAMETER"),
    (lambda s,w,r,a: r.update(market_budget=0), "MINIMUM_BUDGET_INFEASIBLE"),
    (lambda s,w,r,a: s["rows"][1].update(score=None,signal_valid=False), "SELECTED_SCORE_MISSING"),
    (lambda s,w,r,a: a["risk_codes"].pop(), "RISK_HISTORY_MISSING"),
    (lambda s,w,r,a: a.update(snapshot_id="other"), "ALLOCATION_EVIDENCE_MISMATCH"),
    (lambda s,w,r,a: r.update(industries={"A":"false_sector"}), "INDUSTRY_OVERRIDE_CONFLICT"),
    (lambda s,w,r,a: r.update(minimum_weight=.05,volatility_cap=.001), "ALLOCATION_CONSTRAINT_INFEASIBLE"),
])
def test_failures_never_drop_selected_stocks_or_fall_back_to_equal(edit,code):
    args = example(); edit(*args)
    with pytest.raises(AllocationError) as exc:
        build_plan(*args)
    assert exc.value.code == code


def test_no_evidence_and_no_holdings_are_distinct():
    s,w,r,a = example()
    with pytest.raises(AllocationError,match="证据"):
        build_plan(s,w,r)
    r["holdings"] = None
    p = build_plan(s,w,r,a)
    assert p["estimated_cost"] is None and p["projected_cash_weight"] is None
    assert all(x["current_weight"] is None and x["target_weight"] > 0 for x in p["positions"])


def test_blocked_old_holding_remains_on_balance_sheet_and_cannot_fund_new_buys():
    s,w,r,a = example()
    w["codes"] = ["A", "B"]
    r.update(market_budget=.03,holdings=[{"code":"C", "current_weight":.03, "can_sell":False}])
    p = build_plan(s,w,r,a)
    rows = {x["code"]:x for x in p["positions"]}
    assert rows["C"]["target_weight"] == 0 and rows["C"]["projected_weight"] == .03
    assert all(rows[c]["target_weight"] > 0 and rows[c]["projected_weight"] == 0 for c in w["codes"])
    assert p["status"] == "EXECUTION_INCOMPLETE"
    assert p["projected_cash_weight"] == pytest.approx(.97)


def test_lot_rounding_can_block_positive_target_but_never_claim_success():
    s,w,r,a = example()
    r["account_value"] = 10000.
    p = build_plan(s,w,r,a)
    assert all(x["target_weight"] > 0 and x["projected_weight"] == 0 for x in p["positions"])
    assert all(x["delta_quantity"] == 0 for x in p["orders_preview"])
    assert p["simulated_execution_cost"] == 0 and p["status"] == "EXECUTION_INCOMPLETE"


def test_minimum_commission_only_charged_for_actual_nonzero_orders():
    spec = cost_spec({"cost_model":{"commission_rate":.00025,"slippage_rate":.0005,"minimum_commission":5}})
    c = transaction_cost([.02,0,-.03],spec,10000.)
    assert c["minimum_commission_extra"] == pytest.approx((5-.05+5-.075)/10000)
    assert c["total"] == pytest.approx((10+(.02+.03)*10000*.0005)/10000)


def test_unchanged_holdings_do_not_need_cash_reserved_for_nonexistent_orders():
    from strategy_manager.portfolio_optimization import PortfolioProblem
    s,w,r,a=example()
    r.update(holdings=[{"code":c,"current_weight":.001} for c in w["codes"]],account_value=1000.,
             cost_model={"commission_rate":.00025,"minimum_commission":500.})
    for row in a["expected"].values(): row["expected_return"]=-.01
    problem=PortfolioProblem(s,w,r,a)
    result,_=problem.solve()
    assert result == pytest.approx(problem.current,abs=1e-7)
    assert problem.metrics(result)["transaction_cost"] < 1e-9


def test_zero_holding_quantity_may_be_omitted_and_blocked_stress_is_numeric():
    s,w,r,a=example()
    r.update(holdings=[{"code":"A","current_weight":0.,"quantity":None}],account_value=1e6)
    p=build_plan(s,w,r,a)
    assert p["orders_preview"] is not None
    assert p["stress_checks"]["all_buys_blocked"]["cash_weight"]==1.
    assert all(v==0. for v in p["stress_checks"]["all_buys_blocked"]["projected_weights"].values())


def test_no_trade_preserves_positive_target_and_flags_unfulfilled_selection():
    s,w,r,a = example()
    for item in a["expected"].values(): item["expected_return"] = -.01
    p = build_plan(s,w,r,a)
    assert p["status"] == "NO_TRADE_REQUEST_UNSATISFIED"
    assert all(x["target_weight"] > 0 and x["projected_weight"] == 0 for x in p["positions"])


def test_deadband_target_is_also_preserved_in_execution():
    s,w,r,a = example()
    first = build_plan(s,w,r,a)
    r["holdings"] = [{"code":x["code"],"current_weight":x["target_weight"]} for x in first["positions"]]
    a["expected"]["A"]["expected_return"] += .0001
    p = build_plan(s,w,r,a)
    assert all(x["target_weight"] == pytest.approx(x["projected_weight"],abs=1e-7) for x in p["positions"])


@pytest.fixture(scope="module")
def rich_bundle(tmp_path_factory):
    from scripts.build_optimizer_demo import build_demo
    from web.portfolio_service import PortfolioService
    root = tmp_path_factory.mktemp("optimizer")
    manifest = build_demo(root/"offline_demo_scores",root/"offline_demo_inputs")
    service = PortfolioService(workspace=root,default_bundles=())
    b = service.import_bundle(manifest["score_manifest"])
    snapshot = service.get_scores(b["bundle_id"],manifest["date"],manifest["fold_id"])
    path = root/"offline_demo_inputs/manifest.json"
    result = service.import_analysis(str(path),snapshot["snapshot_id"])
    payload = json.loads((path.parent/"research_inputs.json").read_text(encoding="utf-8"))
    return service,snapshot,result,payload,path


def test_real_calibration_and_covariance_have_mature_time_isolated_history(rich_bundle):
    service,s,a,payload,path = rich_bundle
    assert a["available"] and a["calibrator"]["date_count"] == 504 and a["risk_period_count"] == 100
    assert a["calibrator"]["slope"] > 0 and len(a["expected"]) == 300
    full = service.find_analysis(s["snapshot_id"])
    assert np.linalg.eigvalsh(full["covariance"]).min() >= 0
    assert service.import_analysis(str(path),s["snapshot_id"])["analysis_id"] == a["analysis_id"]
    other=deepcopy(s); other["snapshot_id"]="other_immutable_snapshot"
    rebound=prepare_inputs(payload,other)
    assert rebound["analysis_id"] != a["analysis_id"]
    assert rebound["source_sha256"] == a["source_sha256"]


def test_all_300_selected_have_positive_targets_under_joint_constraints(rich_bundle):
    service,s,_,_,_=rich_bundle
    white=service.save_whitelist(s["snapshot_id"],"all_300_engineering_test",[row["code"] for row in s["rows"]])
    plan=service.create_plan({"whitelist_id":white["whitelist_id"],"whitelist_version":1,"minimum_weight":.001})
    assert len(plan["positions"])==300
    assert all(.001-1e-8 <= row["target_weight"] <= .05+1e-8 for row in plan["positions"])
    assert max(plan["industry_weights"].values()) <= .30+1e-8
    assert plan["objective"]["expected_volatility"] <= .12+1e-8
    assert plan["stock_weight"]+plan["cash_weight"] == pytest.approx(1.)
    assert all(.001 <= row["weight_lower"] <= row["target_weight"] <= row["weight_upper"] <= .05
               for row in plan["positions"])


def test_screenshot_seven_cap_saturated_names_have_real_bands_and_parameter_effects(rich_bundle):
    service, s, _, _, _ = rich_bundle
    codes = ["SYNTHETIC_264", "SYNTHETIC_061", "SYNTHETIC_020", "SYNTHETIC_012",
             "SYNTHETIC_220", "SYNTHETIC_276", "SYNTHETIC_114"]
    white = service.save_whitelist(s["snapshot_id"], "screenshot_regression", codes)
    analysis = service.find_analysis(s["snapshot_id"])
    request = {"minimum_weight": .01, "risk_aversion": .71, "volatility_cap": .30,
               "holdings": [], "assume_tradable": True, "no_trade_band": .003}
    base = build_plan(s, white, request, analysis)
    assert all(x["target_weight"] == pytest.approx(.05, abs=1e-7) for x in base["positions"])
    assert all(.029 < x["weight_lower"] < .033 for x in base["positions"])
    assert base["allocation_diagnostics"]["all_selected_at_stock_cap"]
    assert not base["allocation_diagnostics"]["volatility_binding"]
    assert base["allocation_diagnostics"]["risk_aversion_first_cap_release"] == pytest.approx(17.26, abs=.02)
    higher = build_plan(s, white, {**request, "risk_aversion": 30}, analysis)
    assert max(x["target_weight"] for x in higher["positions"]) < .045
    assert np.ptp([x["target_weight"] for x in higher["positions"]]) > .01
    budget = build_plan(s, white, {**request, "market_budget": .2}, analysis)
    assert budget["stock_weight"] == pytest.approx(.2, abs=1e-7)
    assert budget["allocation_diagnostics"]["budget_binding"]
    risk = build_plan(s, white, {**request, "volatility_cap": .025}, analysis)
    assert risk["objective"]["expected_volatility"] == pytest.approx(.025, abs=1e-7)
    assert risk["allocation_diagnostics"]["volatility_binding"]
    expensive = build_plan(s, white, {**request, "buy_cost": .006}, analysis)
    assert expensive["stock_weight"] < base["stock_weight"]-.05
    locked = build_plan(s, white, {**request, "minimum_weight": .05}, analysis)
    assert locked["allocation_diagnostics"]["all_targets_fixed_by_bounds"]
    assert all(x["weight_lower"] == x["weight_upper"] == .05
               and x["interval_status"] == "FIXED_BY_BOUNDS" for x in locked["positions"])


def test_conditional_interval_matches_closed_form_interior_solution():
    s, w, r, a = example()
    w["codes"] = ["A"]
    r["range_utility_tolerance_bps"] = .1
    row = build_plan(s, w, r, a)["positions"][0]
    optimum = (.003-.0001-.00076)/(2*3*.02)
    half_width = np.sqrt(.1/10000/(3*.02))
    assert row["target_weight"] == pytest.approx(optimum, abs=1e-6)
    assert row["weight_lower"] == pytest.approx(optimum-half_width, abs=1e-7)
    assert row["weight_upper"] == pytest.approx(optimum+half_width, abs=1e-7)
    wider = build_plan(s, w, {**r, "range_utility_tolerance_bps": .2}, a)["positions"][0]
    assert wider["target_weight"] == row["target_weight"]
    assert wider["weight_lower"] < row["weight_lower"] and wider["weight_upper"] > row["weight_upper"]


@pytest.mark.parametrize("mode", ["default", "budget", "risk", "fees", "unknown", "zero_tolerance"])
def test_every_point_in_band_respects_portfolio_constraints_and_true_utility(mode):
    from strategy_manager.portfolio_optimization import PortfolioProblem
    s, w, r, a = example()
    if mode == "budget": r["market_budget"] = .015
    if mode == "risk": r["volatility_cap"] = .012
    if mode == "fees":
        r.update(account_value=1e5, no_trade_band=0., holdings=[{"code":"A", "current_weight":.012}],
                 cost_model={"commission_rate":.00025, "slippage_rate":.0005, "minimum_commission":5.})
    if mode == "unknown": r["holdings"] = None
    if mode == "zero_tolerance": r["range_utility_tolerance_bps"] = 0.
    plan = build_plan(s, w, r, a)
    p = PortfolioProblem(s, w, r, a)
    rows = {x["code"]: x for x in plan["positions"]}
    target = np.array([rows[c]["target_weight"] for c in p.codes])
    for i, code in enumerate(p.codes):
        row = rows[code]
        for weight in np.linspace(row["weight_lower"], row["weight_upper"], 21):
            alternative = target.copy(); alternative[i] = weight
            assert p.feasible(alternative)
            assert p.metrics(target)["utility"]-p.metrics(alternative)["utility"] <= p.range_tolerance_bps/10000+2e-11


def test_minimum_commission_jump_cannot_be_bridged_by_a_recommended_band():
    from strategy_manager.portfolio_optimization import PortfolioProblem
    from strategy_manager.portfolio_explanations import conditional_weight_ranges
    s, w, r, a = example()
    w["codes"] = ["A"]
    r.update(account_value=10000., holdings=[{"code":"A", "current_weight":.02}],
             range_utility_tolerance_bps=.1, cost_model={"minimum_commission":50.})
    p = PortfolioProblem(s, w, r, a)
    low, high, status = conditional_weight_ranges(p, p.current)
    assert high[0]-low[0] < 1e-8
    assert status == ["NO_FEASIBLE_WIDTH"]


def test_deadband_frozen_holding_has_explicit_fixed_interval():
    s, w, r, a = example()
    first = build_plan(s, w, r, a)
    r["holdings"] = [{"code":x["code"], "current_weight":x["target_weight"]} for x in first["positions"]]
    second = build_plan(s, w, r, a)
    frozen = second["allocation_diagnostics"]["fixed_by_deadband"]
    assert frozen
    for x in second["positions"]:
        if x["code"] in frozen:
            assert x["weight_lower"] == x["target_weight"] == x["weight_upper"]
            assert x["interval_status"] == "FIXED_BY_DEADBAND"


@pytest.mark.parametrize("edit,code", [
    (lambda p: p["calibration"]["label_end_at"].__setitem__(0,p["calibration_frozen_at"]),"CALIBRATION_TIME_LEAKAGE"),
    (lambda p: p["calibration"]["forecast_fit_end_at"].__setitem__(0,p["calibration_frozen_at"]),"CALIBRATION_TIME_LEAKAGE"),
    (lambda p: p["calibration"]["members"][0].__setitem__(0,False),"INCOMPLETE_CALIBRATION_UNIVERSE"),
    (lambda p: p["calibration"]["scores"][0].__setitem__(0,True),"INVALID_RESEARCH_MATRIX"),
    (lambda p: p["risk"]["end_at"].__setitem__(0,p["risk"]["start_at"][0]),"INVALID_RISK_PERIODS"),
    (lambda p: p["market"]["SYNTHETIC_000"].update(mean_amount_20d=-1),"INVALID_MARKET_INPUT"),
    (lambda p: p["current_scores"].update(SYNTHETIC_000=123),"RESEARCH_INPUT_SCORE_MISMATCH"),
])
def test_research_input_rejects_time_leaks_partial_pools_and_invalid_values(rich_bundle,edit,code):
    _,snapshot,_,payload,_ = rich_bundle
    payload = deepcopy(payload); edit(payload)
    with pytest.raises(AllocationError) as exc: prepare_inputs(payload,snapshot)
    assert exc.value.code == code


def test_reviewed_api_persists_analysis_plan_and_forward_events(rich_bundle):
    from web.portfolio_api import router,get_portfolio_service
    from web.portfolio_service import PortfolioService
    service,s,a,_,_ = rich_bundle
    app=FastAPI(); app.include_router(router); app.dependency_overrides[get_portfolio_service]=lambda:service
    prefix="/api/v2/offline-portfolio"
    with TestClient(app) as c:
        white=c.post(prefix+"/whitelists",json={"snapshot_id":s["snapshot_id"],"name":"test", "codes":[s["rows"][0]["code"],s["rows"][-1]["code"]]}).json()
        request={"whitelist_id":white["whitelist_id"],"whitelist_version":1,"holdings":[],"minimum_weight":.001,"analysis_id":a["analysis_id"],"range_utility_tolerance_bps":.2}
        response=c.post(prefix+"/plans",json=request)
        assert response.status_code == 200,response.text
        plan=response.json(); uri=prefix+"/plans/"+plan["plan_id"]
        assert plan["range_utility_tolerance_bps"] == .2
        assert plan["allocation_diagnostics"]["messages"]
        assert c.post(prefix+"/plans",json={**request,"range_utility_tolerance_bps":-1}).status_code == 422
        assert c.get(uri).json() == plan
        assert c.post(uri+"/events",json={"kind":"adopted","observed_at":s["available_at"]}).status_code == 422
        assert c.post(uri+"/events",json={"kind":"adopted","observed_at":datetime.now(timezone.utc).isoformat(),"note":"synthetic"}).status_code == 200
        assert len(c.get(uri+"/events").json()["events"]) == 1
        assert c.get(uri).json() == plan
        request["minimum_weight"]=0
        assert c.post(prefix+"/plans",json=request).status_code == 422
    restarted=PortfolioService(workspace=service.workspace,default_bundles=())
    assert restarted.get_plan(plan["plan_id"]) == plan
    assert restarted.find_analysis(s["snapshot_id"])["analysis_id"] == a["analysis_id"]
    assert len(restarted.list_forward_events(plan["plan_id"])["events"]) == 1


def replay_fixture():
    import pandas as pd
    s,w,r,a=example()
    a["calibrator"]={"frozen_at":"2019-12-01T13:00:00Z"}
    a["available_at"]=s["available_at"]
    dates=pd.bdate_range("2020-01-03",periods=11,tz="UTC")+pd.Timedelta(hours=1,minutes=30)
    first={"snapshot":s,"whitelist":w,"analysis":a,"selection_kind":"synthetic_policy",
           "selection_available_at":s["available_at"],"settlements":[
               {"start_at":dates[i].isoformat(),"end_at":dates[i+1].isoformat(),
                "returns":{"A":.01,"B":-.005,"C":.001}} for i in range(5)]}
    second=deepcopy(first)
    second["snapshot"]["snapshot_id"]="snapshot_next"
    second["snapshot"]["date"]="2020-01-09"
    second["snapshot"]["available_at"]="2020-01-09T13:00:00Z"
    second["analysis"]["available_at"]=second["snapshot"]["available_at"]
    second["analysis"]["snapshot_id"]="snapshot_next"
    second["whitelist"]["snapshot_id"]="snapshot_next"
    second["settlements"]=[{"start_at":dates[i].isoformat(),"end_at":dates[i+1].isoformat(),
                             "returns":{"A":.01,"B":-.005,"C":.001}} for i in range(5,10)]
    return [first,second],r


def test_replay_uses_realized_holdings_and_never_future_returns_to_allocate():
    from strategy_manager.portfolio_validation import replay,compare_replays
    frames,r=replay_fixture()
    first=replay(frames,r)
    modified=deepcopy(frames)
    for day in modified[0]["settlements"]: day["returns"]["A"]=-.1
    other=replay(modified,r)
    assert first["ledger"][0]["projected_weights"] == other["ledger"][0]["projected_weights"]
    assert first["ledger"][4]["nav"]>other["ledger"][4]["nav"]
    for row in first["ledger"]:
        assert row["cash"]/row["nav"]+row["stock_exposure"] == pytest.approx(1)
    report=compare_replays([first],5)
    assert report["status"]=="INSUFFICIENT_EVIDENCE" and report["production_allowed"] is False
    assert "five_outer_folds" in report["missing_evidence"]


@pytest.mark.parametrize("edit,code",[
    (lambda f: f[1]["analysis"]["calibrator"].update(slope=2),"OUTER_CALIBRATOR_CHANGED"),
    (lambda f: f[1]["settlements"][0].update(start_at="2020-01-11T01:30:00Z"),"UNPRICED_REPLAY_GAP"),
    (lambda f: f[0]["settlements"][0]["returns"].pop("A"),"REPLAY_MISSING_HELD_RETURN"),
])
def test_replay_rejects_recalibrating_outer_fold_and_unpriced_returns(edit,code):
    from strategy_manager.portfolio_validation import replay
    frames,r=replay_fixture(); edit(frames)
    with pytest.raises(AllocationError) as exc: replay(frames,r)
    assert exc.value.code==code


def test_delisting_loss_of_a_blocked_unselected_old_holding_is_not_erased():
    from strategy_manager.portfolio_validation import replay
    frames,r=replay_fixture()
    frames[0]["analysis"]["expected"]["C"]["expected_return"] = .003
    frames[1]["whitelist"]["codes"]=["A","B"]
    frames[1]["tradability"]={"C":{"can_sell":False}}
    frames[1]["settlements"][0]["returns"]={"A":0.,"B":0.,"C":-1.}
    result=replay(frames,r)
    assert result["ledger"][5]["projected_weights"]["C"]>0
    assert result["ledger"][5]["net_return"]<-.001


def test_statistics_require_same_dates_ids_and_account_for_initial_drawdown():
    from strategy_manager.portfolio_validation import compare_replays,performance,METHODS,COST_MULTIPLIERS
    import pandas as pd
    rng=np.random.default_rng(6)
    dates=pd.bdate_range("2018-01-01",periods=260)
    rows=[{"date":str(day.date()),"fold_id":str(i//52),"net_return":float(rng.normal(.0005,.01)),
           "stock_exposure":.5,"trade_cost":.0001,"snapshot_id":str(i),"analysis_id":str(i),
           "whitelist_id":"test","whitelist_version":1,"net_excess_return":float(rng.normal(.0005,.01))}
          for i,day in enumerate(dates)]
    rows[0]["net_return"]=-.1
    assert performance(rows)["max_drawdown"]>=.1-1e-12
    runs=[{"method":m,"cost_multiplier":c,"phase":0,"ledger":deepcopy(rows),"research_eligible":False}
          for m in METHODS for c in COST_MULTIPLIERS]
    result=compare_replays(runs,1)
    assert result["paired_95pct_interval"]==[0.,0.]
    assert not result["comparison_passed"]
    runs[4]["ledger"][0]["whitelist_id"]="other"
    with pytest.raises(AllocationError,match="同一输入"): compare_replays(runs,1)
