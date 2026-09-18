from __future__ import annotations

import math
import random

import pytest

from strategy_manager.offline_allocation import build_reference_plan as build_plan


def inputs(n=300, selected=5):
    snapshot = {"snapshot_id": "snap", "data_role": "synthetic", "production_allowed": False,
                "date": "2020-01-02", "fold_id": "test", "rows": [
                    {"code": f"SYN_{i:03}", "score": -i*.01, "rank": i+1, "signal_valid": True}
                    for i in range(n)]}
    whitelist = {"snapshot_id": "snap", "whitelist_id": "list", "version": 1,
                 "codes": [r["code"] for r in snapshot["rows"][:selected]]}
    return snapshot, whitelist


def test_only_selected_five_receive_5_percent_no_holdings_no_trade_claim():
    snap, white = inputs()
    result = build_plan(snap, white, {})
    assert result["stock_weight"] == pytest.approx(.25)
    assert result["cash_weight"] == pytest.approx(.75)
    assert all(p["target_weight"] == .05 for p in result["positions"])
    assert all(p["delta_weight"] is None and p["action"] == "TARGET_ONLY" for p in result["positions"])
    assert result["estimated_cost"] is None
    assert result["production_allowed"] is False


def test_missing_industry_is_one_capped_group_not_300_independent_industries():
    snap, white = inputs(selected=30)
    result = build_plan(snap, white, {})
    assert result["stock_weight"] == pytest.approx(.3)
    assert result["cash_weight"] == pytest.approx(.7)


def test_invalid_score_remains_in_plan_with_zero_target():
    snap, white = inputs()
    snap["rows"][0].update(score=None, rank=None, signal_valid=False)
    result = build_plan(snap, white, {"holdings": []})
    p = next(p for p in result["positions"] if p["code"] == "SYN_000")
    assert p["target_weight"] == 0
    assert "INVALID_SCORE_NO_POSITIVE_TARGET" in p["reason_codes"]
    assert result["cash_weight"] == pytest.approx(.8)


def test_full_budget_reserves_fees_and_simulation_conserves_equity():
    snap, white = inputs(selected=30)
    industries = {c: f"sector_{i//5}" for i, c in enumerate(white["codes"])}
    result = build_plan(snap, white, {"holdings": [], "industries": industries, "assume_tradable": True})
    assert result["stock_weight"] < 1
    assert result["cash_weight"] >= result["estimated_cost"]-1e-10
    assert result["projected_cash_weight"] >= 0
    assert sum(p["projected_weight"] for p in result["positions"])+result["projected_cash_weight"]+result["simulated_execution_cost"] == pytest.approx(1)


def test_blocked_old_holding_not_erased_and_buys_cannot_spend_locked_proceeds():
    snap, white = inputs()
    result = build_plan(snap, white, {"holdings": [{"code": "OLD", "current_weight": 1,
                                                  "can_sell": False}], "assume_tradable": True})
    old = next(p for p in result["positions"] if p["code"] == "OLD")
    assert old["target_weight"] == 0 and old["projected_weight"] == 1
    assert old["execution_status"] == "BLOCKED_SELL"
    assert all(p["projected_weight"] == 0 for p in result["positions"] if p["selected"])
    assert result["projected_cash_weight"] == 0
    assert "PROJECTED_HOLDINGS_EXCEED_TARGET_LIMITS_DUE_TO_BLOCKS" in result["reason_codes"]


def test_missing_execution_information_never_claims_filled():
    snap, white = inputs()
    result = build_plan(snap, white, {"holdings": []})
    assert result["projected_cash_weight"] is None
    assert all(p["projected_weight"] is None for p in result["positions"])


@pytest.mark.parametrize("budget,old_industry,new_industry,max_buys", [
    (.2, "old_sector", "new_sector", 0), (1, "same", "same", .1),
])
def test_blocked_holding_consumes_budget_and_industry_capacity(budget, old_industry, new_industry, max_buys):
    snap, white = inputs(selected=6)
    result = build_plan(snap, white, {"market_budget": budget,
        "holdings": [{"code": "OLD", "current_weight": .2, "can_sell": False, "industry": old_industry}],
        "industries": {c: new_industry for c in white["codes"]}, "assume_tradable": True})
    assert sum(p["executable_delta_weight"] for p in result["positions"] if p["selected"]) == pytest.approx(max_buys)
    assert "BUYS_LIMITED_BY_RETAINED_HOLDING_RISK" in result["reason_codes"]


def test_provided_empty_holdings_enables_buy_delta():
    snap, white = inputs()
    result = build_plan(snap, white, {"holdings": [], "account_value": 100000})
    assert result["positions"][0]["delta_weight"] == .05
    assert result["positions"][0]["delta_value"] == 5000
    assert result["estimated_cost"] == pytest.approx(.25*.00076)


@pytest.mark.parametrize("plan_input", [
    {"market_budget": float("nan")}, {"market_budget": -1}, {"market_budget": True},
    {"account_value": 0}, {"account_value": 10**400}, {"assume_tradable": "yes"},
    {"holdings": [{"code": "X", "current_weight": .7}, {"code": "Y", "current_weight": .4}]},
    {"holdings": [{"code": "X", "current_weight": .1}, {"code": "X", "current_weight": .1}]},
    {"tradability": {"SYN_000": {"can_buy": "false"}}},
])
def test_invalid_inputs_refused(plan_input):
    snap, white = inputs()
    with pytest.raises(ValueError):
        build_plan(snap, white, plan_input)


def test_wrong_snapshot_and_empty_selection_refused():
    snap, white = inputs()
    white["snapshot_id"] = "other"
    with pytest.raises(ValueError):
        build_plan(snap, white, {})
    white.update(snapshot_id="snap", codes=[])
    with pytest.raises(ValueError):
        build_plan(snap, white, {})


def test_varied_selection_sizes_budgets_industries_preserve_constraints():
    rng = random.Random(20260916)
    for n in (1, 5, 19, 30, 100, 300):
        snap, white = inputs(selected=n)
        sectors = {c: f"sector{rng.randrange(8)}" for c in white["codes"]}
        for budget in (0, .07, .25, .63, 1):
            result = build_plan(snap, white, {"market_budget": budget, "industries": sectors,
                                             "holdings": [], "assume_tradable": True})
            sector_weights = {}
            for p in result["positions"]:
                assert math.isfinite(p["target_weight"]) and 0 <= p["target_weight"] <= .05+1e-10
                sector_weights[p["industry"]] = sector_weights.get(p["industry"], 0)+p["target_weight"]
            assert all(w <= .3+1e-10 for w in sector_weights.values())
            assert result["stock_weight"] <= budget+1e-10
            assert result["stock_weight"]+result["cash_weight"] == pytest.approx(1)
            assert result["projected_cash_weight"] >= 0


def test_selection_order_does_not_change_allocation():
    snap, white = inputs(selected=30)
    a = build_plan(snap, white, {})
    white["codes"].reverse()
    b = build_plan(snap, white, {})
    assert a == b
