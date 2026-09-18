from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_research.demo_pipeline import DemoConfig, OfflineFactorDemo
from factor_research.demo_formulas import FormulaSpec
from factor_research.group_evaluator import SelectionFold
from factor_research.joint_runner import JointPanel, OuterFold
from factor_research.lgbm_baseline import FixedLightGBMBaseline


def inputs():
    rng = np.random.default_rng(77)
    n, t = 16, 280
    x = {k: rng.normal(size=(n, t)) for k in ("A", "B", "C")}
    y = x["A"]*x["B"] + .005*rng.normal(size=(n,t))
    time = pd.date_range("2014-01-01", periods=t, tz="UTC")
    stamps = np.array([v.value for v in time], dtype=np.int64)
    avail = {k: np.broadcast_to(stamps, (n,t)).copy() for k in x}
    panel = JointPanel(x,y,np.ones((n,t),dtype=bool),tuple(f"fake{i}" for i in range(n)),
                       tuple(v.isoformat() for v in time),
                       tuple(v.isoformat() for v in time+pd.Timedelta(days=5)),avail,
                       "2016-01-01T00:00:00Z", data_role="synthetic")
    inner = (SelectionFold(0,40,60,80),SelectionFold(0,70,90,110),SelectionFold(0,100,120,140))
    return panel, (OuterFold("f1",0,160,180,200,inner),OuterFold("f2",0,220,240,260,inner))


@pytest.fixture
def fake_backend(monkeypatch):
    calls=[]
    monkeypatch.setattr(FixedLightGBMBaseline,"_module",staticmethod(lambda: object()))
    monkeypatch.setattr("factor_research.demo_pipeline.version",lambda name:"test-substitute")
    def fit(self,x,y,train,valid):
        calls.append((y.copy(),train.copy(),valid.copy()))
        if x is None:
            return np.full(len(valid),y.ravel()[train].mean())
        z=np.column_stack((np.ones(y.size),x.reshape(-1,x.shape[-1])))
        weights=np.linalg.lstsq(z[train],y.ravel()[train],rcond=None)[0]
        return z[valid]@weights
    monkeypatch.setattr(FixedLightGBMBaseline,"_fit_predict",fit)
    return calls


def test_offline_demo_real_structure_without_research_go(fake_backend):
    p,f=inputs()
    report,scores=OfflineFactorDemo().run(p,f)
    assert report["status"]=="OFFLINE_DEMO_COMPLETED_NOT_VALIDATED"
    assert report["data_role"]=="synthetic"
    assert report["proposal_source"]=="offline_replay"
    assert not any(report[k] for k in ("research_go","holdout_read","sota_promoted","production_allowed","whitelist_applied","network_used"))
    assert report["expected_score_rows"]==640==len(scores)
    assert set(scores)=={"date","code","score","score_available_at","fold_id","is_member","signal_valid"}
    assert report["selected_features"]
    assert report["formulas"]
    assert report["budget"]["charged"]<=report["budget"]["reserved_upper_bound"]
    for y,train,valid in fake_backend:
        if y.shape[1] in (200,260):
            mask=np.ones(y.size,dtype=bool);mask[train]=False
            assert np.isnan(y.ravel()[mask]).all()


def test_future_outer_labels_not_used_for_selection_formula_feedback_or_fit(fake_backend):
    p,f=inputs();demo=OfflineFactorDemo()
    report,first=demo.run(p,f)
    changed=p.target.copy();changed[:,240:260]*=-100
    other,second=demo.run(replace(p,target=changed),f)
    np.testing.assert_allclose(first.score,second.score,equal_nan=True)
    assert report["formulas"]==other["formulas"]
    assert [r["selected"] for r in report["selection_runs"]]==[r["selected"] for r in other["selection_runs"]]


def test_late_outer_input_propagates_missing(fake_backend):
    p,f=inputs()
    a={k:v.copy() for k,v in p.available_at_ns.items()}
    for value in a.values(): value[0,180]+=1
    report,scores=OfflineFactorDemo().run(replace(p,available_at_ns=a),f)
    row=scores[(scores.code=="fake0")&(scores.date=="2014-06-30")]
    assert len(row)==1 and row.score.isna().all() and not row.signal_valid.any()
    assert len(scores)==640


def test_all_unhelpful_is_labelled_fallback_not_fake_pass(fake_backend,monkeypatch):
    monkeypatch.setattr(FixedLightGBMBaseline,"_fit_predict",lambda self,x,y,tr,va:np.zeros(len(va)))
    report,scores=OfflineFactorDemo().run(*inputs())
    assert all(r["demo_fallback"] for r in report["selection_runs"])
    assert any("NO_RELIABLE_SUBSET" in w for w in report["warnings"])
    assert any("OUTER_IC_NONPOSITIVE" in w for w in report["warnings"])
    assert not report["research_go"] and report["status"].endswith("NOT_VALIDATED")


def test_formula_and_joint_fit_can_recover_interaction(fake_backend,monkeypatch):
    class Source:
        def generate(self,**kw):
            assert kw["data_role"]=="development_inner"
            assert set(kw)=={"round_index","count","feedback","data_role"}
            return [FormulaSpec("product",("A","B","MUL"))]
    monkeypatch.setattr(OfflineFactorDemo,"_source",lambda self,names:Source())
    report,_=OfflineFactorDemo().run(*inputs())
    assert all(any(k.startswith("FORMULA_") for k in r["fitted"]) for r in report["selection_runs"])
    assert all(r["outer_rank_ic"]>.98 for r in report["model_evidence"]["folds"])


def test_generation_failure_is_disclosed_and_never_claims_remote_success(fake_backend,monkeypatch):
    class Broken:
        def generate(self,**kw):raise RuntimeError("private API error text")
    monkeypatch.setattr(OfflineFactorDemo,"_source",lambda self,names:Broken())
    report,_=OfflineFactorDemo().run(*inputs())
    assert any("PROPOSAL_ERROR" in w for w in report["warnings"])
    assert "private API error text" not in str(report)
    assert not report["network_used"]


@pytest.mark.parametrize("bad",["holdout","role","budget","network","postponed_holdout"])
def test_demo_does_not_bypass_safety(fake_backend,bad):
    p,f=inputs();cfg=DemoConfig()
    if bad=="holdout":p=replace(p,holdout_start=p.signal_at[240])
    elif bad=="role":p=replace(p,data_role="holdout")
    elif bad=="budget":cfg=DemoConfig(max_total_opportunities=1)
    elif bad=="postponed_holdout":p=replace(p,data_role="development",holdout_start="2025-01-01T00:00:00Z")
    else:
        with pytest.raises(ValueError):DemoConfig(proposal_mode="rd_agent")
        return
    with pytest.raises(ValueError):OfflineFactorDemo(cfg).run(p,f)
    assert not fake_backend


@pytest.mark.parametrize("horizon", [1, 3, "5", True, 5.0])
def test_h5_target_declaration_is_required(horizon):
    with pytest.raises(ValueError, match="H5"):
        DemoConfig(label_horizon_bars=horizon)


def test_model_fingerprint_tracks_underlying_implementation(fake_backend, monkeypatch):
    demo = OfflineFactorDemo()
    report, _ = demo.run(*inputs())
    monkeypatch.setattr(demo.guard, "implementation_identity", lambda: "0" * 64)
    changed, _ = demo.run(*inputs())
    assert report["fingerprints"]["model"] != changed["fingerprints"]["model"]
    assert set(report["model_evidence"]["implementation_hashes"]) == {"joint_stack", "demo_pipeline", "demo_formulas"}


def test_real_lightgbm_offline_demo():
    pytest.importorskip("lightgbm")
    report,scores=OfflineFactorDemo(DemoConfig(formula_rounds=1,proposals_per_round=2,num_boost_round=10)).run(*inputs())
    assert report["model_evidence"]["backend"]=="lightgbm"
    assert report["model_evidence"]["version"]!="test-substitute"
    assert scores.signal_valid.any() and not report["research_go"]
