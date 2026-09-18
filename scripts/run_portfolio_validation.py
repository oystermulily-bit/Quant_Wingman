"""Replay registered offline inputs and persist a non-production comparison.

Manifest schema: {horizon, request, phases: {"0": [{snapshot_id, analysis_id,
whitelist_id, whitelist_version, selection_kind, selection_available_at,
tradability, settlements: [{start_at,end_at,returns,cash_return}]}]}}.
All referenced snapshots/analyses/whitelists must already be in the local store.
Use --synthetic-demo for the five-day engineering smoke replay. That example
deliberately reports insufficient research evidence, not five-fold validation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT/".demo_runtime", ROOT): sys.path.insert(0,str(folder))

from strategy_manager.portfolio_validation import replay,compare_replays,METHODS,COST_MULTIPLIERS
from web.portfolio_service import PortfolioService,_json,_now,_plain_json


def run_registered(service, manifest):
    if manifest.get("production_allowed") is not False or manifest.get("holdout_read") is not False:
        raise ValueError("Explicit offline and no-Holdout flags required")
    runs=[]; snapshots=set(); analyses=set()
    for phase,items in sorted(manifest["phases"].items()):
        frames=[]
        for item in items:
            snapshot=service.get_snapshot(item["snapshot_id"])
            white=service.get_whitelist(item["whitelist_id"],item["whitelist_version"])
            analysis=service.find_analysis(snapshot["snapshot_id"],item["analysis_id"])
            if analysis["horizon"]!=manifest["horizon"]: raise ValueError("Replay horizon differs from frozen model")
            snapshots.add(snapshot["snapshot_id"]); analyses.add(analysis["analysis_id"])
            frames.append({**item,"snapshot":snapshot,"whitelist":white,"analysis":analysis})
        for method in METHODS:
            for multiplier in COST_MULTIPLIERS:
                runs.append(replay(frames,manifest["request"],method=method,cost_multiplier=multiplier,phase=int(phase)))
    report=compare_replays(runs,manifest["horizon"])
    report.update({"created_at":_now(),"input_sha256":sha256(_json(manifest).encode()).hexdigest(),
                   "snapshot_ids":sorted(snapshots),"analysis_ids":sorted(analyses)})
    report["validation_id"]="validation_"+sha256(_json(report).encode()).hexdigest()[:24]
    return report,runs


def synthetic_manifest(service):
    from scripts.build_optimizer_demo import synthetic_inputs
    report,scores,payload,settlements=synthetic_inputs(include_settlements=True)
    meta=next(service.get_bundle(b["bundle_id"]) for b in service.list_bundles()["bundles"]
              if b["label"]=="offline_demo_optimizer_300_20260917")
    snapshot=service.get_scores(meta["bundle_id"],payload["date"],payload["fold_id"])
    if snapshot["fingerprints"]!=report["fingerprints"]: raise ValueError("Synthetic data version mismatch")
    analysis=service.find_analysis(snapshot["snapshot_id"])
    selected=[snapshot["rows"][i]["code"] for i in range(0,300,10)]
    white=service.save_whitelist(snapshot["snapshot_id"],"合成回放测试：固定每十名选一只",selected)
    return {"production_allowed":False,"holdout_read":False,"horizon":5,
            "request":{"minimum_weight":.001,"market_budget":.5,"assume_tradable":True},
            "phases":{"0":[{"snapshot_id":snapshot["snapshot_id"],"analysis_id":analysis["analysis_id"],
                             "whitelist_id":white["whitelist_id"],"whitelist_version":1,
                             "selection_kind":"synthetic_policy","selection_available_at":snapshot["available_at"],
                             "settlements":settlements}]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest",type=Path)
    source.add_argument("--synthetic-demo",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    destination=args.output.resolve()
    if (not destination.is_relative_to(ROOT) or not destination.name.startswith("offline_demo_")
            or destination.exists() or any("holdout" in p.lower() or "sealed" in p.lower() for p in destination.parts)):
        raise ValueError("Output must be a fresh offline_demo_ directory inside workspace")
    service=PortfolioService()
    manifest=(synthetic_manifest(service) if args.synthetic_demo else
              _plain_json(service._read(service._safe_path(args.manifest,manifest=True),50*1024*1024)))
    report,runs=run_registered(service,manifest)
    destination.mkdir(parents=True)
    (destination/"manifest.json").write_text(_json(manifest),encoding="utf-8")
    (destination/"validation_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    (destination/"replay_ledgers.json").write_text(_json(runs),encoding="utf-8")
    service.save_validation_report(report)
    print(json.dumps({"validation_id":report["validation_id"],"status":report["status"],
                      "replays":len(runs),"days_per_replay":len(runs[0]["ledger"]),
                      "report":str(destination/"validation_report.json")},ensure_ascii=False))


if __name__=="__main__": main()
