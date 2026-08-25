"""Patch frozen tables and write Development labels without reading Holdout metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.hs300.config import SNAPSHOT_ROOT
from data_pipeline.hs300.industry_stats import industry_loo_context
from data_pipeline.hs300.raw_store import sha256_file
from data_pipeline.hs300_panel import HS300PanelDataManager
from research_stage3.labels import LabelRegistry
from research_stage3.protocol import DateSplitProtocol


def main() -> int:
    raw_path = SNAPSHOT_ROOT / "standardized" / "raw_request_manifest.parquet"
    frame = pd.read_parquet(raw_path)
    if "parameters_json" in frame.columns and "parameters" not in frame.columns:
        frame = frame.rename(columns={"parameters_json": "parameters"})
        frame.to_parquet(raw_path, index=False)
    manifest_path = SNAPSHOT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"]["raw_request_manifest"]
    entry["sha256"] = sha256_file(raw_path)
    entry["rows"] = int(len(frame))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    splits = json.loads((SNAPSHOT_ROOT / "splits" / "split_plan.json").read_text(encoding="utf-8"))
    if splits.get("holdout_date_hash"):
        print("locked_holdout_hash", splits["holdout_date_hash"])
    manager = HS300PanelDataManager(SNAPSHOT_ROOT)
    manager.load(raise_on_gate_failure=False)
    gate_path = SNAPSHOT_ROOT / "validation" / "panel_gate.json"
    gate_path.write_text(
        json.dumps(manager.report.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("panel_status", manager.report.status if manager.report else None)
    print("panel_rows", 0 if manager.panel is None else len(manager.panel))
    if manager.panel is None:
        return 2

    plan = DateSplitProtocol().build(manager.calendar)
    development = manager.development_panel(plan.development_dates)
    context, stock_loo = industry_loo_context(development)
    derived = SNAPSHOT_ROOT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    context.to_parquet(derived / "industry_context.parquet", index=False)
    stock_loo.to_parquet(derived / "industry_loo_stock.parquet", index=False)
    status = manager.trading_status
    if status is not None:
        status = status[status["date"].isin(pd.DatetimeIndex(plan.development_dates))]
    labels = LabelRegistry().build(
        development,
        manager.bars_for_dates(plan.development_dates),
        pd.DatetimeIndex(plan.development_dates),
        status,
    )
    labels_dir = SNAPSHOT_ROOT / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_dir / "development_labels.parquet", index=False)
    (labels_dir / "README.txt").write_text(
        "Development labels only. Holdout labels are not written.\n"
        f"holdout_start={plan.holdout_start.date().isoformat()}\n",
        encoding="utf-8",
    )
    print("development_rows", len(development))
    print("label_rows", len(labels))
    print("holdout_start", plan.holdout_start.date().isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
