"""CLI for D21-v2 Stage 4R. Does not overwrite D21-v1 or unseal Holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_d21_v2.gate import Stage4RFeasibilityRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 D21-v2 Stage 4R（D18-v2 + A–D 消融）")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--stage3r", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    args = parser.parse_args()
    report = Stage4RFeasibilityRunner().run(
        snapshot_dir=Path(args.snapshot),
        stage3r_dir=Path(args.stage3r),
        output_dir=Path(args.output),
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(
        {
            "status": report.get("status"),
            "horizon_conclusion": report.get("horizon_conclusion"),
            "rd_agent_allowed": report.get("rd_agent_allowed"),
            "system_status": report.get("system_status"),
            "d21_v1_gate_unchanged": report.get("d21_v1_gate_unchanged"),
        },
        indent=2,
        ensure_ascii=False,
    ))
    status = report.get("status")
    if status in {"HORIZON_FEASIBLE_5D_V2", "D21_V2_FAILED", "INSUFFICIENT_EVIDENCE"}:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
