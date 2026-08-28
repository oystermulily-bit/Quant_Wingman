"""CLI for the frozen stage-4 Development OOF feasibility gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_stage4.runner import Stage4FeasibilityRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="运行阶段4配对Bootstrap / Holm / GO门禁")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--stage3", required=True, help="阶段3输出目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-members", type=int, default=300)
    parser.add_argument("--required-start", default="2014-01-02")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="正式运行保持2000；单测可覆盖",
    )
    args = parser.parse_args()
    report = Stage4FeasibilityRunner().run(
        snapshot_dir=Path(args.snapshot),
        stage3_dir=Path(args.stage3),
        output_dir=Path(args.output),
        expected_members=args.expected_members,
        required_start=args.required_start or None,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(
        {
            "status": report.get("status"),
            "horizon_conclusion": report.get("horizon_conclusion"),
            "allowed_horizons": report.get("allowed_horizons"),
            "rd_agent_allowed": report.get("rd_agent_allowed"),
            "system_status": report.get("system_status"),
        },
        indent=2,
        ensure_ascii=False,
    ))
    status = report.get("status")
    if status == "RESEARCH_GATE_PASSED":
        return 0
    if status == "KEEP_1D_BASELINE":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
