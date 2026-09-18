"""CLI for the D21-v2 Stage 3R Development loop. Does not overwrite D21-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_d21_v2.runner import Stage3RResearchRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 D21-v2 Stage 3R（保留 D21-v1 证据，仅研究 H=5）"
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True, help="必须是独立目录，不得指向 v1 ledger")
    parser.add_argument("--expected-members", type=int, default=300)
    parser.add_argument("--required-start", default="2014-01-02")
    args = parser.parse_args()
    output = Path(args.output)
    report = Stage3RResearchRunner().run(
        Path(args.snapshot),
        output,
        expected_members=args.expected_members,
        required_start=args.required_start or None,
    )
    print(json.dumps(
        {
            "status": report.get("status"),
            "hypothesis_id": report.get("hypothesis_id"),
            "config_hash": report.get("config_hash"),
            "oof_slow_coverage": report.get("oof_slow_coverage"),
            "holdout_read": report.get("holdout_read"),
            "rd_agent_allowed": report.get("rd_agent_allowed"),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    ))
    return 0 if report.get("status") == "STAGE3R_D21_V2_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
