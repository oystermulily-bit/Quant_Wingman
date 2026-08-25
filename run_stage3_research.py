"""CLI for the confirmed stage-3 minimum HS300 research loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_stage3.runner import Stage3ResearchRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 quant_w1ngman 阶段3点时沪深300最小研究闭环"
    )
    parser.add_argument("--snapshot", required=True, help="冻结快照目录（含manifest.json）")
    parser.add_argument("--output", required=True, help="阶段3输出目录")
    parser.add_argument("--expected-members", type=int, default=300)
    parser.add_argument(
        "--required-start",
        default="2014-01-02",
        help="正式覆盖起点；测试时可传空字符串关闭",
    )
    args = parser.parse_args()
    report = Stage3ResearchRunner().run(
        Path(args.snapshot),
        Path(args.output),
        expected_members=args.expected_members,
        required_start=args.required_start or None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["status"] == "STAGE3_MINIMUM_LOOP_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
