"""Read-only S2 admission check. Does not load market data or start training."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factor_research.joint_protocol import FrozenJointPlan


def check_plan(path: Path) -> dict:
    blockers = []
    try:
        plan = FrozenJointPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
        plan.validate(mode="development")
    except (ValueError, TypeError, OSError) as exc:
        blockers.append(str(exc))
    if importlib.util.find_spec("lightgbm") is None:
        blockers.append("required LightGBM backend is not installed")
    return {"status": "PLAN_CHECK_BLOCKED" if blockers else "PLAN_STATIC_CHECK_PASSED",
            "blockers": blockers, "read_only": True, "training_started": False,
            "research_go": False, "holdout_read": False, "sota_promoted": False,
            "production_allowed": False,
            "note": "Static plan check only; runner still verifies actual data/split/config/code identities. "
                    "Economic execution and risk validation remain required."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    result = check_plan(args.plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
