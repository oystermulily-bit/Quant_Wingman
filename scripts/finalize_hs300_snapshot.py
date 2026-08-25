"""Attach remaining CSI 300 snapshot artifacts without rereading Holdout metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.hs300.config import SNAPSHOT_ROOT
from data_pipeline.hs300.finalize import finalize_snapshot


def main() -> int:
    summary = finalize_snapshot(SNAPSHOT_ROOT)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
