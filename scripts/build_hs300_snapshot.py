"""Build the CSI 300 AmazingData raw snapshot. Training must not call this."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.hs300.fetch import fetch_raw_snapshot


def main() -> int:
    report = fetch_raw_snapshot()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
