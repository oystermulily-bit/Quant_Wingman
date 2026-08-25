"""Build standardized CSI 300 tables from a frozen raw snapshot. Not used in training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.hs300.config import SNAPSHOT_ROOT
from data_pipeline.hs300.standardize import freeze_panel_snapshot


def main() -> int:
    status = (SNAPSHOT_ROOT / "STATUS.txt").read_text(encoding="utf-8")
    if "RAW_SNAPSHOT_COMPLETE" not in status:
        print("raw snapshot is not complete; refuse to freeze standardized tables")
        return 2
    manifest = freeze_panel_snapshot(SNAPSHOT_ROOT)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
