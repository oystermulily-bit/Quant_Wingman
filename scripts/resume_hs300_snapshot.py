"""Resume CSI 300 raw fetch until STATUS contains RAW_SNAPSHOT_COMPLETE."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.hs300.config import RAW_SNAPSHOT_ROOT

BUILDER = ROOT / "scripts" / "build_hs300_snapshot.py"
STATUS = RAW_SNAPSHOT_ROOT / "STATUS.txt"


def _complete() -> bool:
    if not STATUS.exists():
        return False
    return "RAW_SNAPSHOT_COMPLETE" in STATUS.read_text(encoding="utf-8")


def main() -> int:
    while not _complete():
        proc = subprocess.run(
            [sys.executable, "-u", str(BUILDER)],
            cwd=str(ROOT),
        )
        if _complete():
            return 0
        print(
            f"snapshot still incomplete after exit={proc.returncode}; resume in 5s",
            flush=True,
        )
        time.sleep(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
