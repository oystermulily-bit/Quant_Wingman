"""Build the CSI 300 v2 research snapshot from the immutable v1 raw tree.

Does not fetch, does not start RD-Agent, and does not read Holdout metrics.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.hs300.config import RAW_SNAPSHOT_ROOT, SNAPSHOT_ROOT
from data_pipeline.hs300.standardize import freeze_panel_snapshot


def _ensure_raw_junction(snapshot_root: Path, raw_src: Path) -> None:
    dest = snapshot_root / "raw"
    if dest.exists():
        return
    snapshot_root.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(raw_src)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not dest.exists():
        raise RuntimeError(
            "failed to junction v2/raw -> v1/raw: "
            f"{completed.stdout} {completed.stderr}"
        )


def main() -> int:
    v1_status = (RAW_SNAPSHOT_ROOT / "STATUS.txt").read_text(encoding="utf-8")
    if "RAW_SNAPSHOT_COMPLETE" not in v1_status:
        print("v1 raw snapshot is not complete; refuse to build v2")
        return 2
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_raw_junction(SNAPSHOT_ROOT, RAW_SNAPSHOT_ROOT / "raw")
    src_gap = RAW_SNAPSHOT_ROOT / "coverage_gap.json"
    dest_gap = SNAPSHOT_ROOT / "coverage_gap.json"
    if src_gap.is_file() and not dest_gap.is_file():
        shutil.copy2(src_gap, dest_gap)
    (SNAPSHOT_ROOT / "STATUS.txt").write_text(
        "RAW_SNAPSHOT_COMPLETE\n", encoding="utf-8"
    )
    (SNAPSHOT_ROOT / "RAW_SOURCE.txt").write_text(
        f"raw -> {RAW_SNAPSHOT_ROOT / 'raw'}\nDo not re-fetch. Do not mutate v1 raw.\n",
        encoding="utf-8",
    )
    manifest = freeze_panel_snapshot(SNAPSHOT_ROOT)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
