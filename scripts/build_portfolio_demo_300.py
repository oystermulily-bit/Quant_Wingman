"""Fit a new 300-name SYNTHETIC offline score bundle for page acceptance.

All names and observations are fake. This uses the existing time-isolated
offline fitter, never reads market data or Holdout, and makes no network calls.
It only writes score/evidence files; portfolio outputs belong to the consumer.
Run from the repository root with D:/anaconda/python.exe. Existing outputs
are rejected before fitting; pass --output with a new offline_demo_* name to
repeat the demonstration. The process-local backend path changes no install.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / ".demo_runtime", ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import numpy as np
import pandas as pd

from factor_research.demo_artifacts import _reject_sensitive_path, write_demo_bundle
from factor_research.demo_pipeline import DemoConfig, OfflineFactorDemo
from factor_research.group_evaluator import SelectionFold
from factor_research.joint_runner import JointPanel, OuterFold


SEED = 20260916
UNIVERSE_SIZE = 300
DEFAULT_OUTPUT = ROOT / "experiments" / "offline_demo_portfolio_300_20260916"


def synthetic_bundle() -> tuple[JointPanel, tuple[OuterFold, ...]]:
    """Fixed fake universe and original demo temporal separation, at full size."""
    rng = np.random.default_rng(SEED)
    n, t = UNIVERSE_SIZE, 320
    features = {name: rng.normal(size=(n, t)) for name in ("A", "B", "C")}
    target = (.02 * features["A"] + .03 * features["A"] * features["B"]
              + .002 * features["C"] + .005 * rng.normal(size=(n, t)))
    times = pd.date_range("2015-01-01 13:00", periods=t, freq="B", tz="UTC")
    ns = np.asarray([stamp.value for stamp in times], dtype=np.int64)
    available = {name: np.broadcast_to(ns, (n, t)).copy() for name in features}
    inner = (SelectionFold(0, 40, 60, 80), SelectionFold(0, 70, 90, 110),
             SelectionFold(0, 100, 120, 140))
    folds = (OuterFold("demo_outer_1", 0, 160, 180, 200, inner),
             OuterFold("demo_outer_2", 0, 220, 240, 260, inner))
    # Keep one explicitly unavailable member on the latest display date. It
    # remains a member; neither the target nor a missing score is imputed.
    for values in features.values():
        values[-1, folds[-1].test_end - 1] = np.nan
    panel = JointPanel(
        features, target, np.ones((n, t), dtype=bool),
        tuple(f"SYNTHETIC_{i:03d}" for i in range(n)),
        tuple(stamp.isoformat() for stamp in times),
        tuple((stamp + pd.offsets.BDay(5)).isoformat() for stamp in times),
        available, "2017-01-01T00:00:00Z", data_role="synthetic",
    )
    return panel, folds


def demo_config() -> DemoConfig:
    """Frozen small engineering budget, recorded verbatim in the manifest."""
    return DemoConfig(
        seed=SEED, formula_rounds=1, proposals_per_round=4,
        search_trials_per_round=24, num_boost_round=12,
        max_total_opportunities=128,
    )


def validate_full_grid(scores: pd.DataFrame, panel: JointPanel,
                       folds: tuple[OuterFold, ...]) -> None:
    """Reject loss/substitution of members, including the intentionally invalid one."""
    expected_codes = set(panel.codes)
    if len(expected_codes) != UNIVERSE_SIZE:
        raise ValueError("the synthetic fixture must have exactly 300 distinct names")
    expected_groups = {
        (pd.Timestamp(panel.signal_at[i]).tz_convert("Asia/Shanghai").date().isoformat(), fold.fold_id)
        for fold in folds for i in range(fold.test_start, fold.test_end)
    }
    actual = scores.copy()
    actual["date"] = pd.to_datetime(actual["date"]).dt.strftime("%Y-%m-%d")
    groups = actual.groupby(["date", "fold_id"], sort=False)
    if set(groups.groups) != expected_groups:
        raise ValueError("scores do not preserve the frozen date/fold grid")
    for _, group in groups:
        if (len(group) != UNIVERSE_SIZE or set(group["code"]) != expected_codes
                or not group["is_member"].eq(True).all()):
            raise ValueError("each date/fold must retain all 300 original members")
    last_date = pd.Timestamp(panel.signal_at[folds[-1].test_end - 1]).tz_convert("Asia/Shanghai").date().isoformat()
    missing = actual[(actual["code"] == panel.codes[-1]) & (actual["date"] == last_date)
                     & (actual["fold_id"] == folds[-1].fold_id)]
    if len(missing) != 1 or missing["signal_valid"].any() or missing["score"].notna().any():
        raise ValueError("the unavailable synthetic member must retain its missing score")


def _run_pipeline(panel: JointPanel, folds: tuple[OuterFold, ...]):
    return OfflineFactorDemo(config=demo_config()).run(panel, folds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New offline_demo_* directory; parent must already exist")
    args = parser.parse_args(argv)
    try:
        _reject_sensitive_path(args.output)
        output = args.output.resolve()
        _reject_sensitive_path(output)
        if output.exists() or args.output.is_symlink():
            raise FileExistsError("existing demo output will not be overwritten")
        if not output.name.startswith("offline_demo_") or output.name == "offline_demo_":
            raise ValueError("output must be an explicitly named offline_demo_* directory")
        if not output.parent.is_dir():
            raise ValueError("output parent must already exist")
        panel, folds = synthetic_bundle()
        report, scores = _run_pipeline(panel, folds)
        validate_full_grid(scores, panel, folds)
        report["synthetic_fixture"] = {
            "purpose": "PORTFOLIO_PAGE_ENGINEERING_ACCEPTANCE_ONLY",
            "universe_description": "300 fabricated names; not CSI 300 constituent data",
            "universe_count": UNIVERSE_SIZE, "seed": SEED,
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "missing_input_code": panel.codes[-1],
            "missing_input_signal_at": panel.signal_at[folds[-1].test_end - 1],
            "missing_input_features": list(panel.features),
            "budget_note": "Reduced fixed demo budget; no claim of optimal selection or strategy validation",
        }
        paths = write_demo_bundle(output, report, scores)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"synthetic 300-name demo stopped: {exc}\n")
    print(json.dumps({
        "status": report["status"], "data_role": panel.data_role,
        "universe_count": UNIVERSE_SIZE,
        "date_fold_count": len(scores.groupby(["date", "fold_id"])),
        "score_rows": len(scores), "valid_score_rows": int(scores["signal_valid"].sum()),
        "holdout_read": False, "network_used": False,
        "production_allowed": False, "sota_promoted": False,
        "output": str(output), "artifacts": paths,
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
