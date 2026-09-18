"""Run the score-only, non-promotable factor fitting demonstration.

There is no dataset discovery, whitelist, SOTA or Holdout reader. The default
proposal replay makes no network requests; RD-Agent proposals need explicit
mode, network and model choices. Development inputs must be an explicitly
supplied prepared bundle; its declarations still need independent auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PureWindowsPath

# The optional isolated backend directory affects this process only.
ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / ".demo_runtime", ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import numpy as np
import pandas as pd

from factor_research.group_evaluator import SelectionFold, _utc
from factor_research.joint_runner import JointPanel, JointResearchRunner, OuterFold
from factor_research.demo_artifacts import write_demo_bundle


_FEATURE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
# Trusted project metadata: stage3_implementation.md, "正式 Development OOF".
# A supplied bundle cannot move this boundary later to relabel Holdout as data
# for Development. Earlier caller boundaries are deliberately still accepted.
TRUSTED_DEVELOPMENT_END = "2024-08-26T00:00:00+08:00"
_MANIFEST_FIELDS = {
    "data_role", "codes", "feature_ids", "signal_at", "label_end_at",
    "holdout_start", "feature_contract_hash", "feature_manifest_hash",
    "arrays_filename", "outer_folds", "label_horizon_bars",
}


def _reject_sensitive_path(path: Path) -> None:
    if any("holdout" in part.lower() or "sealed" in part.lower() for part in path.parts):
        raise ValueError("Holdout/sealed input and output paths are forbidden")


def _safe_child(directory: Path, filename: str) -> Path:
    # Check Windows separators even when tests are executed on another OS.
    if (not isinstance(filename, str) or not filename
            or Path(filename).is_absolute() or PureWindowsPath(filename).is_absolute()
            or Path(filename).name != filename or PureWindowsPath(filename).name != filename
            or filename in {".", ".."} or ":" in filename):
        raise ValueError("bundle filenames must be plain relative filenames")
    supplied = directory / filename
    _reject_sensitive_path(supplied)
    resolved = supplied.resolve(strict=True)
    _reject_sensitive_path(resolved)
    if supplied.is_symlink() or resolved.parent != directory or not resolved.is_file():
        raise ValueError("bundle files must not escape their directory or use symlinks")
    return resolved


def _array_of_strings(value, name: str) -> tuple[str, ...]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(x, str) or not x.strip() for x in value)):
        raise ValueError(f"{name} must be a nonempty array of strings")
    return tuple(value)


def _parse_folds(value) -> tuple[OuterFold, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("at least two frozen outer folds are required")
    folds = []
    outer_keys = {"fold_id", "train_start", "train_end", "test_start", "test_end", "inner_folds"}
    inner_keys = {"train_start", "train_end", "val_start", "val_end"}
    for record in value:
        if not isinstance(record, dict) or set(record) != outer_keys:
            raise ValueError("invalid outer fold fields")
        if not isinstance(record["fold_id"], str) or not record["fold_id"].strip():
            raise ValueError("outer fold ID must be nonempty")
        if any(type(record[key]) is not int for key in outer_keys - {"fold_id", "inner_folds"}):
            raise ValueError("outer fold indices must be integers")
        inner = record["inner_folds"]
        if not isinstance(inner, list) or len(inner) < 3:
            raise ValueError("at least three frozen inner folds are required")
        nested = []
        for item in inner:
            if (not isinstance(item, dict) or set(item) != inner_keys
                    or any(type(item[key]) is not int for key in inner_keys)):
                raise ValueError("invalid inner fold fields or indices")
            nested.append(SelectionFold(**item))
        folds.append(OuterFold(**{key: record[key] for key in outer_keys - {"inner_folds"}},
                               inner_folds=tuple(nested)))
    return tuple(folds)


def load_development_bundle(source: str | Path) -> tuple[JointPanel, tuple[OuterFold, ...]]:
    """Validate metadata and boundaries BEFORE opening the numeric payload.

    Expected files: development_manifest.json and development_arrays.npz.
    Arrays: target, member_mask, feature__ID and available__ID for each ID.
    There is deliberately no scan of repository or external data directories.
    """
    supplied = Path(source)
    _reject_sensitive_path(supplied)
    directory = supplied.resolve(strict=True)
    _reject_sensitive_path(directory)
    if supplied.is_symlink() or not directory.is_dir():
        raise ValueError("development bundle must be an explicit nonsymlink directory")
    manifest_path = _safe_child(directory, "development_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) - _MANIFEST_FIELDS:
        raise ValueError("unknown or invalid Development manifest fields")
    required = _MANIFEST_FIELDS - {"feature_manifest_hash"}
    if not required <= set(manifest) or manifest["data_role"] != "development":
        raise ValueError("explicit Development role and complete metadata are required")
    if type(manifest["label_horizon_bars"]) is not int or manifest["label_horizon_bars"] != 5:
        raise ValueError("label_horizon_bars must be the integer 5 for this H5-only demo")
    codes = _array_of_strings(manifest["codes"], "codes")
    names = _array_of_strings(manifest["feature_ids"], "feature_ids")
    if len(set(codes)) != len(codes) or len(set(names)) != len(names):
        raise ValueError("duplicate codes or feature IDs")
    if any(not _FEATURE_ID.fullmatch(name) for name in names):
        raise ValueError("feature IDs must be safe identifiers")
    signals = _array_of_strings(manifest["signal_at"], "signal_at")
    label_ends = _array_of_strings(manifest["label_end_at"], "label_end_at")
    # No NPZ open or array read is allowed above or inside these checks.
    times, ends = _utc(signals), _utc(label_ends)
    if not isinstance(manifest["holdout_start"], str):
        raise ValueError("holdout_start must be an explicit timestamp")
    holdout = _utc([manifest["holdout_start"]])[0]
    trusted_end = _utc([TRUSTED_DEVELOPMENT_END])[0]
    if holdout > trusted_end:
        raise ValueError("declared Holdout boundary exceeds the trusted Development limit")
    if (len(times) != len(ends) or not np.all(np.diff(times) > 0)
            or np.any(ends < times)):
        raise ValueError("invalid ordered signal / label-end timestamps")
    if (np.any(times >= holdout) or np.any(ends >= holdout)
            or np.any(times >= trusted_end) or np.any(ends >= trusted_end)):
        raise ValueError("Development input or label reaches Holdout boundary")
    contract = manifest["feature_contract_hash"]
    if not isinstance(contract, str) or not _HASH.fullmatch(contract):
        raise ValueError("feature_contract_hash must be lowercase SHA256")
    feature_hash = hashlib.sha256(json.dumps(names, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False).encode()).hexdigest()
    if manifest.get("feature_manifest_hash", feature_hash) != feature_hash:
        raise ValueError("feature manifest fingerprint mismatch")
    folds = _parse_folds(manifest["outer_folds"])
    JointResearchRunner()._validate_folds(folds, len(times), times, ends)
    filename = manifest["arrays_filename"]
    if filename != "development_arrays.npz":
        raise ValueError("arrays_filename must be development_arrays.npz")
    arrays_path = _safe_child(directory, filename)
    expected = {"target", "member_mask"}
    expected.update("feature__" + name for name in names)
    expected.update("available__" + name for name in names)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected or len(arrays.files) != len(expected):
            raise ValueError("NPZ must contain exactly the declared Development arrays")
        target = arrays["target"].copy()
        members = arrays["member_mask"].copy()
        features = {name: arrays["feature__" + name].copy() for name in names}
        available = {name: arrays["available__" + name].copy() for name in names}
    shape = (len(codes), len(times))
    if (target.shape != shape or target.dtype.kind not in "fiu"
            or members.shape != shape or members.dtype != np.dtype(bool)):
        raise ValueError("target / original boolean member grid is invalid")
    if any(features[name].shape != shape or features[name].dtype.kind not in "fiu"
           or available[name].shape != shape or available[name].dtype != np.dtype("int64")
           for name in names):
        raise ValueError("feature / UTC nanosecond availability arrays are invalid")
    panel = JointPanel(features, target, members, codes, signals, label_ends, available,
                       manifest["holdout_start"], feature_manifest_hash=feature_hash,
                       feature_contract_hash=contract, data_role="development")
    JointResearchRunner._prepare(panel)
    return panel, folds


def synthetic_bundle() -> tuple[JointPanel, tuple[OuterFold, ...]]:
    """Deterministic FAKE data for engineering, never investment evidence."""
    rng = np.random.default_rng(20260915)
    n, t = 16, 320
    features = {name: rng.normal(size=(n, t)) for name in ("A", "B", "C")}
    target = (.02 * features["A"] + .03 * features["A"] * features["B"]
              + .002 * features["C"] + .005 * rng.normal(size=(n, t)))
    times = pd.date_range("2015-01-01 13:00", periods=t, freq="B", tz="UTC")
    ns = np.asarray([stamp.value for stamp in times], dtype=np.int64)
    available = {name: np.broadcast_to(ns, (n, t)).copy() for name in features}
    panel = JointPanel(features, target, np.ones((n, t), dtype=bool),
                       tuple(f"SYNTHETIC_{i:03d}" for i in range(n)),
                       tuple(stamp.isoformat() for stamp in times),
                       tuple((stamp + pd.offsets.BDay(5)).isoformat() for stamp in times),
                       available, "2017-01-01T00:00:00Z", data_role="synthetic")
    inner = (SelectionFold(0, 40, 60, 80), SelectionFold(0, 70, 90, 110),
             SelectionFold(0, 100, 120, 140))
    folds = (OuterFold("demo_outer_1", 0, 160, 180, 200, inner),
             OuterFold("demo_outer_2", 0, 220, 240, 260, inner))
    return panel, folds


def _run_pipeline(panel: JointPanel, folds: tuple[OuterFold, ...], *, config):
    from factor_research.demo_pipeline import OfflineFactorDemo
    return OfflineFactorDemo(config=config).run(panel, folds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New score-only output directory")
    parser.add_argument("--proposal-mode", choices=("offline_replay", "rd_agent"),
                        default="offline_replay", help="Replay by default; RD-Agent requires explicit network opt-in")
    parser.add_argument("--allow-network", action="store_true",
                        help="Explicitly allow the RD-Agent formula proposal service; forbidden for replay")
    parser.add_argument("--rd-model", default="", help="Explicit RD-Agent model name; no default remote model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Use clearly labelled fake data")
    source.add_argument("--development-bundle", type=Path, help="Explicit prepared Development directory")
    args = parser.parse_args(argv)
    try:
        _reject_sensitive_path(args.output)
        output = args.output.resolve()
        _reject_sensitive_path(output)
        if output.exists():
            raise ValueError("output directory must be new; existing artifacts are not overwritten")
        # Validate configuration BEFORE loading either numeric data or proposals.
        # Merely constructing DemoConfig performs no network request.
        from factor_research.demo_pipeline import DemoConfig
        if args.proposal_mode == "offline_replay" and args.rd_model.strip():
            raise ValueError("--rd-model requires --proposal-mode rd_agent")
        config = DemoConfig(proposal_mode=args.proposal_mode, allow_network=args.allow_network,
                            rd_model=args.rd_model)
        panel, folds = synthetic_bundle() if args.synthetic else load_development_bundle(args.development_bundle)
        report, scores = _run_pipeline(panel, folds, config=config)
        paths = write_demo_bundle(output, report, scores)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"offline demo stopped: {exc}\n")
    print(json.dumps({"status": report["status"], "data_role": panel.data_role,
                      "holdout_read": False, "production_allowed": False,
                      "sota_promoted": False, "output": str(output), "artifacts": paths},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
