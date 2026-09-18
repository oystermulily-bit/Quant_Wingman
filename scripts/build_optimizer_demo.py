"""Build an evidence-rich, entirely synthetic 300-stock optimizer example.

An actual frozen least-squares formula is fitted before all calibration
forecasts. No real stock data, existing research runs or Holdout are read.
Use a fresh output name; existing artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / ".demo_runtime", ROOT):
    sys.path.insert(0, str(location))

import numpy as np
import pandas as pd

from factor_research.demo_artifacts import write_demo_bundle, _reject_sensitive_path

SEED = 20260917


def synthetic_inputs(*, include_settlements=False):
    rng = np.random.default_rng(SEED)
    count, size, horizon = 1010, 300, 5
    days = pd.bdate_range("2010-01-04", periods=count)
    signals = days.tz_localize("UTC") + pd.Timedelta(hours=13)
    opens = days.tz_localize("UTC") + pd.Timedelta(hours=1, minutes=30)
    codes = [f"SYNTHETIC_{i:03d}" for i in range(size)]
    x = rng.normal(size=(count, size))
    for i in range(1, count):
        x[i] = .8*x[i-1]+.6*x[i]
    sector = np.arange(size) % 10
    volatility = np.linspace(.009, .032, size)
    daily = (rng.normal(0, .004, (count-1, 1))
             + rng.normal(0, .005, (count-1, 10))[:, sector]
             + rng.normal(size=(count-1, size))*volatility)
    # The next opening-to-opening return depends only on earlier closing features.
    daily[1:] += .0012*x[:-2]
    daily = np.clip(daily, -.15, .15)
    prices = np.vstack([np.full(size, 30.), 30.*np.cumprod(1+daily, axis=0)])
    y = prices[horizon+1:]/prices[1:-horizon]-1
    train = np.arange(0, 190)
    design = np.column_stack([np.ones(len(train)*size), x[train].ravel()])
    coefficients = np.linalg.lstsq(design, y[train].ravel(), rcond=None)[0]
    predictions = coefficients[0]+coefficients[1]*x
    fingerprints = {
        "model": sha256(coefficients.tobytes()).hexdigest(),
        "formula": sha256(b"intercept + beta * SYNTHETIC_AR_FEATURE; OLS frozen train 0:190").hexdigest(),
        "fold": sha256(b"train0:190; calibration340:844; freeze850; display900; horizon5").hexdigest(),
        "data": sha256(x.tobytes()+prices.tobytes()+str(SEED).encode()).hexdigest(),
    }
    current, freeze = 900, 850
    indices = np.arange(340, 844)
    calendar = opens[current-504:current+1]
    starts = np.arange(current-500, current, horizon)
    payload = {
        "schema_version": "portfolio_research_inputs_v1", "data_role": "synthetic",
        "production_allowed": False, "holdout_read": False, "fingerprints": fingerprints,
        "date": str(days[current].date()), "fold_id": "synthetic_frozen_1", "horizon": horizon,
        "calibration_frozen_at": signals[freeze].isoformat(),
        "return_semantics": "GROSS_TOTAL_RETURN_MINUS_CASH_SAME_EXECUTION_HORIZON",
        "codes": codes, "current_scores": dict(zip(codes, predictions[current].tolist())),
        "calibration": {
            "codes": codes, "signal_at": [signals[i].isoformat() for i in indices],
            "label_end_at": [opens[i+horizon+1].isoformat() for i in indices],
            "forecast_fit_end_at": [opens[195].isoformat()]*len(indices), "oof": [True]*len(indices),
            "scores": predictions[indices].tolist(), "forward_excess_returns": y[indices].tolist(),
            "members": np.ones((len(indices), size), dtype=bool).tolist(),
        },
        "risk": {"trading_at": [t.isoformat() for t in calendar],
                 "start_at": [opens[i].isoformat() for i in starts],
                 "end_at": [opens[i+horizon].isoformat() for i in starts],
                 "returns": (prices[starts+horizon]/prices[starts]-1).tolist()},
        "market": {c: {"available_at": signals[current].isoformat(), "industry": f"合成行业_{sector[i]+1}",
                       "price": float(prices[current, i]), "lot_size": 100,
                       "mean_amount_20d": 100_000_000., "median_amount_20d": 90_000_000.}
                   for i, c in enumerate(codes)},
    }
    report = {"run_id": "synthetic_optimizer_300_20260917", "data_role": "synthetic",
              "status": "OFFLINE_DEMO_COMPLETED_NOT_VALIDATED", "research_go": False,
              "holdout_read": False, "sota_promoted": False, "production_allowed": False,
              "whitelist_applied": False, "fingerprints": fingerprints, "expected_score_rows": size,
              "signal_output": "model_prediction", "selected_features": ["SYNTHETIC_AR_FEATURE"],
              "formulas": [{"expression": "intercept + beta * SYNTHETIC_AR_FEATURE"}],
              "model_evidence": {"estimator": "numpy.linalg.lstsq", "coefficients": coefficients.tolist(),
                                 "training_rows": len(train)*size, "fit_label_end_at": opens[195].isoformat(),
                                 "horizon": horizon, "seed": SEED, "synthetic_only": True},
              "warnings": ["SYNTHETIC_DATA_NOT_MARKET_EVIDENCE"]}
    scores = pd.DataFrame({"date": [days[current]]*size, "code": codes,
                           "score": predictions[current], "score_available_at": [signals[current]]*size,
                           "fold_id": [payload["fold_id"]]*size, "is_member": True, "signal_valid": True})
    if include_settlements:
        settlements = [{"start_at": opens[i].isoformat(), "end_at": opens[i+1].isoformat(),
                        "returns": dict(zip(codes, (prices[i+1]/prices[i]-1).tolist())), "cash_return": 0.}
                       for i in range(current+1, current+horizon+1)]
        return report, scores, payload, settlements
    return report, scores, payload


def build_demo(score_dir: Path, inputs_dir: Path):
    for path in (score_dir, inputs_dir):
        _reject_sensitive_path(path)
        if path.exists() or not path.name.startswith("offline_demo_"):
            raise ValueError("Use fresh, explicitly named offline_demo_ directories")
    report, scores, payload = synthetic_inputs()
    paths = write_demo_bundle(score_dir, report, scores)
    inputs_dir.mkdir()
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    (inputs_dir / "research_inputs.json").write_bytes(raw)
    manifest = {"artifact_kind": "OFFLINE_PORTFOLIO_RESEARCH_INPUTS", "schema_version": "1.0",
                "production_allowed": False, "holdout_read": False, "data_role": "synthetic",
                "score_manifest": str(Path(paths["manifest"]).resolve()),
                "date": payload["date"], "fold_id": payload["fold_id"], "sha256": sha256(raw).hexdigest()}
    (inputs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "experiments/offline_demo_optimizer_300_20260917")
    parser.add_argument("--inputs", type=Path, default=ROOT / "experiments/offline_demo_optimizer_inputs_20260917")
    args = parser.parse_args()
    print(json.dumps(build_demo(args.scores, args.inputs), ensure_ascii=False))
