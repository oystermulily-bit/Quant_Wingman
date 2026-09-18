"""Nested, date-forward joint feature research. No SOTA/portfolio/Holdout I/O.

Selection sees only the outer-training prefix. Outer fitting receives NaN in
every non-training label cell. Predictions are not a profitability verdict.
Dynamic formula generation is deliberately not wired into this first runner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .group_evaluator import FixedLightGBMGroupEvaluator, SelectionFold, _utc, clean_training_candidates
from .group_search import GroupSearchConfig, StepwiseGroupSearch
from .lgbm_baseline import FixedLightGBMBaseline


def identity(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class OuterFold:
    fold_id: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    # Absolute date-column indices, not stock-row splits.
    inner_folds: tuple[SelectionFold, ...]


@dataclass(frozen=True)
class JointPanel:
    features: dict[str, np.ndarray]
    target: np.ndarray
    member_mask: np.ndarray
    codes: tuple[str, ...]
    signal_at: tuple[str, ...]
    label_end_at: tuple[str, ...]
    # Per-stock/per-date UTC Unix nanoseconds, int64; NaT=int64.min.
    # This is the MAX availability of all inputs needed for this feature.
    available_at_ns: dict[str, np.ndarray]
    holdout_start: str
    feature_manifest_hash: str = ""
    feature_contract_hash: str = ""
    data_role: str = "development"


class JointResearchRunner:
    """Engineering runner for fixed, audited feature manifests.

    Live admission is explicit; a synthetic run never claims research success.
    Economic execution and concentration approval are separate required stages.
    """

    def __init__(self, *, search_config=None, model_config=None):
        self.search_config = search_config or GroupSearchConfig()
        self.model_config = model_config or ResearchConfig()
        if self.model_config.purge_gap < 0 or self.model_config.embargo_bars < 0:
            raise ValueError("negative purge/embargo")
        if (self.model_config.lgbm_feature_fraction != 1
                or self.model_config.lgbm_bagging_fraction != 1):
            raise ValueError("joint evaluator requires fixed full feature/sample fractions")

    def config_identity(self, *, signal_output, pair_seeds):
        model = FixedLightGBMBaseline(self.model_config)
        try:
            backend_version = version("lightgbm")
        except PackageNotFoundError:
            backend_version = "UNAVAILABLE"
        return identity({"search": asdict(self.search_config), "model": model._params(),
                         "library_versions": {"lightgbm": backend_version,
                                              "numpy": np.__version__, "pandas": pd.__version__},
                         "rounds": self.model_config.lgbm_num_boost_round,
                         "purge": self.model_config.purge_gap,
                         "embargo": self.model_config.embargo_bars,
                         "signal_output": signal_output, "pair_seeds": pair_seeds})

    @staticmethod
    def split_identity(folds):
        return identity([asdict(f) for f in folds])

    @staticmethod
    def implementation_identity():
        root = Path(__file__).parent
        files = ("joint_runner.py", "joint_protocol.py", "group_search.py", "group_evaluator.py",
                 "lgbm_baseline.py", "validation_unit.py", "config.py")
        return identity({name: sha256((root/name).read_bytes()).hexdigest() for name in files})

    @staticmethod
    def _prepare(panel):
        if panel.data_role not in ("development", "synthetic"):
            raise ValueError("only Development or synthetic panels are permitted")
        y = np.array(panel.target, dtype=float, copy=True)
        members = np.asarray(panel.member_mask)
        if y.ndim != 2 or members.shape != y.shape or members.dtype != np.dtype(bool):
            raise ValueError("original members must be a boolean assets x dates panel")
        members = members.copy()
        n, t = y.shape
        if (len(panel.codes) != n or len(set(panel.codes)) != n
                or any(not isinstance(c, str) or not c for c in panel.codes)):
            raise ValueError("unique stock codes required")
        times, ends = _utc(panel.signal_at), _utc(panel.label_end_at)
        holdout = _utc([panel.holdout_start])[0]
        if (len(times) != t or len(ends) != t or not np.all(np.diff(times) > 0)
                or np.any(ends < times)):
            raise ValueError("invalid ordered signal/label times")
        if np.any(times >= holdout) or np.any(ends >= holdout):
            raise ValueError("panel inputs/labels reach Holdout boundary")
        names = tuple(sorted(panel.features))
        if (not names or any(not isinstance(x, str) or not x for x in names)
                or set(names) != set(panel.available_at_ns)):
            raise ValueError("every feature requires explicit availability metadata")
        values, late_counts = {}, {}
        fingerprint = sha256(identity({"codes": panel.codes, "names": names,
                                      "holdout": panel.holdout_start}).encode())
        for value in (y, members, times, ends):
            fingerprint.update(value.tobytes())
        for name in names:
            x = np.array(panel.features[name], dtype=float, copy=True)
            available = np.asarray(panel.available_at_ns[name])
            if x.shape != y.shape or available.shape != y.shape or available.dtype != np.dtype("int64"):
                raise ValueError("feature/availability must match assets x dates; availability is UTC int64 ns")
            fingerprint.update(x.tobytes())
            fingerprint.update(available.tobytes())
            unavailable = (available == np.iinfo(np.int64).min) | (available > times[None, :])
            late_counts[name] = int((unavailable & members).sum())
            x[unavailable | ~members | ~np.isfinite(x)] = np.nan
            values[name] = x
        return y, members, times, ends, values, late_counts, fingerprint.hexdigest()

    def _validate_folds(self, folds, t, times, ends):
        if len(folds) < 2 or len({f.fold_id for f in folds}) != len(folds):
            raise ValueError("at least two distinct outer folds required")
        previous_end = -1
        for f in folds:
            a, b, c, d = f.train_start, f.train_end, f.test_start, f.test_end
            if (not f.fold_id or any(type(x) is not int for x in (a, b, c, d))
                    or not 0 <= a < b <= c < d <= t or c < previous_end):
                raise ValueError("invalid or overlapping outer folds")
            cutoff = min(b, c - self.model_config.purge_gap)
            if cutoff <= a or np.any(ends[a:cutoff] >= times[c]):
                raise ValueError("outer training labels overlap test signal time")
            if len(f.inner_folds) < self.search_config.min_folds:
                raise ValueError("insufficient nested inner folds")
            for inner in f.inner_folds:
                if not a <= inner.train_start < inner.train_end <= inner.val_start < inner.val_end <= cutoff:
                    raise ValueError("inner fold reaches outside outer-training scope")
            previous_end = d

    def run_synthetic(self, panel, folds, *, signal_output, pair_seeds=()):
        """Synthetic engineering checks only, never a bypass for real data."""
        if panel.data_role != "synthetic":
            raise ValueError("synthetic entry requires explicitly synthetic data")
        return self._run(panel, folds, signal_output=signal_output, pair_seeds=pair_seeds)

    def run_development(self, panel, folds, *, plan, run_id, pair_seeds=()):
        """Explicitly frozen run only; legacy Stage4 failure is not an input.

        The loader must supply only authorised Development data. Neither role
        strings nor audit hashes constitute OS-level data isolation.
        """
        from .joint_protocol import RunLedger

        plan.validate(mode="development")
        if panel.data_role != "development":
            raise ValueError("live research requires Development data")
        prepared = self._prepare(panel)
        expected = {
            "data_fingerprint": prepared[-1],
            "split_fingerprint": self.split_identity(folds),
            "config_fingerprint": self.config_identity(signal_output=plan.signal_output, pair_seeds=pair_seeds),
            "implementation_hash": self.implementation_identity(),
            "feature_manifest_hash": panel.feature_manifest_hash,
            "feature_contract_hash": panel.feature_contract_hash,
        }
        for name, observed in expected.items():
            if getattr(plan, name) != observed:
                raise ValueError("frozen " + name + " mismatch")
        if tuple(sorted(plan.feature_ids)) != tuple(sorted(panel.features)):
            raise ValueError("frozen feature IDs mismatch")
        if not np.asarray(panel.member_mask).sum(axis=0).tolist() == [300]*len(panel.signal_at):
            raise ValueError("PIT_HS300 original daily 300-member denominator changed")
        if len(folds) != plan.max_outer_scopes or self.search_config.max_trials != plan.max_trials_per_scope:
            raise ValueError("frozen outer scopes/search budget mismatch")
        if plan.max_total_trials < len(folds)*self.search_config.max_trials:
            raise ValueError("total budget cannot reserve all outer scopes before starting")
        self._validate_folds(folds, panel.target.shape[1], prepared[2], prepared[3])
        # Missing backend is a dependency error, not negative economic evidence.
        FixedLightGBMBaseline._module()
        ledger = RunLedger(plan.ledger_directory)
        ledger.begin_run(plan, run_id, mode="development")

        class ScopedLedger:
            def reserve_scope(self, scope, limit):
                ledger.reserve_scope(run_id, scope, limit)

            def complete_scope(self, scope, count):
                ledger.complete_scope(run_id, scope, count)

        try:
            result = self._run(panel, folds, signal_output=plan.signal_output,
                               pair_seeds=pair_seeds, ledger=ScopedLedger())
        except Exception:
            ledger.finish_run(run_id, success=False)
            raise
        ledger.finish_run(run_id, success=True)
        result["budget_ledger"] = ledger.snapshot(run_id)
        result["plan_fingerprint"] = plan.fingerprint
        return result

    def _run(self, panel, folds, *, signal_output, pair_seeds=(), ledger=None):
        if signal_output not in ("model_prediction", "equal_weight"):
            raise ValueError("freeze model_prediction or equal_weight explicitly")
        y, members, times, ends, features, late, data_hash = self._prepare(panel)
        if len(features) > self.search_config.max_candidates:
            raise ValueError("candidate manifest exceeds frozen cap")
        if len(pair_seeds) > self.search_config.max_pair_seeds:
            raise ValueError("too many frozen pairs")
        if any(len(p) != 2 or len(set(p)) != 2 or not set(p) <= set(features) for p in pair_seeds):
            raise ValueError("pair outside frozen manifest")
        self._validate_folds(folds, y.shape[1], times, ends)
        records, previous_test_ends = [], []
        predictions = np.full(y.shape, np.nan)
        for fold in folds:
            scope = str(fold.fold_id)
            if ledger is not None:
                ledger.reserve_scope(scope, self.search_config.max_trials)
            try:
                record, score = self._fold(panel, fold, y, members, features, data_hash,
                                           previous_test_ends, signal_output, pair_seeds)
                predictions[:, fold.test_start:fold.test_end] = score
                records.append(record)
                if ledger is not None:
                    ledger.complete_scope(scope, record["selection"]["trial_count"])
            except Exception:
                # A started scope retains its full reservation on failure.
                raise
            previous_test_ends.append(fold.test_end)
        counts = {}
        for row in records:
            for name in row["selection"]["selected"]:
                counts[name] = counts.get(name, 0) + 1
        usable = sum(r["status"] != "NO_CANDIDATE" for r in records)
        prediction_status = ("NO_JOINT_CANDIDATE" if not usable else
                             "JOINT_OUTER_PARTIAL_NO_CANDIDATE" if usable < len(records) else
                             "JOINT_OUTER_PREDICTIONS_READY")
        return {"protocol": "JOINT_RESEARCH_S2", "data_fingerprint": data_hash,
                "split_fingerprint": self.split_identity(folds),
                "config_fingerprint": self.config_identity(signal_output=signal_output, pair_seeds=pair_seeds),
                "status": ("SYNTHETIC_ENGINEERING_ONLY" if panel.data_role == "synthetic"
                           else prediction_status),
                "prediction_status": prediction_status, "usable_outer_folds": usable,
                "economic_status": "ECONOMIC_VALIDATION_PENDING",
                "signal_output": signal_output, "folds": records,
                "selection_frequency": counts,
                "total_selection_trials": sum(r["selection"]["trial_count"] for r in records),
                "unavailable_member_days": late, "predictions": predictions,
                "development_adaptive": True, "holdout_read": False,
                "stage5_allowed": False, "sota_promoted": False,
                "production_allowed": False, "research_go": False}

    def _fold(self, panel, fold, y, members, features, data_hash, previous_ends,
              signal_output, pair_seeds):
        a, c, d = fold.train_start, fold.test_start, fold.test_end
        b = min(fold.train_end, c - self.model_config.purge_gap)
        allowed = np.ones(b - a, dtype=bool)
        global_dates = np.arange(a, b)
        for end in previous_ends:
            allowed &= (global_dates < end) | (global_dates >= end + self.model_config.embargo_bars)
        inner_members = members[:, a:b].copy() & allowed[None, :]
        # Do not mutate the original denominator to hide embargoed dates:
        # embargo may remove training rows, not inner validation membership.
        for inner in fold.inner_folds:
            if not allowed[inner.val_start-a:inner.val_end-a].all():
                raise ValueError("inner validation overlaps prior outer embargo")
        earliest = fold.inner_folds[0]
        clean_mask = np.zeros(inner_members.shape, dtype=bool)
        clean_end = min(earliest.train_end, earliest.val_start-self.model_config.purge_gap)
        clean_mask[:, earliest.train_start-a:clean_end-a] = inner_members[:, earliest.train_start-a:clean_end-a]
        # No outer columns or labels are supplied to candidate cleaning/search.
        prefix = {k: v[:, a:b].copy() for k, v in features.items()}
        clean, exclusions = clean_training_candidates(prefix, clean_mask)
        if not clean:
            raise ValueError("no usable training candidates")
        retained_pairs = tuple(p for p in pair_seeds if set(p) <= set(clean))
        local_folds = tuple(SelectionFold(i.train_start-a, i.train_end-a, i.val_start-a, i.val_end-a)
                            for i in fold.inner_folds)
        evaluator = FixedLightGBMGroupEvaluator(
            clean, y[:, a:b].copy(), inner_members, local_folds,
            signal_at=panel.signal_at[a:b], label_end_at=panel.label_end_at[a:b],
            selection_end_at=panel.signal_at[c], holdout_start=panel.holdout_start,
            data_fingerprint=data_hash, data_role="development_inner", config=self.model_config)
        # Context names the new route; inner-only scores are not outer evidence.
        result = StepwiseGroupSearch(self.search_config).run(
            list(clean), evaluator, pair_seeds=retained_pairs, data_role="development_inner")
        if result.status != "INNER_SELECTION_ONLY":
            raise ValueError("joint inner selection did not complete: " + result.status)
        selection = result.to_dict()
        selection["trial_count"] = len(result.trials)
        selection["excluded_features"] = exclusions
        selection["excluded_pairs"] = [p for p in pair_seeds if p not in retained_pairs]
        selected = result.selected
        if not selected:
            return ({"fold_id": fold.fold_id, "selection": selection,
                     "status": "NO_CANDIDATE", "outer_rank_ic": None,
                     "signal_coverage": 0.0, "label_coverage": 0.0},
                    np.full((y.shape[0], d-c), np.nan))
        # Shared support is fixed across all surviving candidates, not selected
        # anew to improve the winning group's measured coverage.
        all_x = np.stack([features[k][:, a:d] for k in sorted(clean)], axis=2)
        finite_x = np.isfinite(all_x).all(axis=2) & members[:, a:d]
        n, length = y.shape[0], d-a
        train_mask = np.zeros((n, length), dtype=bool)
        train_mask[:, :b-a] = finite_x[:, :b-a] & inner_members & np.isfinite(y[:, a:b])
        test_mask = np.zeros((n, length), dtype=bool)
        test_mask[:, c-a:d-a] = finite_x[:, c-a:d-a]
        denominator = int(members[:, c:d].sum())
        coverage = float(test_mask.sum()/denominator) if denominator else 0.0
        if coverage < self.search_config.min_coverage or (test_mask[:, c-a:].sum(axis=0) < 10).any():
            raise ValueError("outer signal coverage insufficient; original members retained")
        train, valid = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
        if len(train) < max(20, 3*len(clean)):
            raise ValueError("insufficient outer-training support")
        chosen = np.stack([features[k][:, a:d] for k in selected], axis=2)
        target_for_fit = np.full((n, length), np.nan)
        target_for_fit[train_mask] = y[:, a:d][train_mask]
        model = FixedLightGBMBaseline(self.model_config)
        if signal_output == "model_prediction":
            predicted = model._fit_predict(chosen, target_for_fit, train, valid)
        else:
            # Fixed cross-sectional average ranks, not fitted weights. Average
            # uses all selected columns; it never averages a partial signal.
            ranked = np.stack([pd.DataFrame(chosen[:, :, k]).rank(axis=0, pct=True).to_numpy()
                               for k in range(len(selected))], axis=2)
            predicted = ranked.mean(axis=2).ravel()[valid]
        if predicted.shape != valid.shape or not np.isfinite(predicted).all():
            raise ValueError("invalid outer predictions")
        out = np.full((n, length), np.nan)
        out.ravel()[valid] = predicted
        # Only after prediction exists do we consult the outer evaluation labels.
        labelled = test_mask & np.isfinite(y[:, a:d])
        eval_idx = np.flatnonzero(labelled)
        label_coverage = float(labelled.sum()/denominator)
        if label_coverage < self.search_config.min_coverage or (labelled[:, c-a:].sum(axis=0) < 10).any():
            raise ValueError("outer label coverage insufficient")
        if any(np.unique(y[:, a:d][labelled[:, j], j]).size < 2 for j in range(c-a, d-a)):
            raise ValueError("degenerate outer labels")
        metric = model._oof_score(out.ravel()[eval_idx], y[:, a:d].ravel()[eval_idx], eval_idx, n, length)
        return ({"fold_id": fold.fold_id, "selection": selection,
                 "status": "OUTER_PREDICTED_NOT_ECONOMICALLY_VALIDATED",
                 "outer_rank_ic": metric, "signal_coverage": coverage,
                 "label_coverage": label_coverage, "member_days": denominator,
                 "test_start": panel.signal_at[c], "test_end": panel.signal_at[d-1],
                 "fit_label_digest": sha256(target_for_fit.tobytes()).hexdigest()}, out[:, c-a:])
