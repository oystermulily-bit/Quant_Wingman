"""Offline-only joint feature / RD-style formula / LightGBM demonstration.

Economic criteria are warnings here, never fabricated passes. Time boundaries,
missingness, valid finite samples and budget exhaustion remain real constraints.
The existing formal S2 runner, SOTA and recommendation APIs are not modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .group_evaluator import FixedLightGBMGroupEvaluator, SelectionFold, _utc, clean_training_candidates
from .group_search import GroupSearchConfig, StepwiseGroupSearch
from .joint_runner import JointResearchRunner, identity
from .lgbm_baseline import FixedLightGBMBaseline
from .demo_formulas import FormulaLimits, OfflineReplaySource, evaluate_formula, validate_formula


# Trusted project metadata, not a value supplied by a Development bundle.
# See docs/implementation_gates_20260825/stage3_implementation.md.
PROJECT_HOLDOUT_START = "2024-08-26T00:00:00+08:00"


@dataclass(frozen=True)
class DemoConfig:
    purpose: str = "OFFLINE_DEMO_ONLY"
    proposal_mode: str = "offline_replay"
    allow_network: bool = False
    rd_model: str = ""
    label_horizon_bars: int = 5
    formula_rounds: int = 2
    proposals_per_round: int = 4
    max_group_size: int = 5
    max_base_candidates: int = 256
    search_trials_per_round: int = 96
    max_total_opportunities: int = 4000
    max_outer_folds: int = 5
    coverage_warning_threshold: float = .85
    seed: int = 20260915
    num_boost_round: int = 30

    def __post_init__(self):
        if self.purpose != "OFFLINE_DEMO_ONLY":
            raise ValueError("demo cannot be used as a production/research release")
        if type(self.label_horizon_bars) is not int or self.label_horizon_bars != 5:
            raise ValueError("this demo requires an explicitly declared H5 target")
        if self.proposal_mode not in ("offline_replay", "rd_agent") or type(self.allow_network) is not bool:
            raise ValueError("invalid proposal mode/network choice")
        if self.proposal_mode == "rd_agent" and (not self.allow_network or not self.rd_model.strip()):
            raise ValueError("RD-Agent requires explicit network opt-in and model name")
        if self.proposal_mode == "offline_replay" and self.allow_network:
            raise ValueError("offline replay cannot enable network")
        for name in ("formula_rounds", "proposals_per_round", "max_group_size", "max_base_candidates",
                     "search_trials_per_round", "max_total_opportunities", "max_outer_folds", "num_boost_round"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.formula_rounds > 5 or self.proposals_per_round > 16 or self.max_outer_folds > 5:
            raise ValueError("demo generation/fold bounds exceeded")
        if not 0 < self.coverage_warning_threshold <= 1:
            raise ValueError("invalid coverage warning threshold")


class OfflineFactorDemo:
    def __init__(self, config=None):
        self.config = config or DemoConfig()
        self.model_config = ResearchConfig(random_seed=self.config.seed,
                                           lgbm_num_boost_round=self.config.num_boost_round,
                                           lgbm_num_leaves=7, lgbm_min_child_samples=20)
        self.guard = JointResearchRunner(model_config=self.model_config)

    def _source(self, names):
        if self.config.proposal_mode == "offline_replay":
            return OfflineReplaySource(tuple(names), limits=FormulaLimits())
        from .demo_formulas import RestrictedRDFormulaSource
        # Constructor is only reached on explicit opt-in; no implicit API key
        # lookup, remote request or fake fallback in default offline runs.
        return RestrictedRDFormulaSource(tuple(names), project_root=Path(__file__).resolve().parents[1], model=self.config.rd_model,
                                          allow_network=True, limits=FormulaLimits())

    def run(self, panel, folds):
        cfg = self.config
        if panel.data_role == "development" and _utc([panel.holdout_start])[0] > _utc([PROJECT_HOLDOUT_START])[0]:
            raise ValueError("Development cannot postpone the project's fixed Holdout boundary")
        y, members, times, ends, prepared, late, data_hash = self.guard._prepare(panel)
        self.guard._validate_folds(folds, y.shape[1], times, ends)
        if len(folds) > cfg.max_outer_folds or len(prepared) > cfg.max_base_candidates:
            raise ValueError("demo manifest/fold cap exceeded")
        # A conservative bound includes requested proposals (also failures),
        # every search baseline/comparison and every final outer model fit.
        upper_bound = len(folds)*(cfg.formula_rounds*(cfg.proposals_per_round+cfg.search_trials_per_round+1)+1)
        if upper_bound > cfg.max_total_opportunities:
            raise ValueError("insufficient declared budget for all demo folds")
        FixedLightGBMBaseline._module()  # Missing backend never means success.
        report = {
            "run_id": "offline-demo-"+uuid4().hex,
            "data_role": panel.data_role, "status": "OFFLINE_DEMO_COMPLETED_NOT_VALIDATED",
            "research_go": False, "holdout_read": False, "sota_promoted": False,
            "production_allowed": False, "whitelist_applied": False,
            "economic_validation": "NOT_RUN", "concentration_validation": "NOT_RUN",
            "formal_sota_written": False, "signal_output": "model_prediction", "horizon": cfg.label_horizon_bars,
            "label_contract": "declared_H5; source calculation requires upstream audit",
            "proposal_source": cfg.proposal_mode, "network_used": False,
            "config": asdict(cfg), "warnings": ["OFFLINE_DEMO_NOT_A_VALIDATED_STRATEGY"],
            "unavailable_member_days": late, "selected_features": [], "selection_runs": [], "formulas": [],
            "expected_score_rows": y.shape[0] * sum(f.test_end-f.test_start for f in folds),
            "timestamp_semantics": "score_available_at is a simulated historical information cutoff, not actual generation time",
            "feedback_rank_semantics": "stable selected-list ordinal, not factor importance",
            "budget": {"limit": cfg.max_total_opportunities, "reserved_upper_bound": upper_bound,
                       "charged": 0, "events": []},
            "model_evidence": {"backend": "lightgbm", "version": version("lightgbm"),
                               "params": FixedLightGBMBaseline(self.model_config)._params(),
                               "boost_rounds": cfg.num_boost_round, "folds": []},
        }
        if panel.data_role == "synthetic":
            report["warnings"].append("SYNTHETIC_DATA_NOT_MARKET_EVIDENCE")

        def charge(kind, fold_id, amount=1):
            if report["budget"]["charged"]+amount > cfg.max_total_opportunities:
                raise ValueError("demo budget exhausted")
            report["budget"]["charged"] += amount
            report["budget"]["events"].append({"kind": kind, "fold_id": fold_id, "count": amount})

        score_frames, previous_ends = [], []
        for fold in folds:
            a, c, d = fold.train_start, fold.test_start, fold.test_end
            b = min(fold.train_end, c-self.model_config.purge_gap)
            allowed = np.ones(b-a, dtype=bool)
            for end in previous_ends:
                idx = np.arange(a, b)
                allowed &= (idx < end) | (idx >= end+self.model_config.embargo_bars)
            inner_members = members[:, a:b].copy() & allowed[None, :]
            for inner in fold.inner_folds:
                if not allowed[inner.val_start-a:inner.val_end-a].all():
                    raise ValueError("inner validation intersects outer embargo")
            earliest = fold.inner_folds[0]
            clean_mask = np.zeros_like(inner_members)
            clean_end = min(earliest.train_end, earliest.val_start-self.model_config.purge_gap)
            clean_mask[:, earliest.train_start-a:clean_end-a] = inner_members[:, earliest.train_start-a:clean_end-a]
            clean, exclusions = clean_training_candidates({k: v[:, a:b] for k, v in prepared.items()}, clean_mask)
            if not clean:
                raise ValueError("no usable training features; cannot fabricate a fitted demo")
            base_names = tuple(sorted(clean))
            original_support = inner_members & np.isfinite(y[:, a:b]) & np.isfinite(np.stack(list(clean.values()), axis=2)).all(axis=2)
            source = self._source(base_names)  # New source per fold, no cross-fold feedback.
            inner_features = dict(clean)
            inner_available = {k: panel.available_at_ns[k][:, a:b].copy() for k in base_names}
            selected_formulas = {}
            feedback, searches, formula_rows = [], [], []
            local_folds = tuple(SelectionFold(i.train_start-a, i.train_end-a, i.val_start-a, i.val_end-a)
                                for i in fold.inner_folds)
            for round_index in range(cfg.formula_rounds):
                charge("formula_proposal_slots", fold.fold_id, cfg.proposals_per_round)
                if cfg.proposal_mode == "rd_agent":
                    report["network_used"] = True
                try:
                    proposed = source.generate(round_index=round_index, count=cfg.proposals_per_round,
                                               feedback=feedback, data_role="development_inner")
                except Exception as exc:
                    # Never publish provider error bodies/credentials in reports.
                    proposed = []
                    formula_rows.append({"round": round_index, "status": "PROPOSAL_ERROR", "error_type": type(exc).__name__})
                    report["warnings"].append(f"{fold.fold_id}:PROPOSAL_ERROR")
                if len(proposed) > cfg.proposals_per_round:
                    raise ValueError("proposal source exceeded charged slots")
                for spec in proposed:
                    row = {"fold_id": fold.fold_id, "round": round_index, "source": spec.source,
                           "tokens": list(spec.tokens)}
                    try:
                        validate_formula(spec, base_names, limits=FormulaLimits())
                        formula_id = "FORMULA_"+identity(list(spec.tokens))[:20]
                        row["formula_id"] = formula_id
                        if formula_id in selected_formulas or formula_id in clean:
                            row["status"] = "DUPLICATE"
                        else:
                            evaluation = evaluate_formula(spec, clean, inner_available, times[a:b], inner_members)
                            if not np.isfinite(evaluation.values[original_support]).all():
                                # Do not let formulas win merely by deleting hard rows.
                                raise ValueError("formula changes fixed comparison support")
                            inner_features[formula_id] = evaluation.values
                            selected_formulas[formula_id] = spec
                            row["status"] = "ELIGIBLE_FOR_JOINT_DEMO"
                    except Exception as exc:
                        row["status"] = "INVALID_OR_UNAVAILABLE"
                        row["error_type"] = type(exc).__name__
                    formula_rows.append(row)
                evaluator = FixedLightGBMGroupEvaluator(
                    inner_features, y[:, a:b], inner_members, local_folds,
                    signal_at=panel.signal_at[a:b], label_end_at=panel.label_end_at[a:b],
                    selection_end_at=panel.signal_at[c], holdout_start=panel.holdout_start,
                    data_fingerprint=data_hash, data_role="development_inner", config=self.model_config)
                if evaluator.coverage < cfg.coverage_warning_threshold:
                    report["warnings"].append(f"{fold.fold_id}:COVERAGE_BELOW_RESEARCH_THRESHOLD")
                search_config = GroupSearchConfig(
                    max_group_size=min(cfg.max_group_size, len(inner_features)),
                    max_candidates=len(inner_features), max_trials=cfg.search_trials_per_round,
                    forward_budget=cfg.search_trials_per_round, backward_budget=cfg.search_trials_per_round,
                    pair_budget=cfg.search_trials_per_round, min_coverage=1e-12)
                # Finite pairs are not conditioned on singleton significance.
                pairs = tuple((base_names[i], base_names[j]) for i in range(len(base_names))
                              for j in range(i+1, len(base_names)))[:12]
                result = StepwiseGroupSearch(search_config).run(
                    list(inner_features), evaluator, pair_seeds=pairs, data_role="development_inner")
                charge("inner_comparisons_plus_baseline", fold.fold_id, len(result.trials)+1)
                if result.status != "INNER_SELECTION_ONLY":
                    raise ValueError("model/search infrastructure failed: "+result.status)
                searches.append(result.to_dict())
                feedback = [{"formula": list(selected_formulas[name].tokens), "rank": rank+1,
                             "status": "inner_selected"} for rank, name in enumerate(result.selected)
                            if name in selected_formulas]
            fitted = tuple(result.selected)
            fallback = not fitted
            if fallback:
                # A labelled diagnostic fallback, not a passed selection gate.
                fitted = base_names
                report["warnings"].append(f"{fold.fold_id}:NO_RELIABLE_SUBSET_ALL_BASE_FEATURES_DEMO_FALLBACK")
            features_until_test = {k: prepared[k][:, a:d] for k in base_names}
            avail_until_test = {k: panel.available_at_ns[k][:, a:d] for k in base_names}
            outer_columns = {}
            for name in fitted:
                if name in features_until_test:
                    outer_columns[name] = features_until_test[name]
                else:
                    # Fixed formula only. No outer labels or feedback go to source.
                    outer_columns[name] = evaluate_formula(selected_formulas[name], features_until_test,
                        avail_until_test, times[a:d], members[:, a:d]).values
            n, length = y.shape[0], d-a
            x = np.stack([outer_columns[k] for k in fitted], axis=2)
            train_mask = np.zeros((n, length), dtype=bool)
            train_mask[:, :b-a] = original_support
            test_mask = np.zeros((n, length), dtype=bool)
            test_mask[:, c-a:] = members[:, c:d] & np.isfinite(x[:, c-a:]).all(axis=2)
            train, test = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
            if not len(test):
                raise ValueError("no available outer feature rows; cannot fabricate predictions")
            if len(train) < max(20, 3*len(fitted)):
                raise ValueError("insufficient training rows for actual model fitting")
            target_for_fit = np.full((n, length), np.nan)
            target_for_fit.ravel()[train] = y[:, a:d].ravel()[train]
            charge("outer_model_fit", fold.fold_id)
            model = FixedLightGBMBaseline(self.model_config)
            predicted = model._fit_predict(x, target_for_fit, train, test)
            if predicted.shape != test.shape or not np.isfinite(predicted).all():
                raise ValueError("invalid fitted predictions")
            values = np.full((n, length), np.nan)
            values.ravel()[test] = predicted
            evaluated = test_mask & np.isfinite(y[:, a:d])
            eval_idx = np.flatnonzero(evaluated)
            metric = model._oof_score(values.ravel()[eval_idx], y[:, a:d].ravel()[eval_idx], eval_idx, n, length) if len(eval_idx) else None
            if metric is None or metric <= 0:
                report["warnings"].append(f"{fold.fold_id}:OUTER_IC_NONPOSITIVE_OR_UNAVAILABLE")
            denominator = int(members[:, c:d].sum())
            coverage = float(test_mask.sum()/denominator) if denominator else 0.
            if coverage < cfg.coverage_warning_threshold:
                report["warnings"].append(f"{fold.fold_id}:OUTER_COVERAGE_BELOW_RESEARCH_THRESHOLD")
            report["selection_runs"].append({"fold_id": fold.fold_id,
                "selected": list(result.selected), "fitted": list(fitted), "demo_fallback": fallback,
                "excluded_training_features": exclusions, "inner_searches": searches})
            report["formulas"].extend(formula_rows)
            report["model_evidence"]["folds"].append({"fold_id": fold.fold_id,
                "selected_features": list(result.selected), "fitted_features": list(fitted),
                "demo_fallback": fallback,
                "outer_rank_ic": metric, "signal_coverage": coverage,
                "original_member_days": denominator, "prediction_rows": len(test),
                "evaluated_label_rows": len(eval_idx), "train_rows": len(train),
                "training_labels_only": True, "formula_source_saw_outer_feedback": False,
                "test_start": panel.signal_at[c], "test_end": panel.signal_at[d-1],
                "score_hash": sha256(values[:, c-a:].tobytes()).hexdigest()})
            report["model_evidence"]["folds"][-1]["daily_available_stock_counts"] = test_mask[:, c-a:].sum(axis=0).tolist()
            # Keep every stock/date cell, including nonmembers and missing scores;
            # consumers can filter explicitly but cannot mistake omission for coverage.
            dates = pd.DatetimeIndex(panel.signal_at[c:d]).tz_convert("Asia/Shanghai")
            out_values = values[:, c-a:]
            score_frames.append(pd.DataFrame({
                "date": np.tile([t.date().isoformat() for t in dates], n),
                "code": np.repeat(panel.codes, d-c), "score": out_values.ravel(),
                "score_available_at": np.tile([t.isoformat() for t in dates], n),
                "fold_id": fold.fold_id, "is_member": members[:, c:d].ravel(),
                "signal_valid": (members[:, c:d] & np.isfinite(out_values)).ravel()}))
            previous_ends.append(d)
        scores = pd.concat(score_frames, ignore_index=True)
        if len(scores) != report["expected_score_rows"]:
            raise ValueError("score output does not preserve the frozen full outer grid")
        report["selected_features"] = sorted({name for f in report["selection_runs"] for name in f["fitted"]})
        report["warnings"] = sorted(set(report["warnings"]))
        implementation = {
            "joint_stack": self.guard.implementation_identity(),
            "demo_pipeline": sha256(Path(__file__).read_bytes()).hexdigest(),
            "demo_formulas": sha256(Path(__file__).with_name("demo_formulas.py").read_bytes()).hexdigest(),
        }
        report["model_evidence"]["implementation_hashes"] = implementation
        report["fingerprints"] = {
            "model": identity({"params": report["model_evidence"]["params"], "versions": {
                                   "lightgbm": version("lightgbm"), "numpy": np.__version__, "pandas": pd.__version__},
                               "demo_config": asdict(cfg), "research_config": {
                                   k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.model_config).items()},
                               "implementation": implementation}),
            "formula": identity(report["formulas"]), "fold": identity([asdict(f) for f in folds]),
            "data": data_hash}
        return report, scores
