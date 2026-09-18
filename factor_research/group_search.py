"""Bounded inner-Development group search; never writes or promotes SOTA.

The evaluator must score every subset on one fixed inner-validation support.
An inner selection is a research candidate, NOT an independent validation result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Callable


@dataclass(frozen=True)
class GroupSearchConfig:
    max_group_size: int = 5
    max_candidates: int = 65
    max_pair_seeds: int = 12
    max_trials: int = 480
    forward_budget: int = 320
    backward_budget: int = 64
    pair_budget: int = 96
    min_add_delta: float = 0.001
    removal_tolerance: float = 0.0005
    min_positive_fold_ratio: float = 0.60
    min_coverage: float = 0.85
    min_folds: int = 3

    def __post_init__(self):
        for name in ("max_group_size", "max_candidates", "max_trials", "min_folds"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_group_size > self.max_candidates:
            raise ValueError("group cap exceeds candidate cap")
        for name in ("max_pair_seeds", "forward_budget", "backward_budget", "pair_budget"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")
        for name in ("min_add_delta", "removal_tolerance"):
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")
        if self.min_add_delta <= self.removal_tolerance:
            raise ValueError("add threshold must exceed deletion tolerance")
        for name in ("min_positive_fold_ratio", "min_coverage"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class GroupEvaluation:
    # Both IDs must identify the same data / model / folds / finite sample for
    # all subsets. The sample must not shrink whenever a candidate is added.
    context_id: str
    sample_id: str
    fold_scores: tuple[float, ...]
    coverage: float
    valid: bool = True

    @property
    def mean_score(self) -> float:
        return sum(self.fold_scores) / len(self.fold_scores)


@dataclass
class GroupSearchResult:
    selected: tuple[str, ...]
    initial: tuple[str, ...]
    stop_reason: str
    config: dict
    context_id: str
    sample_id: str
    trials: list[dict] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    evaluation_calls: int = 0
    cache_hits: int = 0
    status: str = "INNER_SELECTION_ONLY"
    stage5_allowed: bool = False
    sota_promoted: bool = False
    holdout_read: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class _BudgetStop(Exception):
    pass


class StepwiseGroupSearch:
    """Forward additions, bounded atomic pairs, and floating backward removal.

    Every proposed comparison costs one trial, even on a cache hit or failure.
    A sweep is all-or-nothing with respect to budget: never pick the best of an
    arbitrary prefix. Scoring failures are recorded, not silently promoted.
    """

    def __init__(self, config: GroupSearchConfig | None = None):
        self.config = config or GroupSearchConfig()

    def run(
        self,
        candidates: list[str],
        evaluate: Callable[[tuple[str, ...]], GroupEvaluation],
        *,
        initial: tuple[str, ...] = (),
        pair_seeds: tuple[tuple[str, str], ...] = (),
        data_role: str,
    ) -> GroupSearchResult:
        if data_role != "development_inner":
            raise ValueError("search is allowed only on development_inner")
        cfg = self.config
        if any(not isinstance(x, str) or not x.strip() for x in candidates):
            raise ValueError("candidate IDs must be nonempty strings")
        universe = tuple(sorted(set(candidates)))
        if not universe or len(universe) > cfg.max_candidates:
            raise ValueError("candidate manifest empty or exceeds frozen cap")
        current = tuple(sorted(set(initial)))
        if len(current) != len(initial) or not set(current) <= set(universe):
            raise ValueError("invalid initial group")
        if len(current) > cfg.max_group_size:
            raise ValueError("initial group exceeds cap (baseline columns count)")
        pairs = set()
        for pair in pair_seeds:
            if len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= set(universe):
                raise ValueError("pair seeds must be two distinct manifest factors")
            pairs.add(tuple(sorted(pair)))
        if len(pairs) > cfg.max_pair_seeds:
            raise ValueError("too many preregistered pairs")
        pairs = tuple(sorted(pairs))
        cache: dict[tuple[str, ...], GroupEvaluation | Exception] = {}
        calls = 0
        hits = 0

        def score(group):
            nonlocal calls, hits
            if group in cache:
                hits += 1
            else:
                calls += 1
                try:
                    value = evaluate(group)
                    if not isinstance(value, GroupEvaluation):
                        raise ValueError("evaluator must return GroupEvaluation")
                    if (not value.context_id or not value.sample_id
                            or len(value.fold_scores) < cfg.min_folds
                            or not all(isfinite(x) for x in value.fold_scores)
                            or not isfinite(value.coverage) or not 0 <= value.coverage <= 1):
                        raise ValueError("invalid or insufficient group evidence")
                    cache[group] = value
                except Exception as exc:
                    cache[group] = exc
            value = cache[group]
            if isinstance(value, Exception):
                raise ValueError(str(value)) from value
            return value

        baseline = score(current)
        result = GroupSearchResult(
            selected=current, initial=current, stop_reason="",
            config=asdict(cfg), context_id=baseline.context_id, sample_id=baseline.sample_id,
        )
        if not baseline.valid or baseline.coverage < cfg.min_coverage:
            result.status = "INSUFFICIENT_EVIDENCE"
            result.stop_reason = "BASELINE_SUPPORT_INSUFFICIENT"
            result.evaluation_calls = calls
            return result
        visited = {current}
        # Deletions are bounded against this accepted high-water mark, not only
        # their immediate predecessor; tolerance cannot accumulate by pruning.
        anchor = baseline
        used = {"forward": 0, "backward": 0, "pair": 0}

        def sweep(groups, phase):
            groups = sorted(set(groups) - visited)
            if not groups:
                return []
            limit = getattr(cfg, phase + "_budget")
            if len(result.trials) + len(groups) > cfg.max_trials or used[phase] + len(groups) > limit:
                raise _BudgetStop(phase.upper() + "_BUDGET_EXHAUSTED")
            eligible = []
            for group in groups:
                used[phase] += 1
                trial = {"index": len(result.trials) + 1, "phase": phase,
                         "baseline": current, "candidate": group, "eligible": False}
                result.trials.append(trial)
                try:
                    value = score(group)
                    if (value.context_id != baseline.context_id or value.sample_id != baseline.sample_id
                            or len(value.fold_scores) != len(baseline.fold_scores)
                            or value.coverage != baseline.coverage):
                        raise ValueError("context/folds/common finite support changed")
                    deltas = tuple(a - b for a, b in zip(value.fold_scores, baseline.fold_scores))
                    delta = sum(deltas) / len(deltas)
                    positive = sum(x > 0 for x in deltas) / len(deltas)
                    trial.update(mean_score=value.mean_score, delta=delta,
                                 fold_deltas=deltas, positive_fold_ratio=positive,
                                 coverage=value.coverage)
                    ok = value.valid and value.coverage >= cfg.min_coverage
                    if phase == "backward":
                        losses = tuple(b - a for a, b in zip(value.fold_scores, anchor.fold_scores))
                        ok = ok and (sum(losses) / len(losses) <= cfg.removal_tolerance + 1e-12
                                     and sum(x <= cfg.removal_tolerance + 1e-12 for x in losses)
                                     / len(losses) >= cfg.min_positive_fold_ratio)
                        trial["loss_vs_high_water"] = sum(losses) / len(losses)
                    else:
                        ok = ok and delta >= cfg.min_add_delta and positive >= cfg.min_positive_fold_ratio
                    trial["eligible"] = bool(ok)
                    if ok:
                        eligible.append((group, value))
                except Exception as exc:
                    trial["error"] = f"{type(exc).__name__}: {exc}"
            return eligible

        def accept(options, phase):
            nonlocal current, baseline, anchor
            # Comparable fixed scores, then fewer factors, then canonical IDs.
            group, value = sorted(options, key=lambda x: (-x[1].mean_score, len(x[0]), x[0]))[0]
            result.transitions.append({"phase": phase, "before": current, "after": group,
                                       "before_score": baseline.mean_score,
                                       "after_score": value.mean_score})
            current, baseline = group, value
            visited.add(current)
            if value.mean_score > anchor.mean_score:
                anchor = value

        try:
            while True:
                # Also runs at the cap, and for a caller-supplied initial group.
                while current:
                    options = sweep([tuple(x for x in current if x != remove) for remove in current], "backward")
                    if not options:
                        break
                    accept(options, "backward")
                if len(current) >= cfg.max_group_size:
                    result.stop_reason = "GROUP_CAP_REACHED"
                    break
                remaining = [x for x in universe if x not in current]
                options = sweep([tuple(sorted((*current, x))) for x in remaining], "forward")
                if options:
                    accept(options, "forward")
                    continue
                # Pairs can succeed even when neither singleton passes. They
                # consume TWO slots, not one synthetic group slot.
                options = []
                if len(current) + 2 <= cfg.max_group_size:
                    options = sweep([tuple(sorted((*current, *pair))) for pair in pairs
                                     if not set(pair) & set(current)], "pair")
                if options:
                    accept(options, "pair")
                    continue
                result.stop_reason = "NO_RELIABLE_INCREMENT"
                break
        except _BudgetStop as exc:
            result.stop_reason = str(exc)
        result.selected = current
        result.evaluation_calls = calls
        result.cache_hits = hits
        if any("error" in trial for trial in result.trials):
            # Missing backend / invalid evidence is not evidence that every
            # factor lacks increment. Preserve any provisional group for audit
            # but distinguish incomplete search from a clean negative result.
            result.status = "INNER_SELECTION_INCOMPLETE"
            if result.stop_reason == "NO_RELIABLE_INCREMENT":
                result.stop_reason = "EVALUATION_ERRORS"
        return result
