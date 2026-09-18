"""Restricted formula proposals for offline demonstrations, never formal SOTA.

This is NOT the production StackVM: it interprets a small string-token RPN
language over already audited feature arrays, including context features outside
the historical 65-token vocabulary. No formula can execute code, read a file,
request data, or shift an input into the future. Missing/late inputs remain NaN.

Default proposals are deterministic local replay, not LLM output. The optional
RD adapter reuses the existing proposal transport only after explicit network
opt-in; its prompt is restricted to approved identifiers and inner-fold ranks.
The owner must create a NEW source for each outer fold and account for every
returned proposal, including invalid and duplicate proposals, before scoring.
Neither source reads Holdout nor provides profitability/acceptance decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


NAT_NS = np.iinfo(np.int64).min
FEEDBACK_RANK_SEMANTICS = "selected_list_stable_ordinal_not_importance"
OPERATORS = {"ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2,
             "NEG": 1, "ABS": 1, "CS_RANK": 1}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,95}\Z")
_SAFE_STATUSES = frozenset({"inner_candidate", "inner_elite", "inner_selected", "accepted",
                            "rejected", "invalid", "elite"})


class FormulaValidationError(ValueError):
    """A proposal is inadmissible; caller must still record/charge its trial."""


@dataclass(frozen=True)
class FormulaLimits:
    max_tokens: int = 32
    max_depth: int = 7
    max_operators: int = 12

    def __post_init__(self):
        for name, ceiling in (("max_tokens", 128), ("max_depth", 16),
                              ("max_operators", 64)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be a positive integer <= {ceiling}")


@dataclass(frozen=True)
class FormulaSpec:
    # Construction deliberately does not approve RPN structure: malformed
    # proposals must be returned to the owner for rejected-trial accounting.
    formula_id: str
    tokens: tuple[str, ...]
    source: str = "offline_replay"

    def to_dict(self) -> dict:
        return {"formula_id": self.formula_id, "tokens": list(self.tokens),
                "source": self.source}


@dataclass(frozen=True)
class FormulaEvaluation:
    values: np.ndarray
    valid_mask: np.ndarray
    available_at_ns: np.ndarray
    dependencies: tuple[str, ...]


def _feature_manifest(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FormulaValidationError("feature manifest must be an identifier sequence")
    names = tuple(values)
    if any(not isinstance(x, str) or not _IDENTIFIER.fullmatch(x) for x in names):
        raise FormulaValidationError("only audited identifier-shaped feature names are permitted")
    if not names or len(names) > 512 or len(set(names)) != len(names):
        raise FormulaValidationError("feature manifest empty, duplicated or over 512 entries")
    if set(names) & set(OPERATORS):
        raise FormulaValidationError("feature/operator name collision")
    return names


def validate_formula(spec: FormulaSpec, allowed_features: Iterable[str], *,
                     limits: FormulaLimits | None = None) -> tuple[str, ...]:
    """Validate bounded RPN and return its sorted dependency identifiers."""
    limits = limits or FormulaLimits()
    allowed = set(_feature_manifest(allowed_features))
    if not isinstance(spec, FormulaSpec):
        raise FormulaValidationError("expected FormulaSpec")
    if not isinstance(spec.formula_id, str) or not _IDENTIFIER.fullmatch(spec.formula_id):
        raise FormulaValidationError("formula ID must be a safe identifier")
    if not isinstance(spec.tokens, tuple) or not 1 <= len(spec.tokens) <= limits.max_tokens:
        raise FormulaValidationError("formula token count invalid or exceeds limit")
    depths, dependencies, operators = [], set(), 0
    for token in spec.tokens:
        if not isinstance(token, str):
            raise FormulaValidationError("formula tokens must be exact strings")
        if token in allowed:
            depths.append(1)
            dependencies.add(token)
            continue
        if token not in OPERATORS:
            raise FormulaValidationError(f"unknown token or missing dependency: {token!r}")
        arity = OPERATORS[token]
        if len(depths) < arity:
            raise FormulaValidationError("RPN stack underflow")
        depth = 1 + max(depths.pop() for _ in range(arity))
        operators += 1
        if depth > limits.max_depth or operators > limits.max_operators:
            raise FormulaValidationError("formula depth/operator limit exceeded")
        depths.append(depth)
    if len(depths) != 1:
        raise FormulaValidationError("RPN must finish with exactly one value")
    return tuple(sorted(dependencies))


def _rank_column(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile ranks; no constant/NaN-to-zero substitution."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, 1 + np.flatnonzero(ordered[1:] != ordered[:-1])]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat((starts + 1 + ends) / (2.0 * len(values)), ends - starts)
    return ranks


def evaluate_formula(spec: FormulaSpec, features: Mapping[str, np.ndarray],
                     available_at_ns: Mapping[str, np.ndarray],
                     signal_times_ns: np.ndarray, members: np.ndarray, *,
                     limits: FormulaLimits | None = None) -> FormulaEvaluation:
    """Evaluate assets x dates inputs using the same-time original member mask.

    Every required input must have int64 UTC-ns availability <= the exact signal
    instant. NaT, non-finite, zero denominators, overflow, and nonmembers produce
    NaN, never zero. Rank availability includes ALL valid same-date participants;
    a late participant is absent at that instant, never backfilled later.
    Membership availability must already have been enforced by the panel owner.
    This array-only function has no data loading or Holdout-reading capability.
    """
    dependencies = validate_formula(spec, features.keys(), limits=limits)
    membership = np.asarray(members)
    times = np.asarray(signal_times_ns)
    if membership.ndim != 2 or membership.dtype != np.dtype(bool):
        raise FormulaValidationError("members must be boolean assets x dates")
    if (times.ndim != 1 or times.dtype != np.dtype("int64")
            or len(times) != membership.shape[1] or np.any(times == NAT_NS)
            or np.any(times[1:] <= times[:-1])):
        raise FormulaValidationError("signal times must be ordered int64 UTC ns")
    leaves = {}
    for name in dependencies:
        if name not in available_at_ns:
            raise FormulaValidationError(f"missing input availability: {name}")
        value = np.array(features[name], dtype=float, copy=True)
        available = np.asarray(available_at_ns[name])
        if (value.shape != membership.shape or available.shape != value.shape
                or available.dtype != np.dtype("int64")):
            raise FormulaValidationError("feature and int64 availability shapes must match members")
        available = available.copy()
        valid = (membership & np.isfinite(value) & (available != NAT_NS)
                 & (available <= times[None, :]))
        value[~valid], available[~valid] = np.nan, NAT_NS
        leaves[name] = (value, available)
    stack = []
    for token in spec.tokens:
        if token in leaves:
            stack.append(tuple(x.copy() for x in leaves[token]))
            continue
        if OPERATORS[token] == 1:
            value, available = stack.pop()
            if token == "CS_RANK":
                ranked, rank_time = np.full_like(value, np.nan), np.full_like(available, NAT_NS)
                for date_index in range(value.shape[1]):
                    valid = np.isfinite(value[:, date_index])
                    if not valid.any():
                        continue
                    ranked[valid, date_index] = _rank_column(value[valid, date_index])
                    rank_time[valid, date_index] = available[valid, date_index].max()
                value, available = ranked, rank_time
            elif token == "NEG":
                value = -value
            else:
                value = np.abs(value)
        else:
            right, right_time = stack.pop()
            left, left_time = stack.pop()
            available = np.maximum(left_time, right_time)
            input_valid = np.isfinite(left) & np.isfinite(right)
            with np.errstate(all="ignore"):
                if token == "ADD":
                    value = left + right
                elif token == "SUB":
                    value = left - right
                elif token == "MUL":
                    value = left * right
                else:
                    value = np.full_like(left, np.nan)
                    np.divide(left, right, out=value, where=input_valid & (right != 0))
            value[~input_valid] = np.nan
        valid = np.isfinite(value) & membership & (available != NAT_NS)
        value[~valid], available[~valid] = np.nan, NAT_NS
        stack.append((value, available))
    values, available = stack.pop()
    valid = np.isfinite(values) & membership & (available != NAT_NS) & (available <= times[None, :])
    values[~valid], available[~valid] = np.nan, NAT_NS
    return FormulaEvaluation(values, valid, available, dependencies)


def sanitise_inner_feedback(feedback, feature_ids: Iterable[str], *,
                            limits: FormulaLimits | None = None) -> list[dict]:
    """Allow only valid token structure, integer ordinal and a fixed status enum.

    Never stringify arbitrary values: dates, labels, prices, paths, raw arrays,
    free-form explanations and outer-fold scores cannot enter the prompt.
    The legacy `rank` field is only the selected list's stable ordinal, NOT
    feature importance, standalone IC, increment, or a performance ranking.
    The caller is responsible for constructing it ONLY in its inner scope.
    """
    if feedback is None:
        return []
    if not isinstance(feedback, (list, tuple)):
        raise FormulaValidationError("feedback must be an inner-record sequence")
    names = _feature_manifest(feature_ids)
    result = []
    for item in feedback[:12]:
        if not isinstance(item, dict):
            continue
        tokens, rank, status = item.get("formula"), item.get("rank"), item.get("status", "inner_candidate")
        if (not isinstance(tokens, (tuple, list)) or type(rank) is not int
                or not 1 <= rank <= 1_000_000 or not isinstance(status, str)
                or status not in _SAFE_STATUSES):
            continue
        try:
            validate_formula(FormulaSpec("feedback", tuple(tokens)), names, limits=limits)
        except FormulaValidationError:
            continue
        result.append({"formula": list(tokens), "rank": rank, "status": status})
    return result


def _request(round_index, count, data_role):
    if data_role != "development_inner":
        raise FormulaValidationError("formula search permits only development_inner")
    if type(round_index) is not int or not 0 <= round_index <= 100_000:
        raise FormulaValidationError("round_index must be a bounded nonnegative integer")
    if type(count) is not int or not 1 <= count <= 256:
        raise FormulaValidationError("count must be a positive integer <= 256")


class OfflineReplaySource:
    """Deterministic local proposals, explicitly NOT an actual RD/LLM call.

    Optional replay items may be FormulaSpec or token sequences. They are not
    prefiltered by validity, novelty or IC: bad proposals reach trial accounting.
    This object owns no market data, labels, file paths or cross-fold memory.
    """
    source = "offline_replay"

    def __init__(self, feature_ids: Iterable[str], *, replay: Sequence | None = None,
                 limits: FormulaLimits | None = None):
        self.feature_ids = _feature_manifest(feature_ids)
        self.limits = limits or FormulaLimits()
        if replay is None:
            # Stable manifest order, not a ranking from future outcomes.
            # Interleave unary and pair proposals so a small demo allowance
            # already exercises joint structure, not just sign/rank changes.
            pool = []
            for index, name in enumerate(self.feature_ids):
                if len(self.feature_ids) > 1:
                    other = self.feature_ids[(index + 1) % len(self.feature_ids)]
                    for unary, binary in (("CS_RANK", "MUL"), ("NEG", "ADD"), ("ABS", "SUB")):
                        pool.extend([(name, unary), (name, "CS_RANK", other, "CS_RANK", binary)])
                    pool.append((name, "CS_RANK", other, "CS_RANK", "DIV"))
                else:
                    pool.extend([(name, "CS_RANK"), (name, "NEG"), (name, "ABS")])
            self._replay = tuple(pool)
        else:
            if isinstance(replay, (str, bytes)) or len(replay) > 65_536:
                raise FormulaValidationError("replay must be a bounded proposal sequence")
            self._replay = tuple(item if isinstance(item, FormulaSpec) else tuple(item)
                                 for item in replay)
        self.audit_records: list[dict] = []

    def generate(self, *, round_index: int, count: int, feedback=None,
                 data_role: str = "development_inner") -> list[FormulaSpec]:
        _request(round_index, count, data_role)
        clean = sanitise_inner_feedback(feedback, self.feature_ids, limits=self.limits)
        start = round_index * count
        result = []
        for index, item in enumerate(self._replay[start:start + count], start=start):
            tokens = item.tokens if isinstance(item, FormulaSpec) else item
            digest = sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest()[:12]
            result.append(FormulaSpec(f"DEMO_RPN_{index}_{digest}", tuple(tokens), self.source))
        self.audit_records.append({"source": self.source, "round_index": round_index,
                                   "requested": count, "returned": len(result),
                                   "inner_feedback": clean,
                                   "feedback_rank_semantics": FEEDBACK_RANK_SEMANTICS,
                                   "network_called": False})
        return result


class RestrictedRDFormulaSource:
    """Opt-in existing RD transport with the demo-only audited vocabulary.

    No transport object is imported/constructed until generate(). Credentials
    are accessed only by the existing transport after explicit allow_network.
    Model, endpoint and credentials remain governed by that existing adapter.
    Returned malformed/duplicate proposals are preserved for rejected trials;
    they are never executed by the production StackVM.
    """
    source = "rd_agent_network"

    def __init__(self, feature_ids: Iterable[str], *, project_root, model: str,
                 allow_network: bool = False, limits: FormulaLimits | None = None):
        if allow_network is not True:
            raise FormulaValidationError("RD network proposals require explicit allow_network=True")
        self.feature_ids = _feature_manifest(feature_ids)
        self.limits = limits or FormulaLimits()
        self.project_root, self.model = project_root, model
        if not isinstance(model, str) or not model.strip():
            raise FormulaValidationError("explicit RD model required")
        self.audit_records: list[dict] = []

    def generate(self, *, round_index: int, count: int, feedback=None,
                 data_role: str = "development_inner") -> list[FormulaSpec]:
        _request(round_index, count, data_role)
        clean = sanitise_inner_feedback(feedback, self.feature_ids, limits=self.limits)
        # Lazy import keeps offline replay free of API/credential/torch actions.
        from model_core.rd_agent_generator import FormulaCandidate, RDFormulaGenerator, RDFormulaLimits

        owner = self

        class _RestrictedTransport(RDFormulaGenerator):
            def __init__(self):
                super().__init__(owner.project_root, model=owner.model, limits=RDFormulaLimits())
                self._proposal_index = 0

            def _system_prompt(self):
                return ("Propose bounded RPN factor formulas as JSON only. This is an offline-demo "
                        "research proposal stage, not trading advice. Only supplied feature/operator "
                        "identifiers are permitted. Never request data, paths, dates or executable code. "
                        "Features push one value; operators pop their arity and push one; finish with one. "
                        "Feedback rank is only a stable selected-list ordinal, not feature importance, "
                        "IC, marginal contribution, or performance ordering. Do not infer quality from it. "
                        + json.dumps({"features": owner.feature_ids, "operators": OPERATORS,
                                      "limits": asdict(owner.limits)})
                        + ' Return {"formulas":[{"tokens":["FEATURE","CS_RANK"]}]}.')

            def _user_prompt(self, **kwargs):
                # Do not invoke the generic prose/research-context sanitizer.
                return json.dumps({"round": round_index, "requested_formula_count": count,
                                   "inner_structure_feedback": clean,
                                   "feedback_rank_semantics": FEEDBACK_RANK_SEMANTICS}, allow_nan=False)

            @staticmethod
            def _extract_json(value):
                result = RDFormulaGenerator._extract_json(value)
                if isinstance(result.get("formulas"), list):
                    result["formulas"] = [row if isinstance(row, dict) else {"tokens": []}
                                           for row in result["formulas"]]
                return result

            def _validate(self, raw_tokens):
                # Preserve structural rejections AND duplicates for the owner's
                # explicit proposal ledger; do not preselect service outputs.
                self._proposal_index += 1
                tokens = raw_tokens if (isinstance(raw_tokens, list)
                                         and all(isinstance(t, str) for t in raw_tokens)) else []
                return FormulaCandidate(tokens=[self._proposal_index], token_names=tokens)

        record = {"source": self.source, "round_index": round_index, "requested": count,
                  "inner_feedback": clean, "feedback_rank_semantics": FEEDBACK_RANK_SEMANTICS,
                  "network_called": False, "status": "PENDING"}
        self.audit_records.append(record)
        try:
            transport = _RestrictedTransport()
            record["network_called"] = True
            candidates = transport.generate(round_index=round_index, count=count, feedback=clean)
            result = [FormulaSpec(f"DEMO_RD_R{round_index}_{index}", tuple(candidate.token_names), self.source)
                      for index, candidate in enumerate(candidates)]
            record.update(status="RETURNED", returned=len(result))
            return result
        except Exception:
            # Never persist provider error bodies or credentials in demo audit.
            record.update(status="PROPOSAL_SERVICE_FAILED", returned=0)
            raise RuntimeError("RD proposal service failed; inspect provider configuration privately") from None
