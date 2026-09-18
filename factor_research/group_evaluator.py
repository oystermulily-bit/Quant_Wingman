"""Fixed-LightGBM subset scorer for a bounded *inner* Development slice.

No file reader, SOTA writer, portfolio simulator or Holdout capability is owned
by this adapter. Supply only the permitted inner slice; outer tests are separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .group_search import GroupEvaluation
from .lgbm_baseline import FixedLightGBMBaseline


@dataclass(frozen=True)
class SelectionFold:
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def _utc(values) -> np.ndarray:
    stamps = [pd.Timestamp(x) for x in values]
    if any(pd.isna(x) or x.tzinfo is None for x in stamps):
        raise ValueError("explicit timezone-aware input times required")
    return np.asarray([x.tz_convert("UTC").value for x in stamps], dtype=np.int64)


def clean_training_candidates(features: dict[str, np.ndarray], train_mask: np.ndarray):
    """Remove only unusable/constant/exactly-duplicated TRAINING outputs.

    Low IC is deliberately not a filter. Near-correlation is not automatic
    deletion. This must be rerun inside each outer-training scope, never on its
    future test data. Canonical IDs decide the representative deterministically.
    """
    mask = np.asarray(train_mask, dtype=bool)
    if not mask.any():
        raise ValueError("empty training mask")
    retained, excluded = {}, {}
    for name in sorted(features):
        value = np.asarray(features[name], dtype=float)
        if value.shape != mask.shape:
            raise ValueError("feature / training mask shape mismatch")
        sample = value[mask]
        finite = sample[np.isfinite(sample)]
        if finite.size < 2 or np.unique(finite).size < 2:
            excluded[name] = "unusable_or_constant_on_training"
            continue
        equivalent = next((other for other in retained
                           if np.array_equal(sample, retained[other][mask], equal_nan=True)), None)
        if equivalent is not None:
            excluded[name] = "training_output_duplicate_of:" + equivalent
        else:
            retained[name] = value.copy()
    return retained, excluded


class FixedLightGBMGroupEvaluator:
    """Score subsets on one immutable, shared finite panel and fixed folds.

    Freeze/clean the shortlist first. Coverage uses ALL original member-days
    in the inner validation windows, including missing labels/features. If the
    all-shortlist common support is inadequate, stop; never silently shrink the
    universe or fill missing observations to rescue coverage.
    """

    def __init__(
        self,
        features: dict[str, np.ndarray],
        target: np.ndarray,
        member_mask: np.ndarray,
        folds: tuple[SelectionFold, ...],
        *,
        signal_at,
        label_end_at,
        selection_end_at: str,
        holdout_start: str,
        data_fingerprint: str,
        data_role: str,
        config: ResearchConfig | None = None,
    ):
        if data_role != "development_inner":
            raise ValueError("outer/holdout data forbidden in selection evaluator")
        if not features or not data_fingerprint:
            raise ValueError("frozen feature manifest and data fingerprint required")
        self.config = config or ResearchConfig()
        if self.config.lgbm_feature_fraction != 1 or self.config.lgbm_bagging_fraction != 1:
            raise ValueError("fixed evaluation requires full feature/sample fractions")
        self.model = FixedLightGBMBaseline(self.config)
        self.names = tuple(sorted(features))
        self.y = np.array(target, dtype=float, copy=True)
        if np.asarray(member_mask).dtype != np.dtype(bool):
            raise ValueError("original member mask must be explicit boolean, not nullable/numeric")
        self.members = np.array(member_mask, dtype=bool, copy=True)
        if self.y.ndim != 2 or self.members.shape != self.y.shape:
            raise ValueError("target and original member mask must be assets x dates")
        self.n, self.t = self.y.shape
        if any(np.shape(features[x]) != self.y.shape for x in self.names):
            raise ValueError("feature panel shape mismatch")
        self.x = np.stack([np.asarray(features[x], dtype=float) for x in self.names], axis=2)
        times, ends = _utc(signal_at), _utc(label_end_at)
        boundary, holdout = _utc([selection_end_at, holdout_start])
        if len(times) != self.t or len(ends) != self.t or not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly ordered, one complete date per column")
        if boundary > holdout or np.any(times >= boundary) or np.any(ends >= boundary):
            raise ValueError("input or labels reach outer evaluation / Holdout boundary")
        if np.any(ends < times):
            raise ValueError("label end precedes signal")
        finite = self.members & np.isfinite(self.y) & np.isfinite(self.x).all(axis=2)
        self.index_pairs = []
        validation_mask = np.zeros(self.t, dtype=bool)
        previous_ends = []
        for fold in folds:
            a, b, c, d = fold.train_start, fold.train_end, fold.val_start, fold.val_end
            if any(type(v) is not int for v in (a, b, c, d)) or not 0 <= a < b <= c < d <= self.t:
                raise ValueError("invalid forward fold boundaries")
            if previous_ends and c < previous_ends[-1]:
                raise ValueError("validation folds overlap or are out of order")
            b = min(b, c - self.config.purge_gap)
            train_times = np.arange(a, max(a, b))
            for end in previous_ends:
                train_times = train_times[(train_times < end) | (train_times >= end + self.config.embargo_bars)]
            if len(train_times) == 0 or np.any(ends[train_times] >= times[c]):
                raise ValueError("training labels overlap validation signal availability")
            validation_mask[c:d] = True
            train = (np.arange(self.n)[:, None] * self.t + train_times).ravel()
            valid = (np.arange(self.n)[:, None] * self.t + np.arange(c, d)).ravel()
            train, valid = train[finite.ravel()[train]], valid[finite.ravel()[valid]]
            if len(train) < max(20, 3 * len(self.names)) or len(valid) < 10:
                raise ValueError("insufficient fixed finite fold support")
            # Every validation day needs a usable cross-section; do not quietly
            # drop low-coverage dates from the average IC.
            if (finite[:, c:d].sum(axis=0) < 10).any():
                raise ValueError("a validation day has fewer than 10 usable stocks")
            for time in range(c, d):
                if np.unique(self.y[finite[:, time], time]).size < 2:
                    raise ValueError("a validation day has degenerate return labels")
            self.index_pairs.append((train, valid))
            previous_ends.append(d)
        if len(self.index_pairs) < 3:
            raise ValueError("at least three inner forward folds required")
        denominator = int(self.members[:, validation_mask].sum())
        if denominator == 0:
            raise ValueError("empty original member-day denominator")
        self.coverage = float(finite[:, validation_mask].sum() / denominator)
        self.sample_id = sha256(
            self.members.tobytes() + finite.tobytes() + validation_mask.tobytes()
            + b"".join(train.tobytes() + valid.tobytes() for train, valid in self.index_pairs)
        ).hexdigest()
        config_identity = {**self.model._params(), "rounds": self.config.lgbm_num_boost_round,
                           "purge": self.config.purge_gap, "embargo": self.config.embargo_bars}
        context = {"protocol": "GROUP_STEPWISE_S1_DRAFT", "data": data_fingerprint,
                   "features": self.names, "folds": [asdict(x) for x in folds],
                   "params": config_identity, "role": data_role, "sample": self.sample_id}
        self.context_id = sha256(
            json.dumps(context, sort_keys=True).encode()
            + self.x.tobytes() + self.y.tobytes() + times.tobytes() + ends.tobytes()
        ).hexdigest()
        # Prevent mutations through the supplied arrays or evaluator attributes.
        self.x.flags.writeable = self.y.flags.writeable = self.members.flags.writeable = False

    def __call__(self, group: tuple[str, ...]) -> GroupEvaluation:
        if len(set(group)) != len(group) or not set(group) <= set(self.names):
            raise ValueError("group not in frozen manifest")
        cols = [self.names.index(name) for name in group]
        features = self.x[:, :, cols] if cols else None
        scores = []
        for train, valid in self.index_pairs:
            prediction = self.model._fit_predict(features, self.y, train, valid)
            if not np.isfinite(prediction).all():
                raise ValueError("nonfinite model predictions")
            scores.append(self.model._oof_score(prediction, self.y.ravel()[valid], valid, self.n, self.t))
        return GroupEvaluation(self.context_id, self.sample_id, tuple(scores), self.coverage)
