from __future__ import annotations

import numpy as np
import torch

from .config import ResearchConfig
from .schemas import LGBMIncrementResult
from .validation_unit import _rank_ic


class LightGBMUnavailable(RuntimeError):
    pass


class FixedLightGBMBaseline:
    """Deterministic OOF marginal test; never tunes per candidate."""

    def __init__(self, config: ResearchConfig):
        self.config = config

    @staticmethod
    def _module():
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise LightGBMUnavailable(
                "缺少必需依赖 lightgbm；请执行 python -m pip install -r requirements.txt"
            ) from exc
        return lgb

    def _params(self) -> dict:
        return {
            "objective": "regression_l2",
            "metric": "None",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": self.config.lgbm_num_leaves,
            "learning_rate": self.config.lgbm_learning_rate,
            "min_child_samples": self.config.lgbm_min_child_samples,
            "feature_fraction": self.config.lgbm_feature_fraction,
            "bagging_fraction": self.config.lgbm_bagging_fraction,
            "bagging_freq": 0,
            "seed": self.config.random_seed,
            "feature_fraction_seed": self.config.random_seed,
            "bagging_seed": self.config.random_seed,
            "data_random_seed": self.config.random_seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": self.config.lgbm_num_threads,
        }

    @staticmethod
    def _features(matrix: torch.Tensor | None, candidate: torch.Tensor | None) -> np.ndarray | None:
        rows: list[torch.Tensor] = []
        if matrix is not None:
            rows.extend(matrix[:, index, :] for index in range(matrix.shape[1]))
        if candidate is not None:
            rows.append(candidate)
        if not rows:
            return None
        return torch.stack(rows, dim=2).detach().cpu().to(torch.float32).numpy()

    def _fit_predict(
        self, features: np.ndarray | None, target: np.ndarray, train: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        labels = target.reshape(-1)
        if features is None:
            return np.full(valid.size, float(np.mean(labels[train])), dtype=np.float64)
        lgb = self._module()
        flat = features.reshape(-1, features.shape[2])
        if not (
            np.all(np.isfinite(labels[train]))
            and np.all(np.isfinite(flat[train]))
            and np.all(np.isfinite(flat[valid]))
        ):
            raise ValueError("baseline and challenge must use one shared finite sample")
        if train.size < max(20, features.shape[2] * 3):
            return np.full(valid.size, float(np.nanmean(labels[train])), dtype=np.float64)
        dataset = lgb.Dataset(flat[train], label=labels[train], free_raw_data=True)
        model = lgb.train(
            self._params(), dataset, num_boost_round=self.config.lgbm_num_boost_round
        )
        return np.asarray(model.predict(flat[valid]), dtype=np.float64)

    @staticmethod
    def _oof_score(
        prediction: np.ndarray,
        labels: np.ndarray,
        valid_indices: np.ndarray,
        n_assets: int,
        time_length: int,
    ) -> float:
        if n_assets <= 1:
            return _rank_ic(prediction, labels)
        times = valid_indices % time_length
        daily = []
        for time_index in np.unique(times):
            mask = times == time_index
            if mask.sum() >= 3:
                daily.append(_rank_ic(prediction[mask], labels[mask]))
        return float(np.mean(daily)) if daily else 0.0

    def compare(
        self,
        sota_matrix: torch.Tensor | None,
        candidate: torch.Tensor,
        target: torch.Tensor,
        folds: list[dict],
        candidate_invalid_mask: torch.Tensor | None = None,
    ) -> LGBMIncrementResult:
        base = self._features(sota_matrix, None)
        challenge = self._features(sota_matrix, candidate)
        y = target.detach().cpu().to(torch.float32).numpy()
        n, t = y.shape
        challenge_flat = challenge.reshape(-1, challenge.shape[2])
        common_finite = np.isfinite(y.reshape(-1)) & np.all(
            np.isfinite(challenge_flat), axis=1
        )
        if candidate_invalid_mask is not None:
            invalid = candidate_invalid_mask.detach().cpu().numpy().astype(bool)
            common_finite &= ~invalid[:, :t].reshape(-1)
        base_deltas: list[float] = []
        base_scores: list[float] = []
        challenge_scores: list[float] = []
        sample_count = 0
        previous_validation_ends: list[int] = []
        for fold in folds:
            existing_gap = max(0, int(fold["val_start"]) - int(fold["train_end"]))
            extra_purge = max(0, self.config.purge_gap - existing_gap)
            train_end = max(
                int(fold["train_start"]), int(fold["train_end"]) - extra_purge
            )
            train_times = np.arange(int(fold["train_start"]), train_end)
            if self.config.embargo_bars > 0 and previous_validation_ends:
                embargoed = np.concatenate([
                    np.arange(end, min(t, end + self.config.embargo_bars))
                    for end in previous_validation_ends
                ])
                train_times = train_times[~np.isin(train_times, embargoed)]
            valid_times = np.arange(int(fold["val_start"]), int(fold["val_end"]))
            if train_times.size < 20 or valid_times.size < 3:
                previous_validation_ends.append(int(fold["val_end"]))
                continue
            train = (np.arange(n)[:, None] * t + train_times[None, :]).reshape(-1)
            valid = (np.arange(n)[:, None] * t + valid_times[None, :]).reshape(-1)
            train = train[common_finite[train]]
            valid = valid[common_finite[valid]]
            if train.size < 20 or valid.size < 3:
                previous_validation_ends.append(int(fold["val_end"]))
                continue
            bp = self._fit_predict(base, y, train, valid)
            cp = self._fit_predict(challenge, y, train, valid)
            labels = y.reshape(-1)[valid]
            bscore = self._oof_score(bp, labels, valid, n, t)
            cscore = self._oof_score(cp, labels, valid, n, t)
            base_scores.append(bscore)
            challenge_scores.append(cscore)
            base_deltas.append(cscore - bscore)
            sample_count += valid.size
            previous_validation_ends.append(int(fold["val_end"]))
        if not base_deltas:
            return LGBMIncrementResult(0.0, 0.0, 0.0, 0.0, [], 0)
        return LGBMIncrementResult(
            baseline_metric=float(np.mean(base_scores)),
            challenge_metric=float(np.mean(challenge_scores)),
            delta=float(np.mean(base_deltas)),
            positive_fold_ratio=float(np.mean(np.asarray(base_deltas) > 0)),
            fold_deltas=[float(value) for value in base_deltas],
            oof_samples=sample_count,
        )
