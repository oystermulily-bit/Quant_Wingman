"""Frozen stage-3 configuration and whole-date chronological splitting."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


SPLIT_PROTOCOL_VERSION = "hs300_expanding5_purge20_embargo5_holdout_v1"


@dataclass(frozen=True)
class Stage3Config:
    horizons: tuple[int, ...] = (1, 3, 5)
    execution_lag_bars: int = 1
    n_folds: int = 5
    purge_bars: int = 20
    embargo_bars: int = 5
    holdout_fraction: float = 0.10
    holdout_min_dates: int = 480
    holdout_min_years: int = 2
    initial_train_fraction: float = 0.40
    top_n: int = 20
    target_weight: float = 0.05
    liquidity_median_amount_20d: float = 20_000_000.0
    buy_cost_rate: float = 0.00076
    sell_cost_rate: float = 0.00126
    annual_trading_days: int = 239
    bootstrap_block_days: int = 20
    bootstrap_samples: int = 2000
    random_seed: int = 20260812

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if self.purge_bars < max(self.horizons) + self.execution_lag_bars:
            raise ValueError("purge_bars does not cover the longest label interval")
        if self.top_n * self.target_weight > 1.0 + 1e-12:
            raise ValueError("reference target weights exceed 100%")
        if not 0.0 < self.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be between 0 and 0.5")

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train_dates: tuple[pd.Timestamp, ...]
    purge_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    embargo_dates: tuple[pd.Timestamp, ...]

    def assert_disjoint(self) -> None:
        groups = [
            set(self.train_dates),
            set(self.purge_dates),
            set(self.validation_dates),
            set(self.embargo_dates),
        ]
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                if groups[left] & groups[right]:
                    raise AssertionError(f"fold {self.fold_id} roles overlap")


@dataclass(frozen=True)
class SplitPlan:
    protocol: str
    config_hash: str
    all_date_count: int
    development_dates: tuple[pd.Timestamp, ...]
    holdout_start: pd.Timestamp
    holdout_date_count: int
    holdout_date_hash: str
    folds: tuple[FoldSplit, ...]

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "config_hash": self.config_hash,
            "all_date_count": self.all_date_count,
            "development_date_count": len(self.development_dates),
            "development_start": self.development_dates[0].date().isoformat(),
            "development_end": self.development_dates[-1].date().isoformat(),
            "holdout_start": self.holdout_start.date().isoformat(),
            "holdout_date_count": self.holdout_date_count,
            "holdout_date_hash": self.holdout_date_hash,
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_dates[0].date().isoformat(),
                    "train_end": fold.train_dates[-1].date().isoformat(),
                    "train_dates": len(fold.train_dates),
                    "purge_start": fold.purge_dates[0].date().isoformat(),
                    "purge_end": fold.purge_dates[-1].date().isoformat(),
                    "purge_dates": len(fold.purge_dates),
                    "validation_start": fold.validation_dates[0].date().isoformat(),
                    "validation_end": fold.validation_dates[-1].date().isoformat(),
                    "validation_dates": len(fold.validation_dates),
                    "embargo_dates": len(fold.embargo_dates),
                }
                for fold in self.folds
            ],
        }

    @property
    def validation_dates(self) -> pd.DatetimeIndex:
        values = sorted({date for fold in self.folds for date in fold.validation_dates})
        return pd.DatetimeIndex(values)

    def assert_panel_dates(self, frame: pd.DataFrame) -> None:
        if "date" not in frame:
            raise ValueError("frame has no date column")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]).dt.normalize().unique())
        if any(date >= self.holdout_start for date in dates):
            raise PermissionError("development view contains holdout dates")


class DateSplitProtocol:
    """Reserve the final holdout, then create five expanding OOF folds."""

    def __init__(self, config: Stage3Config | None = None) -> None:
        self.config = config or Stage3Config()

    @staticmethod
    def _normalise_dates(dates: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
        index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce"))
        if index.tz is not None:
            index = index.tz_convert("Asia/Shanghai").tz_localize(None)
        index = index.dropna().normalize().drop_duplicates().sort_values()
        if len(index) < 2:
            raise ValueError("not enough unique trading dates")
        return index

    def build(self, dates: Iterable[pd.Timestamp]) -> SplitPlan:
        dates = self._normalise_dates(dates)
        cfg = self.config
        fraction_count = int(math.ceil(len(dates) * cfg.holdout_fraction))
        if cfg.holdout_min_years > 0:
            threshold = dates[-1] - pd.DateOffset(years=cfg.holdout_min_years)
            years_count = int((dates >= threshold).sum())
        else:
            years_count = 0
        holdout_count = max(fraction_count, cfg.holdout_min_dates, years_count)
        if holdout_count >= len(dates):
            raise ValueError(
                f"holdout requires {holdout_count} dates but dataset has {len(dates)}"
            )
        development = dates[:-holdout_count]
        holdout = dates[-holdout_count:]
        minimum_development = (
            cfg.purge_bars + cfg.n_folds * 10 + (cfg.n_folds - 1) * cfg.embargo_bars
        )
        if len(development) < minimum_development:
            raise ValueError(
                f"development dates {len(development)} < required {minimum_development}"
            )

        initial = max(
            cfg.purge_bars + 20,
            int(math.floor(len(development) * cfg.initial_train_fraction)),
        )
        remaining = len(development) - initial - (cfg.n_folds - 1) * cfg.embargo_bars
        validation_size = remaining // cfg.n_folds
        if validation_size < 5:
            raise ValueError("validation windows are too short")

        folds: list[FoldSplit] = []
        past_embargo: set[pd.Timestamp] = set()
        for fold_id in range(cfg.n_folds):
            start = initial + fold_id * (validation_size + cfg.embargo_bars)
            end = start + validation_size
            if fold_id == cfg.n_folds - 1:
                end = len(development)
            purge_start = start - cfg.purge_bars
            candidate_train = development[:purge_start]
            train = tuple(date for date in candidate_train if date not in past_embargo)
            purge = tuple(development[purge_start:start])
            validation = tuple(development[start:end])
            embargo_end = min(len(development), end + cfg.embargo_bars)
            embargo = tuple(development[end:embargo_end]) if fold_id < cfg.n_folds - 1 else ()
            fold = FoldSplit(fold_id, train, purge, validation, embargo)
            fold.assert_disjoint()
            if len(purge) != cfg.purge_bars:
                raise AssertionError("requested purge was silently reduced")
            if train[-1] >= validation[0]:
                raise AssertionError("training is not strictly before validation")
            folds.append(fold)
            past_embargo.update(embargo)

        holdout_serial = "\n".join(date.date().isoformat() for date in holdout)
        return SplitPlan(
            protocol=SPLIT_PROTOCOL_VERSION,
            config_hash=cfg.fingerprint(),
            all_date_count=len(dates),
            development_dates=tuple(development),
            holdout_start=holdout[0],
            holdout_date_count=len(holdout),
            holdout_date_hash=hashlib.sha256(holdout_serial.encode("ascii")).hexdigest(),
            folds=tuple(folds),
        )
