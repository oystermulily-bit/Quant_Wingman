"""Training adapter that exposes a frozen CSI 300 snapshot through the Wingman data surface.

Missing PIT values remain NaN on every public tensor view. The adapter is not
wired into the Engine yet; integration must explicitly consume
``feature_valid_mask``. Holdout prices stay sealed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_pipeline.hs300.config import RESEARCH_START, SEAL_CONFIRM_PHRASE
from data_pipeline.hs300.seal import load_sealed_holdout_prices
from data_pipeline.hs300_panel import HS300PanelDataManager
from model_core.vocab import FEATURE_NAMES, VOCAB_VERSION


class HS300WingmanAdapter:
    """Read-only adapter. Not wired into W1ngmanEngine or RD-Agent."""

    def __init__(
        self,
        snapshot_dir: str | Path,
        *,
        expected_members: int = 300,
        required_start: str | None = None,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.manager = HS300PanelDataManager(
            self.snapshot_dir,
            expected_members=expected_members,
            required_start=required_start or RESEARCH_START.isoformat(),
        )
        self._features: np.ndarray | None = None
        self._feature_valid: np.ndarray | None = None
        self._vocab_version: str | None = None
        self._symbols: list[str] = []
        self._dates: pd.DatetimeIndex | None = None

    def load(self) -> "HS300WingmanAdapter":
        self.manager.load(raise_on_gate_failure=True)
        feature_path = self.snapshot_dir / "panel" / "hs300_features.npz"
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing Wingman feature tensor: {feature_path}")
        payload = np.load(feature_path, allow_pickle=True)
        self._features = payload["features"]
        self._feature_valid = payload["feature_valid"].astype(bool)
        names = tuple(payload["feature_names"].tolist())
        if names != tuple(FEATURE_NAMES):
            raise ValueError("feature name order does not match the live Wingman vocab")
        self._vocab_version = str(payload["vocab_version"])
        if self._vocab_version != VOCAB_VERSION:
            raise ValueError(
                f"vocab_version {self._vocab_version!r} != live {VOCAB_VERSION!r}"
            )
        tensors = self.manager.to_tensors()
        self._symbols = [str(code) for code in tensors.symbols]
        self._dates = pd.DatetimeIndex(tensors.dates)
        self._raw = {
            "open": torch.from_numpy(tensors.causal_prices[:, 0, :].astype(np.float32)),
            "high": torch.from_numpy(tensors.causal_prices[:, 1, :].astype(np.float32)),
            "low": torch.from_numpy(tensors.causal_prices[:, 2, :].astype(np.float32)),
            "close": torch.from_numpy(tensors.causal_prices[:, 3, :].astype(np.float32)),
            "volume": torch.from_numpy(tensors.raw_ohlcv[:, 4, :].astype(np.float32)),
        }
        self._quote_valid = torch.from_numpy(tensors.quote_valid_mask)
        self._membership = torch.from_numpy(tensors.membership_mask)
        return self

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def dates(self) -> pd.DatetimeIndex:
        if self._dates is None:
            raise RuntimeError("Call load() first")
        return self._dates

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        if not hasattr(self, "_raw"):
            raise RuntimeError("Call load() first")
        return self._raw

    @property
    def feat_tensor_pit(self) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("Call load() first")
        return torch.from_numpy(self._features)

    @property
    def feature_valid_mask(self) -> torch.Tensor:
        if self._feature_valid is None:
            raise RuntimeError("Call load() first")
        return torch.from_numpy(self._feature_valid)

    @property
    def feat_tensor(self) -> torch.Tensor:
        """PIT feature tensor; suspended and unavailable observations stay NaN."""
        return self.feat_tensor_pit

    @property
    def membership_mask(self) -> torch.Tensor:
        return self._membership

    @property
    def quote_valid_mask(self) -> torch.Tensor:
        return self._quote_valid

    @property
    def vocab_version(self) -> str:
        if self._vocab_version is None:
            raise RuntimeError("Call load() first")
        return self._vocab_version

    @property
    def target_ret(self) -> torch.Tensor:
        labels = self.manager.load_labels(development_only=True)
        one = labels[
            (labels["horizon"] == 1) & (labels["label_type"] == "ABSOLUTE")
        ].copy()
        one["signal_date"] = pd.to_datetime(one["signal_date"]).dt.normalize()
        one["code"] = one["code"].astype(str).str.upper()
        dates = self.dates
        symbols = self.symbols
        date_pos = {pd.Timestamp(date): idx for idx, date in enumerate(dates)}
        symbol_pos = {code: idx for idx, code in enumerate(symbols)}
        target = np.full((len(symbols), len(dates)), np.nan, dtype=np.float32)
        si = one["code"].map(symbol_pos)
        di = one["signal_date"].map(date_pos)
        keep = si.notna() & di.notna()
        target[si[keep].astype(int), di[keep].astype(int)] = pd.to_numeric(
            one.loc[keep, "value"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        return torch.from_numpy(target)

    @property
    def bar_time(self) -> torch.Tensor:
        ns = self.dates.asi8
        return torch.from_numpy(ns.astype(np.int64))

    def load_holdout_prices(self, *, confirm: str) -> dict[str, pd.DataFrame]:
        if confirm != SEAL_CONFIRM_PHRASE:
            raise PermissionError("Holdout prices are sealed")
        return load_sealed_holdout_prices(self.snapshot_dir, confirm=confirm)
