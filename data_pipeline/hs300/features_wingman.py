"""Wingman 65-feature tensors from causal CSI 300 prices.

Stored values keep NaN. Suspended/non-member rows never become numeric zero
observations inside rolling operators. Cross-section features use only that
day's quoted members and average ranks for ties.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import FEATURE_WARMUP_BARS
from .panel_tensors import HS300PanelTensors


def _feature_names() -> tuple[str, ...]:
    from model_core.vocab import FEATURE_NAMES

    return tuple(FEATURE_NAMES)


def _cs_average_rank(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    n_dates = values.shape[1]
    for t in range(n_dates):
        mask = valid[:, t] & np.isfinite(values[:, t])
        if mask.sum() < 2:
            continue
        col = pd_rank(values[mask, t])
        out[mask, t] = col
    return out


def pd_rank(values: np.ndarray) -> np.ndarray:
    import pandas as pd

    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def _cs_demean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    n_dates = values.shape[1]
    for t in range(n_dates):
        mask = valid[:, t] & np.isfinite(values[:, t])
        if not mask.any():
            continue
        col = values[mask, t]
        out[mask, t] = (col - col.mean()).astype(np.float32)
    return out


def _cs_zscore(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    n_dates = values.shape[1]
    for t in range(n_dates):
        mask = valid[:, t] & np.isfinite(values[:, t])
        if mask.sum() < 2:
            continue
        col = values[mask, t]
        std = col.std()
        if not np.isfinite(std) or std < 1e-12:
            continue
        out[mask, t] = ((col - col.mean()) / std).astype(np.float32)
    return out


def _log_return(close: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float32)
    prev = close[:, :-lag]
    cur = close[:, lag:]
    good = np.isfinite(prev) & np.isfinite(cur) & (prev > 0) & (cur > 0)
    out[:, lag:] = np.where(good, np.log(cur / prev), np.nan)
    return out


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous true values."""
    padded = np.concatenate(([False], mask.astype(bool, copy=False), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def compute_wingman_feature_tensors(
    tensors: HS300PanelTensors,
    *,
    warmup_bars: int = FEATURE_WARMUP_BARS,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    import torch

    from model_core.features import FEATURE_REGISTRY

    names = _feature_names()
    close = tensors.causal_prices[:, 3, :].astype(np.float32)
    high = tensors.causal_prices[:, 1, :].astype(np.float32)
    low = tensors.causal_prices[:, 2, :].astype(np.float32)
    open_tr = tensors.causal_prices[:, 0, :].astype(np.float32)
    volume = tensors.raw_ohlcv[:, 4, :].astype(np.float32)
    member = tensors.membership_mask
    quoted = tensors.quote_valid_mask & member
    for array in (close, high, low, open_tr, volume):
        array[~quoted] = np.nan
    usable = quoted.copy()
    for array in (close, high, low, open_tr, volume):
        usable &= np.isfinite(array)

    # Legacy feature operators intentionally normalise warm-up values to zero.
    # Feeding an entire panel with suspended bars replaced by zero would make
    # those synthetic zeros enter future rolling windows. Compute each symbol's
    # contiguous quoted segment independently instead. A gap resets history and
    # the configured warm-up, which is conservative and causal.
    features = np.full(
        (tensors.n_symbols, len(FEATURE_REGISTRY.feature_specs), tensors.n_dates),
        np.nan,
        dtype=np.float32,
    )
    continuous_ready = np.zeros_like(usable, dtype=bool)
    arrays = {
        "open": open_tr,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    for symbol_idx in range(tensors.n_symbols):
        for start, end in _true_runs(usable[symbol_idx]):
            ready_start = min(end, start + max(0, int(warmup_bars)))
            continuous_ready[symbol_idx, ready_start:end] = True
            if ready_start >= end:
                continue
            raw = {
                key: torch.from_numpy(value[symbol_idx : symbol_idx + 1, start:end])
                for key, value in arrays.items()
            }
            run_length = end - start
            computed = []
            for spec in FEATURE_REGISTRY.feature_specs:
                value = spec.compute(raw)
                # A few legacy fixed-lag features emit their full padding width
                # when the contiguous run is shorter than the lag. The causal
                # prefix is still well-defined (warm-up zeros) and is cropped to
                # the actual run. A shorter result is never a valid contract.
                if value.shape[-1] < run_length:
                    raise ValueError(
                        f"feature {spec.name} returned {value.shape[-1]} bars "
                        f"for a {run_length}-bar contiguous quote run"
                    )
                computed.append(value[..., :run_length])
            stacked = torch.stack(computed, dim=1)
            features[symbol_idx : symbol_idx + 1, :, start:end] = (
                stacked.detach().cpu().numpy().astype(np.float32, copy=False)
            )
    name_index = {name: idx for idx, name in enumerate(names)}
    ret5 = _log_return(close, 5)
    ret20 = _log_return(close, 20)
    if "CS_RANK_RET5" in name_index:
        features[:, name_index["CS_RANK_RET5"], :] = _cs_average_rank(ret5, quoted)
    if "CS_ZSCORE_RET20" in name_index:
        features[:, name_index["CS_ZSCORE_RET20"], :] = _cs_zscore(ret20, quoted)
    if "REL_RET5" in name_index:
        features[:, name_index["REL_RET5"], :] = _cs_demean(ret5, quoted)
    if "REL_RET20" in name_index:
        features[:, name_index["REL_RET20"], :] = _cs_demean(ret20, quoted)
    if "REL_VOL" in name_index and "RVOL" in name_index:
        features[:, name_index["REL_VOL"], :] = _cs_demean(
            features[:, name_index["RVOL"], :], quoted
        )

    valid = np.isfinite(features) & continuous_ready[:, None, :]
    features = np.where(valid, features, np.nan).astype(np.float32)
    return features, valid.astype(bool), names


def save_feature_tensors(
    path: Path,
    *,
    features: np.ndarray,
    valid: np.ndarray,
    names: tuple[str, ...],
    vocab_version: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        feature_valid=valid,
        feature_names=np.array(names, dtype=object),
        vocab_version=np.array(vocab_version),
    )
    return {
        "path": str(path).replace("\\", "/"),
        "shape": list(features.shape),
        "feature_count": int(features.shape[1]),
        "vocab_version": vocab_version,
        "missing": "NaN",
    }
