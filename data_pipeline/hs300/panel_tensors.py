"""Dense Wingman tensors from a membership-led CSI 300 panel.

Missing prices stay NaN. Non-members are present in N (historical union) but
have membership_mask=false and do not enter the cross-section. v2 stores the
65 Wingman features in panel/hs300_features.npz and in this object's features
array; pack_panel_tensors itself still emits F=0 until features are attached.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ("open_raw", "high_raw", "low_raw", "close_raw", "volume")
TR_COLUMNS = ("open_tr", "high_tr", "low_tr", "close_tr")


@dataclass(frozen=True)
class HS300PanelTensors:
    dates: np.ndarray
    symbols: np.ndarray
    raw_ohlcv: np.ndarray
    amount: np.ndarray
    causal_prices: np.ndarray
    membership_mask: np.ndarray
    quote_valid_mask: np.ndarray
    buyable_mask: np.ndarray
    sellable_mask: np.ndarray
    industry_id: np.ndarray
    features: np.ndarray
    industry_codes: np.ndarray

    @property
    def n_symbols(self) -> int:
        return int(self.symbols.shape[0])

    @property
    def n_dates(self) -> int:
        return int(self.dates.shape[0])

    def membership_counts(self) -> np.ndarray:
        return self.membership_mask.sum(axis=0)

    def assert_daily_membership(self, expected: int = 300) -> None:
        counts = self.membership_counts()
        bad = np.flatnonzero(counts != expected)
        if bad.size:
            raise AssertionError(
                f"{bad.size} days have membership_mask true-count != {expected}"
            )

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "dates": self.dates,
            "symbols": self.symbols,
            "raw_ohlcv": self.raw_ohlcv,
            "amount": self.amount,
            "causal_prices": self.causal_prices,
            "membership_mask": self.membership_mask,
            "quote_valid_mask": self.quote_valid_mask,
            "buyable_mask": self.buyable_mask,
            "sellable_mask": self.sellable_mask,
            "industry_id": self.industry_id,
            "features": self.features,
            "industry_codes": self.industry_codes,
        }


def pack_panel_tensors(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    expected_members: int | None = 300,
) -> HS300PanelTensors:
    required = {
        "date",
        "code",
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
        "volume",
        "amount",
        "open_tr",
        "close_tr",
        "has_quote",
        "can_buy_open",
        "can_sell_open",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing {missing}")

    dates = pd.DatetimeIndex(calendar).normalize().drop_duplicates().sort_values()
    symbols = (
        panel["code"].astype("string").str.strip().str.upper().drop_duplicates().sort_values()
    )
    date_pos = {pd.Timestamp(date): idx for idx, date in enumerate(dates)}
    symbol_pos = {str(code): idx for idx, code in enumerate(symbols)}
    n_dates = len(dates)
    n_symbols = len(symbols)

    rows = panel.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows["code"] = rows["code"].astype("string").str.strip().str.upper()
    rows = rows[rows["date"].isin(dates)]
    si = rows["code"].map(symbol_pos).to_numpy(dtype=np.int32)
    di = rows["date"].map(date_pos).to_numpy(dtype=np.int32)

    raw_ohlcv = np.full((n_symbols, 5, n_dates), np.nan, dtype=np.float32)
    amount = np.full((n_symbols, n_dates), np.nan, dtype=np.float32)
    causal_prices = np.full((n_symbols, 4, n_dates), np.nan, dtype=np.float32)
    membership_mask = np.zeros((n_symbols, n_dates), dtype=bool)
    quote_valid_mask = np.zeros((n_symbols, n_dates), dtype=bool)
    buyable_mask = np.zeros((n_symbols, n_dates), dtype=bool)
    sellable_mask = np.zeros((n_symbols, n_dates), dtype=bool)
    industry_id = np.full((n_symbols, n_dates), -1, dtype=np.int32)

    membership_mask[si, di] = True
    has_quote = rows["has_quote"].fillna(False).astype(bool).to_numpy()
    quote_valid_mask[si, di] = has_quote
    buyable_mask[si, di] = rows["can_buy_open"].fillna(False).astype(bool).to_numpy() & has_quote
    sellable_mask[si, di] = rows["can_sell_open"].fillna(False).astype(bool).to_numpy() & has_quote

    for idx, column in enumerate(OHLCV_COLUMNS):
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float32)
        values = np.where(has_quote, values, np.nan)
        raw_ohlcv[si, idx, di] = values
    amount[si, di] = np.where(
        has_quote,
        pd.to_numeric(rows["amount"], errors="coerce").to_numpy(dtype=np.float32),
        np.nan,
    )
    for idx, column in enumerate(TR_COLUMNS):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float32)
        values = np.where(has_quote, values, np.nan)
        causal_prices[si, idx, di] = values

    industry_codes = np.array([], dtype=object)
    if "industry_code" in rows.columns:
        mapped = rows["industry_code"].astype("string").str.strip().str.upper()
        unique_industries = mapped.dropna()
        unique_industries = unique_industries[unique_industries.ne("")]
        industry_codes = unique_industries.drop_duplicates().sort_values().to_numpy()
        industry_pos = {str(code): idx for idx, code in enumerate(industry_codes)}
        ids = mapped.map(industry_pos).fillna(-1).astype(np.int32).to_numpy()
        industry_id[si, di] = ids

    features = np.zeros((n_symbols, 0, n_dates), dtype=np.float32)
    packed = HS300PanelTensors(
        dates=dates.to_numpy(dtype="datetime64[D]"),
        symbols=symbols.to_numpy(dtype=object),
        raw_ohlcv=raw_ohlcv,
        amount=amount,
        causal_prices=causal_prices,
        membership_mask=membership_mask,
        quote_valid_mask=quote_valid_mask,
        buyable_mask=buyable_mask,
        sellable_mask=sellable_mask,
        industry_id=industry_id,
        features=features,
        industry_codes=industry_codes,
    )
    if expected_members is not None:
        packed.assert_daily_membership(expected_members)
    return packed


def save_panel_tensors(tensors: HS300PanelTensors, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = tensors.to_dict()
    np.savez_compressed(path, **payload)
    return {
        "path": str(path).replace("\\", "/"),
        "n_symbols": tensors.n_symbols,
        "n_dates": tensors.n_dates,
        "feature_count": int(tensors.features.shape[1]),
        "membership_true_per_day": int(tensors.membership_counts().min())
        if tensors.n_dates
        else 0,
        "dtype_prices": "float32",
        "missing": "NaN",
    }
