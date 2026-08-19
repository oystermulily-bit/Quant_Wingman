"""Load training data from a single Parquet K-line file."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.data_manager import MT5DataManager
from model_core.features import MT5FeatureEngineer

# Canonical labels used across the project
_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")

_ADJUSTMENT_SUFFIXES = {
    "qfq", "hfq", "bfq", "forward", "backward", "adjusted", "adj",
}

_MARKET_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("time", "datetime", "date", "trade_date", "日期", "时间", "交易日期"),
    "open": ("open", "open_price", "开盘", "开盘价"),
    "high": ("high", "high_price", "最高", "最高价"),
    "low": ("low", "low_price", "最低", "最低价"),
    "close": ("close", "close_price", "收盘", "收盘价"),
    "volume": (
        "volume", "tick_volume", "vol", "成交量", "成交量_股", "成交量(股)",
    ),
    "symbol": ("symbol", "code", "stock_code", "ts_code", "股票代码", "证券代码", "代码"),
}

# Filename suffix aliases → canonical (case-insensitive keys)
_TF_ALIASES: dict[str, str] = {
    # M1
    "m1": "M1",
    "1m": "M1",
    "1min": "M1",
    "min1": "M1",
    # M5
    "m5": "M5",
    "5m": "M5",
    "5min": "M5",
    "min5": "M5",
    # M15
    "m15": "M15",
    "15m": "M15",
    "15min": "M15",
    "min15": "M15",
    # M30
    "m30": "M30",
    "30m": "M30",
    "30min": "M30",
    "min30": "M30",
    # H1
    "h1": "H1",
    "1h": "H1",
    "60m": "H1",
    "60min": "H1",
    "min60": "H1",
    "60": "H1",
    # H4
    "h4": "H4",
    "4h": "H4",
    "240m": "H4",
    "240min": "H4",
    "min240": "H4",
    "240": "H4",
    # D1
    "d1": "D1",
    "1d": "D1",
    "day": "D1",
    "daily": "D1",
    "1440m": "D1",
    "1440min": "D1",
    # W1
    "w1": "W1",
    "1w": "W1",
    "week": "W1",
    "weekly": "W1",
    # MN1 (month) — avoid bare "1m" which already maps to M1
    "mn1": "MN1",
    "1mo": "MN1",
    "1mon": "MN1",
    "month": "MN1",
    "monthly": "MN1",
}


def normalize_timeframe_token(token: str) -> str | None:
    """Map a filename timeframe token to canonical M1/M5/.../MN1."""
    raw = (token or "").strip()
    if not raw:
        return None
    key = raw.lower().replace("-", "").replace("_", "")
    if key in _TF_ALIASES:
        return _TF_ALIASES[key]
    upper = raw.upper()
    if upper in _TIMEFRAMES:
        return upper
    return None


def normalize_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with common English/Chinese market columns canonicalised."""
    out = df.copy()
    lookup = {str(column).strip().lower(): column for column in out.columns}
    rename: dict[Any, str] = {}
    for canonical, aliases in _MARKET_COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            source = lookup.get(alias.lower())
            if source is not None and source not in rename:
                rename[source] = canonical
                break
    return out.rename(columns=rename)


def infer_symbol_from_dataframe(df: pd.DataFrame, fallback: str) -> str:
    """Use a unique in-file symbol when present, otherwise keep the filename symbol."""
    normalised = normalize_market_dataframe(df)
    if "symbol" not in normalised.columns:
        return fallback
    values = normalised["symbol"].dropna().astype(str).str.strip()
    values = values[values != ""]
    unique = values.str.upper().unique().tolist()
    return unique[0] if len(unique) == 1 else fallback


def market_time_to_unix_seconds(values: pd.Series) -> pd.Series:
    """Convert datetime, strings, Unix seconds/ms/ns and legacy seconds/1000 to seconds."""
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    elif pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            return pd.Series(np.nan, index=values.index, dtype=float)
        max_abs = float(finite.abs().max())
        if max_abs < 10_000_000:
            numeric = numeric * 1000.0
            max_abs *= 1000.0
        if max_abs > 10_000_000_000_000_000:
            numeric = numeric / 1_000_000_000.0
        elif max_abs > 10_000_000_000:
            numeric = numeric / 1000.0
        return numeric
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")

    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = parsed.notna()
    if valid.any():
        # Explicit second resolution avoids assuming that pandas stores every
        # datetime column in nanoseconds (Parquet often preserves datetime64[ms]).
        naive_utc = parsed.loc[valid].dt.tz_convert(None)
        result.loc[valid] = naive_utc.to_numpy(dtype="datetime64[s]").astype("int64")
    return result


def parse_parquet_filename(path: str | Path) -> tuple[str, str]:
    """Parse ``{symbol}_{timeframe}.parquet``.

    Accepts canonical suffixes (``H1``) and common aliases (``60min``, ``1h``, ``5m``…).
    Examples: ``AAPL_H1.parquet``, ``002008_60min.parquet``, ``BTCUSDT_1h.parquet``.
    """
    name = Path(path).name
    if Path(path).suffix.lower() != ".parquet":
        raise ValueError(f"请选择 .parquet 文件；当前: {name}")
    stem = Path(path).stem
    if "_" not in stem:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet / 002008_60min.parquet；"
            f"当前: {name}"
        )
    symbol, tf_raw = stem.rsplit("_", 1)
    symbol_parts = [part for part in symbol.strip().split("_") if part]
    while len(symbol_parts) > 1 and symbol_parts[-1].lower() in _ADJUSTMENT_SUFFIXES:
        symbol_parts.pop()
    symbol = "_".join(symbol_parts)
    timeframe = normalize_timeframe_token(tf_raw)
    if not symbol or timeframe is None:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet / 002008_60min.parquet；"
            f"支持周期别名: H1/60min/1h, M5/5min, D1/1d …；当前: {name}"
        )
    return symbol, timeframe


def inspect_parquet_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() != ".parquet":
        raise ValueError("请选择 .parquet 文件")

    symbol, timeframe = parse_parquet_filename(p)
    original_df = pd.read_parquet(p)
    symbol = infer_symbol_from_dataframe(original_df, symbol)
    df = normalize_market_dataframe(original_df)
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Parquet 缺少行情列: {missing}")
    bars = len(df)
    if bars < Config.MIN_BARS:
        raise ValueError(
            f"数据不足: {bars} bars（至少需要 {Config.MIN_BARS}）"
        )

    # 从实际时间跨度计算年数（适用于所有周期，比固定公式更准确）
    years = None
    if len(df) > 1:
        try:
            unix_time = market_time_to_unix_seconds(df["time"])
            t_min = float(unix_time.min())
            t_max = float(unix_time.max())
            if t_max > t_min and t_max > 1_000_000_000:  # 合法的 Unix 时间戳
                span_seconds = t_max - t_min
                years = round(span_seconds / (365.25 * 24 * 3600), 2)
        except Exception:
            pass
    # 回退：H1 用固定公式（6240 根/年，24h 市场）
    if years is None and timeframe == "H1":
        years = round(bars / 6240, 2)
    return {
        "data_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "years_h1": years,
        "valid": True,
        "message": "",
    }


class ParquetDataManager:
    """Single-symbol data manager backed by one Parquet file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.symbol, self.timeframe = parse_parquet_filename(self.file_path)
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None

    def load(self) -> None:
        original_df = pd.read_parquet(self.file_path)
        self.symbol = infer_symbol_from_dataframe(original_df, self.symbol)
        df = normalize_market_dataframe(original_df)
        required = ["time", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet 缺少行情列: {missing}")

        sub = df[required].copy()
        sub["time"] = market_time_to_unix_seconds(sub["time"])
        for column in ("open", "high", "low", "close", "volume"):
            sub[column] = pd.to_numeric(sub[column], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        sub = sub[(sub["open"] > 0) & (sub["high"] > 0) & (sub["low"] > 0) & (sub["close"] > 0)]
        if len(sub) < Config.MIN_BARS:
            raise ValueError(
                f"清洗后数据不足: {len(sub)} bars（至少需要 {Config.MIN_BARS}）"
            )

        sub = sub.sort_values("time")
        sub = sub[~sub["time"].duplicated(keep="last")]

        rows = {field: sub[field].values for field in ["open", "high", "low", "close", "volume"]}
        raw: dict[str, torch.Tensor] = {
            field: torch.tensor(np.array([rows[field]]), dtype=torch.float32)
            for field in ["open", "high", "low", "close", "volume"]
        }
        raw["time"] = torch.tensor(
            np.array([sub["time"].to_numpy(dtype="int64")]),
            dtype=torch.int64,
        )

        self._raw_dict = raw
        self._target_ret = MT5DataManager._compute_target_ret(raw["open"])
        logger.info(
            f"[数据] 已加载 {self.symbol} {self.timeframe}，"
            f"共 {raw['open'].shape[1]} 根K线，文件 {self.file_path.name}"
        )

    @property
    def symbols(self) -> list[str]:
        return [self.symbol]

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        if self._raw_dict is None:
            raise RuntimeError("Call load() first")
        return self._raw_dict

    @property
    def feat_tensor(self) -> torch.Tensor:
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        if self._target_ret is None:
            raise RuntimeError("Call load() first")
        return self._target_ret

    @property
    def bar_time(self) -> torch.Tensor:
        raw = self.raw_dict
        if "time" in raw:
            return raw["time"][:, -1].long()
        return torch.zeros(1, dtype=torch.int64)
