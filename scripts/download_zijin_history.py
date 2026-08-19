from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd


SYMBOL = "sh601899"
START_DATE = "20080101"
END_DATE = datetime.now().strftime("%Y%m%d")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "zijin_mining_601899"


def validate(df: pd.DataFrame, label: str) -> None:
    required = {"日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量_股", "成交额_元"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} 缺少字段: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{label} 没有返回数据")
    if df["日期"].duplicated().any():
        raise ValueError(f"{label} 存在重复交易日")
    if not df["日期"].is_monotonic_increasing:
        raise ValueError(f"{label} 日期未按升序排列")
    if (df[["开盘", "收盘", "最高", "最低"]].isna().any(axis=1)).any():
        raise ValueError(f"{label} 存在缺失的 OHLC 数据")
    if (df["最高"] < df[["开盘", "收盘", "最低"]].max(axis=1)).any():
        raise ValueError(f"{label} 存在最高价小于其他价格的记录")
    if (df["最低"] > df[["开盘", "收盘", "最高"]].min(axis=1)).any():
        raise ValueError(f"{label} 存在最低价高于其他价格的记录")
    if (df["成交量_股"] < 0).any() or (df["成交额_元"] < 0).any():
        raise ValueError(f"{label} 存在负成交量或负成交额")


def download(adjust: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            df = ak.stock_zh_a_daily(
                symbol=SYMBOL,
                start_date=START_DATE,
                end_date=END_DATE,
                adjust=adjust,
            )
            df = df.rename(
                columns={
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量_股",
                    "amount": "成交额_元",
                    "outstanding_share": "流通股本_股",
                    "turnover": "换手率",
                }
            )
            df.insert(1, "股票代码", "601899.SH")
            df["日期"] = pd.to_datetime(df["日期"])
            return df.sort_values("日期", kind="stable").reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            if attempt < 6:
                time.sleep(min(4 * attempt, 20))
    raise RuntimeError(f"下载 {adjust or 'raw'} 数据失败") from last_error


def save_dataset(df: pd.DataFrame, name: str) -> dict[str, object]:
    validate(df, name)
    target = OUTPUT_ROOT / name
    yearly = target / "yearly"
    yearly.mkdir(parents=True, exist_ok=True)

    csv_path = target / f"601899_{name}_daily.csv"
    parquet_path = target / f"601899_{name}_daily.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    df.to_parquet(parquet_path, index=False)

    years = sorted(df["日期"].dt.year.unique().tolist())
    for year, part in df.groupby(df["日期"].dt.year, sort=True):
        part.to_csv(
            yearly / f"601899_{int(year)}_{name}_daily.csv",
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )

    return {
        "adjustment": name,
        "rows": int(len(df)),
        "first_date": df["日期"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": df["日期"].iloc[-1].strftime("%Y-%m-%d"),
        "years": years,
        "csv": str(csv_path),
        "parquet": str(parquet_path),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw = download("")
    qfq = download("qfq")
    if not raw["日期"].equals(qfq["日期"]):
        raise ValueError("不复权与前复权交易日序列不一致")

    manifest = {
        "company": "紫金矿业集团股份有限公司",
        "security": "A股",
        "symbol": "601899.SH",
        "frequency": "日线",
        "source": "AkShare stock_zh_a_daily（新浪财经历史行情接口）",
        "source_documentation": "https://akshare.akfamily.xyz/data/stock/stock.html",
        "downloaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "datasets": [
            save_dataset(raw, "raw"),
            save_dataset(qfq, "qfq"),
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
