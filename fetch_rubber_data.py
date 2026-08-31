# -*- coding: utf-8 -*-
"""拉取橡胶主连各时间周期数据并保存到 D:\国内期货K线数据。

使用 tqsdk 天勤量化，前复权 adj_type="F"。
周期: M15, M30, H1, D1
"""
import sys
import time
from pathlib import Path

import pandas as pd
from datetime import timezone, timedelta
from web.settings import load_settings

settings = load_settings()
user = str(settings.get("tqsdk_user", "")).strip()
pwd = str(settings.get("tqsdk_password", "")).strip()
if not user or not pwd:
    raise RuntimeError("请先设置环境变量 TQSDK_USER 和 TQSDK_PASSWORD")

print("已从环境变量加载 tqsdk 凭据")
print("=" * 60)

# 橡胶主连 tqsdk 代码
TQSDK_SYMBOL = "KQ.m@SHFE.ru"

# 周期映射: tqsdk duration_seconds
TIMEFRAMES = {
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "D1": 86400,
}

# 数据长度（tqsdk 最多支持 8000+ 根）
DATA_LENGTH = 8000

BJ_TZ = timezone(timedelta(hours=8))

out_dir = Path(r"D:\国内期货K线数据")
out_dir.mkdir(parents=True, exist_ok=True)

from tqsdk import TqApi, TqAuth, TqSim

for tf_name, duration in TIMEFRAMES.items():
    print(f"\n正在拉取 橡胶主连 {tf_name}...")
    api = None
    try:
        api = TqApi(TqSim(), auth=TqAuth(user, pwd), disable_print=True)
        # adj_type="F" 前复权
        df = api.get_kline_serial(
            TQSDK_SYMBOL, duration, data_length=DATA_LENGTH, adj_type="F"
        )
        print(f"  拉取到 {len(df)} 根 K 线")
    except Exception as exc:
        print(f"  [ERROR] tqsdk 连接失败: {exc}")
        if api is not None:
            try:
                api.close()
            except Exception:
                pass
        continue
    finally:
        if api is not None:
            try:
                api.close()
            except Exception:
                pass

    if df is None or df.empty:
        print(f"  [ERROR] 未拉取到数据")
        continue

    # 过滤无效行
    df = df[df["datetime"] != 0].copy()
    df = df[df["volume"] > 0].copy()
    print(f"  过滤后有效 K 线: {len(df)} 根")

    if len(df) == 0:
        print(f"  [ERROR] 过滤后无有效数据")
        continue

    # 转换为项目兼容格式
    result = pd.DataFrame()
    result["time"] = (df["datetime"].astype("int64") // 1_000_000_000).astype("int64")
    result["open"] = df["open"].astype("float64")
    result["high"] = df["high"].astype("float64")
    result["low"] = df["low"].astype("float64")
    result["close"] = df["close"].astype("float64")
    result["tick_volume"] = df["volume"].astype("int64")

    # 升序排列 + 去重
    result = result.sort_values("time").reset_index(drop=True)
    result = result.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    # 保存
    out_file = out_dir / f"橡胶主连_{tf_name}.parquet"
    result.to_parquet(out_file, index=False)

    # 统计
    first_dt = pd.to_datetime(result["time"].iloc[0], unit="s", utc=True).tz_convert(BJ_TZ)
    last_dt = pd.to_datetime(result["time"].iloc[-1], unit="s", utc=True).tz_convert(BJ_TZ)
    span_days = (result["time"].iloc[-1] - result["time"].iloc[0]) / 86400

    print(f"  保存: {out_file}")
    print(f"  总行数: {len(result)}")
    print(f"  时间范围: {first_dt.strftime('%Y-%m-%d %H:%M')} ~ {last_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  跨度: {span_days:.0f} 天 ({span_days/365.25:.1f} 年)")
    print(f"  收盘价: {result['close'].min():.1f} ~ {result['close'].max():.1f}")
    print(f"  前3行:")
    for _, r in result.head(3).iterrows():
        dt = pd.to_datetime(int(r['time']), unit='s', utc=True).tz_convert(BJ_TZ).strftime('%Y-%m-%d %H:%M')
        print(f"    {dt}  O={r['open']:.1f} H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f} V={int(r['tick_volume'])}")
    print(f"  后3行:")
    for _, r in result.tail(3).iterrows():
        dt = pd.to_datetime(int(r['time']), unit='s', utc=True).tz_convert(BJ_TZ).strftime('%Y-%m-%d %H:%M')
        print(f"    {dt}  O={r['open']:.1f} H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f} V={int(r['tick_volume'])}")

    # 简单质量检查
    o, h, l, c = result["open"], result["high"], result["low"], result["close"]
    bad_h = ((h < o) | (h < c)).sum()
    bad_l = ((l > o) | (l > c)).sum()
    neg = (c < 0).sum()
    zero_vol = (result["tick_volume"] == 0).sum()
    print(f"  质量检查: bad_h={bad_h} bad_l={bad_l} neg_price={neg} zero_vol={zero_vol}")

    time.sleep(1)  # 避免频繁连接

print("\n" + "=" * 60)
print("全部拉取完成!")
print("=" * 60)
