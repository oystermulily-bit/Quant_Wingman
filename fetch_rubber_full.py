# -*- coding: utf-8 -*-
"""拉取橡胶主连全部周期数据，data_length=10000（tqsdk 上限）。

覆盖周期: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w
保存路径: D:\\国内期货K线数据\\橡胶主连_{TF}.parquet
"""
import json
import time
from pathlib import Path

import pandas as pd
from datetime import timezone, timedelta
from tqsdk import TqApi, TqAuth, TqSim

# ── 配置 ──
settings = json.loads(Path(r"D:\cl\quant_w1ngman\web_settings.json").read_text(encoding="utf-8"))
user = settings.get("tqsdk_user", "")
pwd = settings.get("tqsdk_password", "")
if not user or not pwd:
    user, pwd = "", ""

TQSDK_SYMBOL = "KQ.m@SHFE.ru"  # 橡胶主连
DATA_LENGTH = 10000  # tqsdk 上限
BJ_TZ = timezone(timedelta(hours=8))
OUT_DIR = Path(r"D:\国内期货K线数据")

# 全部支持的周期: (duration_seconds, 文件名后缀, 显示名)
TIMEFRAMES = [
    (60,      "M1",  "1分钟"),
    (180,     "M3",  "3分钟"),
    (300,     "M5",  "5分钟"),
    (900,     "M15", "15分钟"),
    (1800,    "M30", "30分钟"),
    (3600,    "H1",  "1小时"),
    (7200,    "H2",  "2小时"),
    (14400,   "H4",  "4小时"),
    (86400,   "D1",  "日线"),
    (604800,  "W1",  "周线"),
]

print(f"tqsdk 账号: {user}")
print(f"品种: 橡胶主连 ({TQSDK_SYMBOL})")
print(f"数据长度上限: {DATA_LENGTH}")
print("=" * 80)

for dur, tf_name, tf_label in TIMEFRAMES:
    print(f"\n正在拉取 橡胶主连 {tf_label} ({tf_name})...")
    api = None
    df = None
    try:
        api = TqApi(TqSim(), auth=TqAuth(user, pwd), disable_print=True)
        df = api.get_kline_serial(TQSDK_SYMBOL, dur, data_length=DATA_LENGTH, adj_type="F")
        print(f"  拉取到 {len(df)} 根 K 线")
    except Exception as exc:
        print(f"  [ERROR] {exc}")
    finally:
        if api is not None:
            try: api.close()
            except Exception: pass

    if df is None or df.empty:
        print(f"  [SKIP] 无数据")
        continue

    # 过滤无效行
    before = len(df)
    df = df[df["datetime"] != 0].copy()
    df = df[df["volume"] > 0].copy()
    after = len(df)
    print(f"  过滤: {before} -> {after} 根 (去除 {before - after} 根无效)")

    if after == 0:
        print(f"  [SKIP] 过滤后无数据")
        continue

    # 转换格式
    result = pd.DataFrame()
    result["time"] = (df["datetime"].astype("int64") // 1_000_000_000).astype("int64")
    result["open"] = df["open"].astype("float64")
    result["high"] = df["high"].astype("float64")
    result["low"] = df["low"].astype("float64")
    result["close"] = df["close"].astype("float64")
    result["tick_volume"] = df["volume"].astype("int64")

    result = result.sort_values("time").reset_index(drop=True)
    result = result.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    out_file = OUT_DIR / f"橡胶主连_{tf_name}.parquet"
    result.to_parquet(out_file, index=False)

    # 统计
    first_dt = pd.to_datetime(int(result["time"].iloc[0]), unit="s", utc=True).tz_convert(BJ_TZ)
    last_dt = pd.to_datetime(int(result["time"].iloc[-1]), unit="s", utc=True).tz_convert(BJ_TZ)
    span_days = (int(result["time"].iloc[-1]) - int(result["time"].iloc[0])) / 86400

    # 质量检查
    o, h, l, c = result["open"], result["high"], result["low"], result["close"]
    bad_h = int(((h < o) | (h < c)).sum())
    bad_l = int(((l > o) | (l > c)).sum())
    neg = int((c < 0).sum())
    zero_v = int((result["tick_volume"] == 0).sum())

    print(f"  保存: {out_file}")
    print(f"  行数: {len(result)}")
    print(f"  时间: {first_dt.strftime('%Y-%m-%d %H:%M')} ~ {last_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  跨度: {span_days:.0f} 天 ({span_days/365.25:.1f} 年)")
    print(f"  收盘价: {result['close'].min():.1f} ~ {result['close'].max():.1f}")
    print(f"  质量: bad_h={bad_h} bad_l={bad_l} neg={neg} zero_vol={zero_v}")

    # 前后各 3 行
    print(f"  前3行:")
    for _, r in result.head(3).iterrows():
        dt = pd.to_datetime(int(r['time']), unit='s', utc=True).tz_convert(BJ_TZ).strftime('%Y-%m-%d %H:%M')
        print(f"    {dt}  O={r['open']:.0f} H={r['high']:.0f} L={r['low']:.0f} C={r['close']:.0f} V={int(r['tick_volume'])}")
    print(f"  后3行:")
    for _, r in result.tail(3).iterrows():
        dt = pd.to_datetime(int(r['time']), unit='s', utc=True).tz_convert(BJ_TZ).strftime('%Y-%m-%d %H:%M')
        print(f"    {dt}  O={r['open']:.0f} H={r['high']:.0f} L={r['low']:.0f} C={r['close']:.0f} V={int(r['tick_volume'])}")

    time.sleep(0.5)

print("\n" + "=" * 80)
print("全部完成！")
print("=" * 80)
