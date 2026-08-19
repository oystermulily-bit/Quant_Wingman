# -*- coding: utf-8 -*-
"""深入排查校验发现的问题。

1. 旧数据残留: 已保存文件比重新拉取多出的 bar
2. D1 时间间隔异常: 看看具体是哪些日期，是否都是节假日
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timezone, timedelta

BJ_TZ = timezone(timedelta(hours=8))
DATA_DIR = Path(r"D:\国内期货K线数据")

# ── 1. 旧数据残留排查 ──
print("=" * 100)
print("1. 旧数据残留排查: 检查全部 600 个文件是否有重复或异常时间戳")
print("=" * 100)

# 之前校验发现沪铜M15多出 2025-03-18 00:00/00:15/00:30
# 这些是夜盘时间段，但 tqsdk 新拉取没有 — 可能是旧数据残留
# 检查所有 M15 文件的夜盘异常时间

night_issues = []
for f in sorted(DATA_DIR.glob("*_M15.parquet")):
    name = f.stem
    df = pd.read_parquet(f)
    # 提取每个 bar 的小时分钟
    dt_list = [pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(BJ_TZ) for ts in df["time"].values]
    hours = np.array([dt.hour for dt in dt_list])
    # 夜盘时间应该是 21:00-23:59 + 09:00-15:15
    # 异常: 00:00-09:00 之间的 M15 bar（除了 09:00 本身）
    # 或者 15:30-21:00 之间的 bar
    abnormal = ((hours > 0) & (hours < 9)) | ((hours >= 15) & (hours < 21) & (hours != 0))
    abnormal_count = abnormal.sum()
    if abnormal_count > 0:
        abnormal_times = [dt for dt, ab in zip(dt_list, abnormal) if ab]
        abnormal_strs = [dt.strftime("%Y-%m-%d %H:%M") for dt in abnormal_times[:5]]
        night_issues.append((name, abnormal_count, abnormal_strs))
        print(f"  {name}: {abnormal_count} 条异常时间 bar, 例: {abnormal_strs}")

if not night_issues:
    print("  无异常时间 bar")

# ── 2. D1 时间间隔异常排查 ──
print("\n" + "=" * 100)
print("2. D1 时间间隔异常排查: 橡胶主连 D1 的异常间隔")
print("=" * 100)

df = pd.read_parquet(DATA_DIR / "橡胶主连_D1.parquet")
times = df["time"].values
diffs = np.diff(times)

# 统计各种间隔
from collections import Counter
diff_days = (diffs / 86400).astype(int)
diff_counter = Counter(diff_days)
print("间隔天数分布:")
for d, cnt in sorted(diff_counter.items()):
    label = ""
    if d == 1: label = " (正常: 次日)"
    elif d == 2: label = " (周末: 周五→周一)"
    elif d == 3: label = " (长假3天)"
    elif d == 4: label = " (长假4天)"
    elif d == 5: label = " (长假5天)"
    elif d == 7: label = " (长假7天)"
    elif d > 7: label = " (超长间隔!)"
    print(f"  {d} 天: {cnt} 次{label}")

# 看看异常的具体是哪些日期
print("\n异常间隔的具体日期 (>3天):")
for i, d in enumerate(diff_days):
    if d > 3:
        from_dt = pd.to_datetime(int(times[i]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
        to_dt = pd.to_datetime(int(times[i+1]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
        print(f"  {from_dt} → {to_dt}  (间隔 {d} 天)")

# ── 3. 检查残留 bar 的具体情况 ──
print("\n" + "=" * 100)
print("3. 残留 bar 具体情况: 沪铜主连 M15 的多出 bar")
print("=" * 100)

# 读沪铜 M15
df = pd.read_parquet(DATA_DIR / "沪铜主连_M15.parquet")
# 看 2025-03-18 附近的数据
mask = (df["time"] >= pd.Timestamp("2025-03-17", tz="UTC").timestamp()) & \
       (df["time"] <= pd.Timestamp("2025-03-19", tz="UTC").timestamp())
sub = df[mask]
print("沪铜主连 M15 在 2025-03-17 ~ 2025-03-19 附近的数据:")
for _, r in sub.iterrows():
    dt = pd.to_datetime(int(r["time"]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d %H:%M")
    print(f"  {dt}  O={r['open']:.0f} H={r['high']:.0f} L={r['low']:.0f} C={r['close']:.0f} V={int(r['tick_volume'])}")

# ── 4. 涨跌>30% 排查 ──
print("\n" + "=" * 100)
print("4. 涨跌>30% 排查: 欧线集运 D1")
print("=" * 100)

df = pd.read_parquet(DATA_DIR / "欧线集运主连_D1.parquet")
rets = df["close"].pct_change().abs()
huge = df[rets > 0.3]
print(f"欧线集运 D1 涨跌>30%: {len(huge)} 条")
for _, r in huge.iterrows():
    idx = df.index.get_loc(r.name)
    if idx > 0:
        prev = df.iloc[idx - 1]
        dt = pd.to_datetime(int(r["time"]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
        prev_dt = pd.to_datetime(int(prev["time"]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
        ret = (r["close"] / prev["close"] - 1) * 100
        print(f"  {prev_dt} C={prev['close']:.0f} → {dt} C={r['close']:.0f}  ({ret:+.1f}%)")

# ── 5. 燃油主连 D1 异常间隔最多(57条) ──
print("\n" + "=" * 100)
print("5. 燃油主连 D1 异常间隔 (57条)")
print("=" * 100)

df = pd.read_parquet(DATA_DIR / "燃油主连_D1.parquet")
times = df["time"].values
diffs = np.diff(times)
diff_days = (diffs / 86400).astype(int)
diff_counter = Counter(diff_days)
print("间隔天数分布:")
for d, cnt in sorted(diff_counter.items()):
    print(f"  {d} 天: {cnt} 次")

print("\n燃油 D1 异常间隔 (>3天) 的具体日期:")
for i, d in enumerate(diff_days):
    if d > 3:
        from_dt = pd.to_datetime(int(times[i]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
        to_dt = pd.to_datetime(int(times[i+1]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
        print(f"  {from_dt} → {to_dt}  (间隔 {d} 天)")
