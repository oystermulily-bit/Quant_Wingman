# -*- coding: utf-8 -*-
"""检查橡胶主连 D1 OHLC 异常的细节，以及燃油主连零成交量问题。"""
import pandas as pd
import numpy as np
from datetime import timezone, timedelta

BJ_TZ = timezone(timedelta(hours=8))

def ts_to_str(ts):
    return pd.to_datetime(int(ts), unit='s', utc=True).tz_convert(BJ_TZ).strftime('%Y-%m-%d')

print("=" * 80)
print("橡胶主连 D1 — OHLC 异常详情")
print("=" * 80)
df = pd.read_parquet(r'D:\国内期货K线数据\橡胶主连_D1.parquet')
o, h, l, c = df['open'], df['high'], df['low'], df['close']

bad_h = df[h < np.maximum(o, c)]
bad_l = df[l > np.minimum(o, c)]

print(f"\nhigh < max(open, close): {len(bad_h)} 行")
for _, r in bad_h.iterrows():
    print(f"  {ts_to_str(r['time'])}  O={r['open']:.1f} H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f}  (H < max(O,C)={max(r['open'],r['close']):.1f})")

print(f"\nlow > min(open, close): {len(bad_l)} 行")
for _, r in bad_l.iterrows():
    print(f"  {ts_to_str(r['time'])}  O={r['open']:.1f} H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f}  (L > min(O,C)={min(r['open'],r['close']):.1f})")

# 检查是否是前复权导致的负价格区域
print(f"\n前5行（最早数据）:")
for _, r in df.head(5).iterrows():
    print(f"  {ts_to_str(r['time'])}  O={r['open']:.1f} H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f} V={r['tick_volume']}")

print(f"\n后5行（最新数据）:")
for _, r in df.tail(5).iterrows():
    print(f"  {ts_to_str(r['time'])}  O={r['open']:.1f} H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f} V={r['tick_volume']}")

# 检查价格是否有负值（前复权可能导致）
neg_price = df[df['close'] < 0]
print(f"\n负价格行数: {len(neg_price)}")
if len(neg_price) > 0:
    print(neg_price[['time','open','high','low','close']].head().to_string())

# 检查价格变化是否合理
print(f"\n收盘价统计: min={df['close'].min():.1f} max={df['close'].max():.1f} mean={df['close'].mean():.1f}")

# ── 燃油主连 D1 零成交量 ──
print("\n" + "=" * 80)
print("燃油主连 D1 — 零成交量问题")
print("=" * 80)
df2 = pd.read_parquet(r'D:\国内期货K线数据\燃油主连_D1.parquet')
zero_vol = df2[df2['tick_volume'] == 0]
print(f"总行数: {len(df2)}, 零成交量行数: {len(zero_vol)} ({len(zero_vol)/len(df2)*100:.1f}%)")
print(f"零成交量行的时间范围: {ts_to_str(zero_vol['time'].iloc[0])} ~ {ts_to_str(zero_vol['time'].iloc[-1])}")
# 看看零成交量是否集中在某个时间段
if len(zero_vol) > 0:
    zero_vol_dates = zero_vol['time'].apply(ts_to_str).values
    years = [d[:4] for d in zero_vol_dates]
    from collections import Counter
    year_counts = Counter(years)
    print(f"零成交量按年分布: {dict(sorted(year_counts.items()))}")

# ── 豆二主连 D1 异常最多 ──
print("\n" + "=" * 80)
print("豆二主连 D1 — OHLC 异常最多")
print("=" * 80)
df3 = pd.read_parquet(r'D:\国内期货K线数据\豆二主连_D1.parquet')
o3, h3, l3, c3 = df3['open'], df3['high'], df3['low'], df3['close']
bad_h3 = df3[h3 < np.maximum(o3, c3)]
bad_l3 = df3[l3 > np.minimum(o3, c3)]
print(f"high < max(open,close): {len(bad_h3)} 行")
print(f"low > min(open,close): {len(bad_l3)} 行")
# 看看异常集中在什么时间段
if len(bad_h3) > 0:
    print(f"异常 high 时间范围: {ts_to_str(bad_h3['time'].iloc[0])} ~ {ts_to_str(bad_h3['time'].iloc[-1])}")
if len(bad_l3) > 0:
    print(f"异常 low 时间范围: {ts_to_str(bad_l3['time'].iloc[0])} ~ {ts_to_str(bad_l3['time'].iloc[-1])}")

# ── 欧线集运 M15 异常涨跌 ──
print("\n" + "=" * 80)
print("欧线集运 M15 — 单根涨跌>50%")
print("=" * 80)
df4 = pd.read_parquet(r'D:\国内期货K线数据\欧线集运主连_M15.parquet')
ret4 = df4['close'].pct_change().abs()
huge4 = df4[ret4 > 0.5]
for _, r in huge4.iterrows():
    prev_close = df4.loc[df4.index.get_loc(r.name) - 1, 'close'] if df4.index.get_loc(r.name) > 0 else 0
    print(f"  {ts_to_str(r['time'])}  prev_close={prev_close:.1f} -> close={r['close']:.1f} (ret={abs((r['close']/prev_close-1)*100) if prev_close else 0:.1f}%)")

print("\n检查完成。")
