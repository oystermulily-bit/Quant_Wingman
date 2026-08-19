# -*- coding: utf-8 -*-
"""深度检查 D:\国内期货K线数据 下全部 parquet 文件的数据质量。"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timezone, timedelta

BJ_TZ = timezone(timedelta(hours=8))
DATA_DIR = Path(r"D:\国内期货K线数据")

# 全部 parquet 文件
files = sorted(DATA_DIR.glob("*.parquet"))
print(f"找到 {len(files)} 个 parquet 文件")
print("=" * 100)

# 汇总统计
total_files = 0
ok_files = 0
problem_files = 0
problems = []  # (filename, issue_type, detail)

# 按品种分组统计
symbol_stats = {}  # {symbol: {tf: {rows, first, last, bad_h, bad_l, neg, zero_vol, huge_ret}}}

for f in files:
    total_files += 1
    name = f.stem  # e.g. 橡胶主连_D1
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        symbol, tf = parts
    else:
        symbol, tf = name, "?"

    try:
        df = pd.read_parquet(f)
    except Exception as e:
        problems.append((name, "READ_ERROR", str(e)))
        problem_files += 1
        continue

    rows = len(df)
    if rows == 0:
        problems.append((name, "EMPTY", "0 rows"))
        problem_files += 1
        continue

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["tick_volume"].values

    # OHLC 逻辑检查
    bad_h = int(np.sum((h < o) | (h < c)))
    bad_l = int(np.sum((l > o) | (l > c)))
    neg_price = int(np.sum(c < 0))
    zero_vol = int(np.sum(v == 0))

    # 异常涨跌 (>50%)
    if rows > 1:
        rets = np.abs(np.diff(c) / c[:-1])
        huge_ret = int(np.sum(rets > 0.5))
    else:
        huge_ret = 0

    # 时间范围
    first_ts = int(df["time"].iloc[0])
    last_ts = int(df["time"].iloc[-1])
    first_dt = pd.to_datetime(first_ts, unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
    last_dt = pd.to_datetime(last_ts, unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")

    # 记录统计
    if symbol not in symbol_stats:
        symbol_stats[symbol] = {}
    symbol_stats[symbol][tf] = {
        "rows": rows,
        "first": first_dt,
        "last": last_dt,
        "bad_h": bad_h,
        "bad_l": bad_l,
        "neg": neg_price,
        "zero_vol": zero_vol,
        "huge_ret": huge_ret,
    }

    has_problem = bad_h > 0 or bad_l > 0 or neg_price > 0 or zero_vol > 0 or huge_ret > 0
    if has_problem:
        problem_files += 1
        if bad_h > 0:
            problems.append((name, "OHLC_BAD_H", f"{bad_h} rows"))
        if bad_l > 0:
            problems.append((name, "OHLC_BAD_L", f"{bad_l} rows"))
        if neg_price > 0:
            problems.append((name, "NEG_PRICE", f"{neg_price} rows"))
        if zero_vol > 0:
            problems.append((name, "ZERO_VOL", f"{zero_vol} rows ({zero_vol/rows*100:.1f}%)"))
        if huge_ret > 0:
            problems.append((name, "HUGE_RET", f"{huge_ret} rows (>50%)"))
    else:
        ok_files += 1

# ── 输出结果 ──
print(f"\n总计: {total_files} 文件, OK: {ok_files}, 有问题: {problem_files}")
print("=" * 100)

if problems:
    print("\n=== 问题清单 ===")
    for name, issue, detail in problems:
        print(f"  {name:30s}  {issue:15s}  {detail}")
else:
    print("\n=== 全部文件通过质量检查，无任何问题！ ===")

# ── 按品种汇总 ──
print("\n" + "=" * 100)
print("按品种汇总 (只显示有问题的品种)")
print("=" * 100)
problem_symbols = set()
for sym, tfs in symbol_stats.items():
    for tf, s in tfs.items():
        if s["bad_h"] or s["bad_l"] or s["neg"] or s["zero_vol"] or s["huge_ret"]:
            problem_symbols.add(sym)

if problem_symbols:
    for sym in sorted(problem_symbols):
        print(f"\n{sym}:")
        for tf in ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1"]:
            if tf in symbol_stats[sym]:
                s = symbol_stats[sym][tf]
                issues = []
                if s["bad_h"]: issues.append(f"bad_h={s['bad_h']}")
                if s["bad_l"]: issues.append(f"bad_l={s['bad_l']}")
                if s["neg"]: issues.append(f"neg={s['neg']}")
                if s["zero_vol"]: issues.append(f"zero_vol={s['zero_vol']}")
                if s["huge_ret"]: issues.append(f"huge_ret={s['huge_ret']}")
                if issues:
                    print(f"  {tf}: rows={s['rows']}, {', '.join(issues)}")
else:
    print("无问题品种！")

# ── 全部品种周期一览表 ──
print("\n" + "=" * 100)
print("全部品种 × 周期 一览表")
print("=" * 100)
header = f"{'品种':<14s} |"
for tf in ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1"]:
    header += f" {tf:>6s} |"
print(header)
print("-" * len(header))

for sym in sorted(symbol_stats.keys()):
    row = f"{sym:<14s} |"
    for tf in ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1"]:
        if tf in symbol_stats[sym]:
            s = symbol_stats[sym][tf]
            if s["bad_h"] or s["bad_l"] or s["neg"] or s["zero_vol"] or s["huge_ret"]:
                row += f" {'BAD':>6s} |"
            else:
                row += f" {s['rows']:>6d} |"
        else:
            row += f" {'---':>6s} |"
    print(row)

print(f"\n总计: {total_files} 文件, OK: {ok_files}, 有问题: {problem_files}")
