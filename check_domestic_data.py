# -*- coding: utf-8 -*-
"""深度检查 D:\\国内期货K线数据 目录下所有 parquet 文件的数据质量。"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

DATA_DIR = Path(r"D:\国内期货K线数据")
BJ_TZ = timezone(timedelta(hours=8))

# 预期周期
TIMEFRAMES = ["M15", "M30", "H1", "D1"]
TF_SECONDS = {"M15": 900, "M30": 1800, "H1": 3600, "D1": 86400}

# 预期列名
EXPECTED_COLS = {"time", "open", "high", "low", "close", "tick_volume"}

problems = []
stats = []

files = sorted(DATA_DIR.glob("*.parquet"))
print(f"共找到 {len(files)} 个 parquet 文件\n")

for f in files:
    symbol = f.stem.rsplit("_", 1)[0]  # 如 "橡胶主连"
    tf = f.stem.rsplit("_", 1)[1]      # 如 "D1"
    issues = []

    try:
        df = pd.read_parquet(f)
    except Exception as e:
        problems.append({"file": f.name, "issues": [f"读取失败: {e}"]})
        continue

    # 1. 列名检查
    cols = set(df.columns)
    missing = EXPECTED_COLS - cols
    if missing:
        issues.append(f"缺少列: {missing}")

    # 2. 行数检查
    n = len(df)
    if n == 0:
        issues.append("空文件（0 行）")
        problems.append({"file": f.name, "issues": issues})
        continue
    if n < 100:
        issues.append(f"行数过少: {n} 行")

    # 3. 时间戳检查
    if "time" in df.columns:
        ts = df["time"].values

        # 检查是否有 0 或负值
        bad_ts = np.sum(ts <= 0)
        if bad_ts > 0:
            issues.append(f"无效时间戳（<=0）: {bad_ts} 行")

        # 检查是否升序
        if not np.all(np.diff(ts) >= 0):
            issues.append("时间戳未升序排列")

        # 检查重复时间戳
        dup_ts = np.sum(np.diff(ts) == 0)
        if dup_ts > 0:
            issues.append(f"重复时间戳: {dup_ts} 行")

        # 时间范围
        valid_ts = ts[ts > 0]
        if len(valid_ts) > 0:
            first_dt = pd.to_datetime(valid_ts[0], unit="s", utc=True).tz_convert(BJ_TZ)
            last_dt = pd.to_datetime(valid_ts[-1], unit="s", utc=True).tz_convert(BJ_TZ)
            span_days = (valid_ts[-1] - valid_ts[0]) / 86400

            # 检查时间间隔是否合理
            if len(valid_ts) > 1:
                diffs = np.diff(valid_ts)
                expected_gap = TF_SECONDS.get(tf, 0)

                # 统计间隔异常
                if expected_gap > 0:
                    # 允许 5% 浮动（夜盘/午盘间隔不同）
                    too_small = np.sum(diffs < expected_gap * 0.5)
                    # 跨夜/跨周/跨月的大间隔是正常的
                    # 但如果 D1 间隔 < 3600（1小时），肯定有问题
                    if tf == "D1":
                        too_small_d1 = np.sum(diffs < 3600)
                        if too_small_d1 > 0:
                            issues.append(f"D1 间隔异常小（<1h）: {too_small_d1} 处")
                else:
                    too_small = 0

        else:
            first_dt = last_dt = None
            span_days = 0
            issues.append("所有时间戳无效")

    # 4. OHLC 逻辑检查
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]

        # high >= low
        hl_bad = np.sum((h < l).values)
        if hl_bad > 0:
            issues.append(f"high < low: {hl_bad} 行")

        # high >= max(open, close)
        hc_bad = np.sum((h < np.maximum(o, c)).values)
        if hc_bad > 0:
            issues.append(f"high < max(open,close): {hc_bad} 行")

        # low <= min(open, close)
        lc_bad = np.sum((l > np.minimum(o, c)).values)
        if lc_bad > 0:
            issues.append(f"low > min(open,close): {lc_bad} 行")

    # 5. 价格合理性检查
    if "close" in df.columns:
        close = df["close"]
        # 负价格
        neg = np.sum((close < 0).values)
        if neg > 0:
            issues.append(f"负价格: {neg} 行")

        # 零价格
        zero = np.sum((close == 0).values)
        if zero > 0:
            issues.append(f"零价格: {zero} 行")

        # 异常大的价格变动（单根 bar 涨跌超过 50%）
        if len(close) > 1:
            ret = close.pct_change().abs()
            huge = np.sum((ret > 0.5).values)
            if huge > 0:
                issues.append(f"单根 bar 涨跌>50%: {huge} 处")

    # 6. 成交量检查
    if "tick_volume" in df.columns:
        vol = df["tick_volume"]
        neg_vol = np.sum((vol < 0).values)
        if neg_vol > 0:
            issues.append(f"负成交量: {neg_vol} 行")
        zero_vol = np.sum((vol == 0).values)
        if zero_vol > 0:
            # volume=0 可能是非交易时段残留，记录但不一定是严重问题
            pct = zero_vol / len(df) * 100
            if pct > 5:
                issues.append(f"零成交量占比 {pct:.1f}% ({zero_vol} 行)")

    # 7. 数据覆盖率检查 — 各周期应有足够历史
    if span_days > 0:
        if tf == "D1" and span_days < 365:
            issues.append(f"D1 历史不足1年: {span_days:.0f} 天")
        elif tf == "H1" and span_days < 180:
            issues.append(f"H1 历史不足半年: {span_days:.0f} 天")
        elif tf == "M15" and span_days < 90:
            issues.append(f"M15 历史不足3月: {span_days:.0f} 天")
        elif tf == "M30" and span_days < 90:
            issues.append(f"M30 历史不足3月: {span_days:.0f} 天")

    stat = {
        "file": f.name,
        "symbol": symbol,
        "tf": tf,
        "rows": n,
        "first": str(first_dt) if first_dt else "N/A",
        "last": str(last_dt) if last_dt else "N/A",
        "span_days": round(span_days, 1) if span_days else 0,
        "issues": issues if issues else "OK",
    }
    stats.append(stat)
    if issues:
        problems.append({"file": f.name, "issues": issues})

# ── 汇总输出 ──
print("=" * 80)
print("数据质量深度检查报告")
print("=" * 80)
print(f"检查时间: {datetime.now(BJ_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总文件数: {len(files)}")
print(f"有问题文件: {len(problems)}")
print(f"正常文件: {len(files) - len(problems)}")
print()

if problems:
    print("-" * 80)
    print("问题详情:")
    print("-" * 80)
    for p in problems:
        print(f"\n  [{p['file']}]")
        for iss in p["issues"]:
            print(f"    ❌ {iss}")
else:
    print("✅ 所有文件检查通过，无异常！")

# 按品种汇总
print("\n" + "=" * 80)
print("按品种汇总:")
print("=" * 80)
symbols = sorted(set(s["symbol"] for s in stats))
print(f"{'品种':<12} {'周期':<5} {'行数':>6} {'起始时间':<22} {'结束时间':<22} {'跨度(天)':>8} {'状态':<8}")
print("-" * 90)
for sym in symbols:
    for tf in TIMEFRAMES:
        matches = [s for s in stats if s["symbol"] == sym and s["tf"] == tf]
        if matches:
            s = matches[0]
            status = "❌" if s["issues"] != "OK" else "✅"
            print(f"{sym:<12} {tf:<5} {s['rows']:>6} {s['first']:<22} {s['last']:<22} {s['span_days']:>8} {status:<8}")

# 按周期统计行数范围
print("\n" + "=" * 80)
print("各周期行数统计:")
print("=" * 80)
for tf in TIMEFRAMES:
    tf_stats = [s for s in stats if s["tf"] == tf]
    if tf_stats:
        rows = [s["rows"] for s in tf_stats]
        print(f"  {tf}: 品种数={len(tf_stats)}, 行数范围={min(rows)}~{max(rows)}, 平均={int(np.mean(rows))}")

print("\n检查完成！")
