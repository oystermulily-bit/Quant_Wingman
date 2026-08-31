# -*- coding: utf-8 -*-
"""数据校验脚本：两个维度校验已保存的 parquet 文件。

维度1: 重新拉取 6 个抽样品种的 D1+M15，和已保存文件做逐 bar 对比
维度2: 对全部 600 个文件做深度一致性检查（时间连续性、价格合理性、成交量合理性）
"""
import time
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import timezone, timedelta
from tqsdk import TqApi, TqAuth, TqSim
from web.settings import load_settings

BJ_TZ = timezone(timedelta(hours=8))
DATA_DIR = Path(r"D:\国内期货K线数据")

settings = load_settings()
user = settings.get("tqsdk_user", "")
pwd = settings.get("tqsdk_password", "")
if not user or not pwd:
    raise RuntimeError("请先设置环境变量 TQSDK_USER 和 TQSDK_PASSWORD")

# ── 维度1: 抽样重新拉取对比 ──
SAMPLE_SYMBOLS = [
    ("橡胶主连", "KQ.m@SHFE.ru"),
    ("沪铜主连", "KQ.m@SHFE.cu"),
    ("铁矿石主连", "KQ.m@DCE.i"),
    ("螺纹钢主连", "KQ.m@SHFE.rb"),
    ("沪深300主连", "KQ.m@CFFEX.IF"),
    ("碳酸锂主连", "KQ.m@GFEX.lc"),
]

SAMPLE_TFS = [
    (900, "M15"),
    (86400, "D1"),
]

def fetch_kline(tqsdk_sym, dur, length=10000):
    api = None
    try:
        api = TqApi(TqSim(), auth=TqAuth(user, pwd), disable_print=True)
        df = api.get_kline_serial(tqsdk_sym, dur, data_length=length, adj_type="F")
    finally:
        if api:
            try: api.close()
            except: pass
    if df is None or df.empty:
        return None
    df = df[df["datetime"] != 0].copy()
    df = df[df["volume"] > 0].copy()
    if len(df) == 0:
        return None
    result = pd.DataFrame()
    result["time"] = (df["datetime"].astype("int64") // 1_000_000_000).astype("int64")
    result["open"] = df["open"].astype("float64")
    result["high"] = df["high"].astype("float64")
    result["low"] = df["low"].astype("float64")
    result["close"] = df["close"].astype("float64")
    result["tick_volume"] = df["volume"].astype("int64")
    result = result.sort_values("time").reset_index(drop=True)
    result = result.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return result

print("=" * 100)
print("维度1: 抽样重新拉取 vs 已保存文件 逐 bar 对比")
print("=" * 100)

mismatch_count = 0
total_compared = 0

for sym_name, tqsdk_sym in SAMPLE_SYMBOLS:
    for dur, tf_name in SAMPLE_TFS:
        print(f"\n[{sym_name} {tf_name}] 重新拉取对比中...")
        fresh = fetch_kline(tqsdk_sym, dur)
        if fresh is None:
            print(f"  [ERROR] 重新拉取失败")
            continue

        saved_path = DATA_DIR / f"{sym_name}_{tf_name}.parquet"
        saved = pd.read_parquet(saved_path)

        # 找共同的时间戳
        fresh_times = set(fresh["time"].values)
        saved_times = set(saved["time"].values)
        common_times = sorted(fresh_times & saved_times)

        total_compared += 1
        print(f"  重新拉取: {len(fresh)} rows, 已保存: {len(saved)} rows, 共同: {len(common_times)} rows")

        if len(common_times) == 0:
            print(f"  [ERROR] 无共同时间戳!")
            mismatch_count += 1
            continue

        # 逐 bar 对比
        fresh_idx = fresh.set_index("time")
        saved_idx = saved.set_index("time")

        mismatches = []
        for ts in common_times:
            f_row = fresh_idx.loc[ts]
            s_row = saved_idx.loc[ts]
            for col in ["open", "high", "low", "close", "tick_volume"]:
                fv = f_row[col]
                sv = s_row[col]
                if col == "tick_volume":
                    if int(fv) != int(sv):
                        mismatches.append((ts, col, fv, sv))
                else:
                    if abs(float(fv) - float(sv)) > 1e-6:
                        mismatches.append((ts, col, fv, sv))

        if len(mismatches) == 0:
            print(f"  ✅ 完全一致! {len(common_times)} bars 逐 bar 对比通过")
        else:
            mismatch_count += 1
            print(f"  ❌ 发现 {len(mismatches)} 处不一致:")
            for ts, col, fv, sv in mismatches[:10]:
                dt = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d %H:%M")
                print(f"    {dt}  {col}: fresh={fv} vs saved={sv}")

        # 检查已保存文件是否比重新拉取多了数据（不应该）
        extra_saved = sorted(saved_times - fresh_times)
        if extra_saved:
            print(f"  ⚠️ 已保存文件多出 {len(extra_saved)} 条 (旧数据残留?)")
            for ts in extra_saved[:3]:
                dt = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d %H:%M")
                print(f"    {dt}")

        time.sleep(0.5)

print(f"\n维度1 结果: 对比 {total_compared} 个文件, 不一致 {mismatch_count} 个")

# ── 维度2: 全部文件深度一致性检查 ──
print("\n" + "=" * 100)
print("维度2: 全部 600 个文件深度一致性检查")
print("=" * 100)

files = sorted(DATA_DIR.glob("*.parquet"))
issue_count = 0
issue_list = []

for f in files:
    name = f.stem
    df = pd.read_parquet(f)
    rows = len(df)
    issues = []

    # 1. 时间唯一性（不应有重复时间戳）
    dup_times = df["time"].duplicated().sum()
    if dup_times > 0:
        issues.append(f"重复时间戳 {dup_times} 条")

    # 2. 时间升序
    if not df["time"].is_monotonic_increasing:
        issues.append("时间未升序排列")

    # 3. OHLC 逻辑
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad_h = ((h < o) | (h < c)).sum()
    bad_l = ((l > o) | (l > c)).sum()
    if bad_h > 0:
        issues.append(f"high<max(O,C) {bad_h} 条")
    if bad_l > 0:
        issues.append(f"low>min(O,C) {bad_l} 条")

    # 4. 负价格
    neg = (df[["open", "high", "low", "close"]] < 0).any(axis=1).sum()
    if neg > 0:
        issues.append(f"负价格 {neg} 条")

    # 5. 零成交量
    zero_v = (df["tick_volume"] == 0).sum()
    if zero_v > 0:
        issues.append(f"零成交量 {zero_v} 条 ({zero_v/rows*100:.1f}%)")

    # 6. 时间间隔异常（相邻 bar 时间差应该等于周期）
    # 只检查 D1（日线最容易判断）
    if name.endswith("_D1"):
        diffs = df["time"].diff().dropna()
        # 日线应该大部分是 86400 秒，周末跳过
        abnormal_diff = ((diffs != 86400) & (diffs != 172800) & (diffs != 259200) & (diffs != 345600) & (diffs != 604800)).sum()
        if abnormal_diff > 0:
            issues.append(f"D1时间间隔异常 {abnormal_diff} 条")

    # 7. 价格跳变检查（相邻 close 涨跌>30%）
    if rows > 1:
        rets = np.abs(np.diff(df["close"].values) / df["close"].values[:-1])
        huge = np.sum(rets > 0.3)
        if huge > 0:
            issues.append(f"涨跌>30% {huge} 条")

    # 8. 检查列完整性
    expected_cols = {"time", "open", "high", "low", "close", "tick_volume"}
    actual_cols = set(df.columns)
    if actual_cols != expected_cols:
        issues.append(f"列不匹配: {actual_cols}")

    if issues:
        issue_count += 1
        issue_list.append((name, issues))
        print(f"  ❌ {name}: {'; '.join(issues)}")

if issue_count == 0:
    print("\n✅ 全部 600 个文件通过深度一致性检查！")
else:
    print(f"\n❌ {issue_count} 个文件有问题:")
    for name, issues in issue_list:
        print(f"  {name}: {'; '.join(issues)}")

print("\n" + "=" * 100)
print(f"校验完成: 维度1 对比 {total_compared} 文件 (不一致 {mismatch_count}), 维度2 检查 {len(files)} 文件 (问题 {issue_count})")
print("=" * 100)
