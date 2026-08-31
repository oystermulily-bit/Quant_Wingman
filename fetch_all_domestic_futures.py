# -*- coding: utf-8 -*-
"""批量拉取全部 60 个国内期货品种 × 10 个周期的主连数据。

tqsdk data_length=10000（上限），前复权 adj_type="F"。
每个品种每个周期建连接→拉数据→关连接，间隔 0.3s。
"""
import time
import sys
from pathlib import Path

import pandas as pd
from datetime import timezone, timedelta
from tqsdk import TqApi, TqAuth, TqSim
from web.settings import load_settings

# ── 配置 ──
settings = load_settings()
user = settings.get("tqsdk_user", "")
pwd = settings.get("tqsdk_password", "")
if not user or not pwd:
    raise RuntimeError("请先设置环境变量 TQSDK_USER 和 TQSDK_PASSWORD")

DATA_LENGTH = 10000
BJ_TZ = timezone(timedelta(hours=8))
OUT_DIR = Path(r"D:\国内期货K线数据")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 全部 60 品种映射
SYMBOL_MAP = {
    "沪铜主连": "KQ.m@SHFE.cu", "沪铝主连": "KQ.m@SHFE.al", "沪锌主连": "KQ.m@SHFE.zn",
    "沪铅主连": "KQ.m@SHFE.pb", "沪镍主连": "KQ.m@SHFE.ni", "沪锡主连": "KQ.m@SHFE.sn",
    "沪金主连": "KQ.m@SHFE.au", "沪银主连": "KQ.m@SHFE.ag", "螺纹钢主连": "KQ.m@SHFE.rb",
    "热卷主连": "KQ.m@SHFE.hc", "橡胶主连": "KQ.m@SHFE.ru", "燃油主连": "KQ.m@SHFE.fu",
    "沥青主连": "KQ.m@SHFE.bu", "纸浆主连": "KQ.m@SHFE.sp", "不锈钢主连": "KQ.m@SHFE.ss",
    "铁矿石主连": "KQ.m@DCE.i", "焦炭主连": "KQ.m@DCE.j", "焦煤主连": "KQ.m@DCE.jm",
    "豆粕主连": "KQ.m@DCE.m", "豆油主连": "KQ.m@DCE.y", "棕榈油主连": "KQ.m@DCE.p",
    "玉米主连": "KQ.m@DCE.c", "淀粉主连": "KQ.m@DCE.cs", "豆一主连": "KQ.m@DCE.a",
    "豆二主连": "KQ.m@DCE.b", "塑料主连": "KQ.m@DCE.l", "PVC主连": "KQ.m@DCE.v",
    "聚丙烯主连": "KQ.m@DCE.pp", "LPG主连": "KQ.m@DCE.pg", "乙二醇主连": "KQ.m@DCE.eg",
    "苯乙烯主连": "KQ.m@DCE.eb",
    "白糖主连": "KQ.m@CZCE.SR", "棉花主连": "KQ.m@CZCE.CF", "PTA主连": "KQ.m@CZCE.TA",
    "甲醇主连": "KQ.m@CZCE.MA", "菜油主连": "KQ.m@CZCE.OI", "菜粕主连": "KQ.m@CZCE.RM",
    "玻璃主连": "KQ.m@CZCE.FG", "纯碱主连": "KQ.m@CZCE.SA", "硅铁主连": "KQ.m@CZCE.SF",
    "锰硅主连": "KQ.m@CZCE.SM", "苹果主连": "KQ.m@CZCE.AP", "红枣主连": "KQ.m@CZCE.CJ",
    "尿素主连": "KQ.m@CZCE.UR", "烧碱主连": "KQ.m@CZCE.SH",
    "沪深300主连": "KQ.m@CFFEX.IF", "中证500主连": "KQ.m@CFFEX.IC",
    "上证50主连": "KQ.m@CFFEX.IH", "中证1000主连": "KQ.m@CFFEX.IM",
    "10年国债主连": "KQ.m@CFFEX.T", "5年国债主连": "KQ.m@CFFEX.TF",
    "2年国债主连": "KQ.m@CFFEX.TS", "30年国债主连": "KQ.m@CFFEX.TL",
    "原油主连": "KQ.m@INE.sc", "低硫燃油主连": "KQ.m@INE.lu",
    "20号胶主连": "KQ.m@INE.nr", "国际铜主连": "KQ.m@INE.bc",
    "欧线集运主连": "KQ.m@INE.ec",
    "工业硅主连": "KQ.m@GFEX.si", "碳酸锂主连": "KQ.m@GFEX.lc",
}

# 10 个周期
TIMEFRAMES = [
    (60,      "M1"),
    (180,     "M3"),
    (300,     "M5"),
    (900,     "M15"),
    (1800,    "M30"),
    (3600,    "H1"),
    (7200,    "H2"),
    (14400,   "H4"),
    (86400,   "D1"),
    (604800,  "W1"),
]

total = len(SYMBOL_MAP) * len(TIMEFRAMES)
done = 0
ok = 0
fail = 0
skipped = 0

print(f"tqsdk: {user}")
print(f"品种: {len(SYMBOL_MAP)} 个 × 周期: {len(TIMEFRAMES)} 个 = {total} 个文件")
print("=" * 80)
sys.stdout.flush()

for sym_name, tqsdk_sym in SYMBOL_MAP.items():
    for dur, tf_name in TIMEFRAMES:
        done += 1
        out_file = OUT_DIR / f"{sym_name}_{tf_name}.parquet"

        # 跳过已存在的文件（避免重复拉取，除非用 --force）
        # 这里不跳过，全部重新拉取

        api = None
        df_raw = None
        try:
            api = TqApi(TqSim(), auth=TqAuth(user, pwd), disable_print=True)
            df_raw = api.get_kline_serial(tqsdk_sym, dur, data_length=DATA_LENGTH, adj_type="F")
        except Exception as exc:
            print(f"[{done}/{total}] {sym_name} {tf_name}: ERROR {exc}")
            fail += 1
            sys.stdout.flush()
            continue
        finally:
            if api is not None:
                try: api.close()
                except Exception: pass

        if df_raw is None or df_raw.empty:
            print(f"[{done}/{total}] {sym_name} {tf_name}: NO DATA")
            fail += 1
            sys.stdout.flush()
            time.sleep(0.3)
            continue

        # 过滤
        df = df_raw[df_raw["datetime"] != 0].copy()
        df = df[df["volume"] > 0].copy()

        if len(df) == 0:
            print(f"[{done}/{total}] {sym_name} {tf_name}: EMPTY after filter")
            fail += 1
            sys.stdout.flush()
            time.sleep(0.3)
            continue

        # 转换
        result = pd.DataFrame()
        result["time"] = (df["datetime"].astype("int64") // 1_000_000_000).astype("int64")
        result["open"] = df["open"].astype("float64")
        result["high"] = df["high"].astype("float64")
        result["low"] = df["low"].astype("float64")
        result["close"] = df["close"].astype("float64")
        result["tick_volume"] = df["volume"].astype("int64")
        result = result.sort_values("time").reset_index(drop=True)
        result = result.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

        result.to_parquet(out_file, index=False)

        first = pd.to_datetime(int(result["time"].iloc[0]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%Y-%m-%d")
        last = pd.to_datetime(int(result["time"].iloc[-1]), unit="s", utc=True).tz_convert(BJ_TZ).strftime("%m-%d")
        o, h, l, c = result["open"], result["high"], result["low"], result["close"]
        bad = int(((h < o) | (h < c) | (l > o) | (l > c)).sum())
        print(f"[{done}/{total}] {sym_name} {tf_name}: {len(result)} rows, {first}~{last}, bad={bad}")
        sys.stdout.flush()
        ok += 1
        time.sleep(0.3)

    # 每个品种完成后短暂休息
    time.sleep(0.5)

print("\n" + "=" * 80)
print(f"完成: total={total} ok={ok} fail={fail}")
print("=" * 80)
