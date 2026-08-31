# -*- coding: utf-8 -*-
"""探索 tqsdk get_kline_serial 的周期和 data_length 上限。"""
from tqsdk import TqApi, TqAuth, TqSim
from web.settings import load_settings

settings = load_settings()
user = settings.get("tqsdk_user", "")
pwd = settings.get("tqsdk_password", "")
if not user or not pwd:
    raise RuntimeError("请先设置环境变量 TQSDK_USER 和 TQSDK_PASSWORD")

api = TqApi(TqSim(), auth=TqAuth(user, pwd), disable_print=True)

symbol = "KQ.m@SHFE.ru"

# 测试各种周期
test_durations = [
    (60, "1m"),
    (180, "3m"),
    (300, "5m"),
    (900, "15m"),
    (1800, "30m"),
    (3600, "1h"),
    (7200, "2h"),
    (14400, "4h"),
    (86400, "1d"),
    (604800, "1w"),
]

print("=" * 70)
print("tqsdk 周期 & data_length 上限测试")
print("=" * 70)

for dur, name in test_durations:
    # 先试 8000
    try:
        df = api.get_kline_serial(symbol, dur, data_length=8000, adj_type="F")
        got_8000 = len(df)
    except Exception as e:
        got_8000 = f"ERROR: {e}"
        df = None

    # 再试 50000
    try:
        df2 = api.get_kline_serial(symbol, dur, data_length=50000, adj_type="F")
        got_50000 = len(df2)
    except Exception as e:
        got_50000 = f"ERROR: {e}"

    # 再试 100000
    try:
        df3 = api.get_kline_serial(symbol, dur, data_length=100000, adj_type="F")
        got_100000 = len(df3)
    except Exception as e:
        got_100000 = f"ERROR: {e}"

    print(f"{name:>4s} (dur={dur:>6d}):  len_8000={got_8000}  len_50000={got_50000}  len_100000={got_100000}")

api.close()
print("done")
