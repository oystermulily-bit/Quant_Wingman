# -*- coding: utf-8 -*-
"""验证更新后的橡胶主连数据。"""
import pandas as pd
from datetime import timezone, timedelta

BJ_TZ = timezone(timedelta(hours=8))

for tf in ['M15', 'M30', 'H1', 'D1']:
    df = pd.read_parquet(rf'D:\国内期货K线数据\橡胶主连_{tf}.parquet')
    first = pd.to_datetime(int(df['time'].iloc[0]), unit='s', utc=True).tz_convert(BJ_TZ).strftime('%Y-%m-%d %H:%M')
    last = pd.to_datetime(int(df['time'].iloc[-1]), unit='s', utc=True).tz_convert(BJ_TZ).strftime('%Y-%m-%d %H:%M')
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    bad_h = ((h < o) | (h < c)).sum()
    bad_l = ((l > o) | (l > c)).sum()
    print(f'{tf}: rows={len(df)}, first={first}, last={last}, bad_h={bad_h}, bad_l={bad_l}, neg_price={(c<0).sum()}, zero_vol={(df["tick_volume"]==0).sum()}')
