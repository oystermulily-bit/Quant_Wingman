import pandas as pd

from scripts.normalize_d21_v3_input_times import previous_quote_times


def test_previous_quote_uses_previous_trading_day_not_last_seen_quote():
    dates = pd.DatetimeIndex(['2020-01-03', '2020-01-06', '2020-01-07'])
    rows = pd.DataFrame({'date': dates, 'code': ['A'] * 3})
    quotes = rows.iloc[[0, 2]].copy()
    quotes['quote_available_at'] = quotes.date.dt.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=21)
    result = previous_quote_times(rows, quotes, dates)
    assert pd.isna(result.previous_quote_available_at.iloc[0])
    assert result.previous_quote_available_at.iloc[1] == quotes.quote_available_at.iloc[0]
    assert pd.isna(result.previous_quote_available_at.iloc[2])


def test_previous_quote_preserves_native_late_time():
    dates = pd.bdate_range('2020-01-01', periods=2)
    rows = pd.DataFrame({'date': dates, 'code': ['A'] * 2})
    quotes = rows.copy()
    quotes['quote_available_at'] = pd.Timestamp('2020-01-03 23:00', tz='Asia/Shanghai')
    result = previous_quote_times(rows, quotes, dates)
    assert result.previous_quote_available_at.iloc[1] == pd.Timestamp('2020-01-03 23:00', tz='Asia/Shanghai')
