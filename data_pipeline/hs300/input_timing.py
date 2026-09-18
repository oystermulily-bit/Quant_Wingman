"""Fail-closed, timezone-aware availability for the v3 reference builder."""
import numpy as np
import pandas as pd
from .time_policy import available_at

RETURN_TIMES = ('quote_available_at', 'preclose_available_at',
                'previous_quote_available_at', 'status_available_at')
REQUIRED_TIMES = (*RETURN_TIMES, 'membership_available_at', 'industry_available_at')


def aware_utc(values, name):
    values = pd.Series(values, copy=False)
    for value in values.dropna().unique():
        try:
            parsed = pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name}: invalid timestamp') from exc
        if pd.isna(parsed) or parsed.tzinfo is None:
            raise ValueError(f'{name}: timestamp must carry an explicit timezone')
    return pd.to_datetime(values, utc=True, errors='raise')


def signal_times(frame):
    result = (aware_utc(frame['signal_at'], 'signal_at') if 'signal_at' in frame
              else available_at(frame['date']).dt.tz_convert('UTC'))
    if result.isna().any():
        raise ValueError('Missing signal_at')
    day = result.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None).dt.normalize()
    if not day.eq(pd.to_datetime(frame['date']).dt.normalize()).all():
        raise ValueError('signal_at must belong to its signal date')
    if not result.groupby(frame['date']).nunique().eq(1).all():
        raise ValueError('Cross-sectional rows must share a signal_at per date')
    return result


def mask_unavailable(frame):
    missing = set(REQUIRED_TIMES) - set(frame)
    if missing:
        raise ValueError(f'Missing availability columns: {sorted(missing)}; normalize source explicitly')
    out = frame.copy()
    cutoff = signal_times(out)
    out['signal_at'] = cutoff
    masks = {}
    for column in REQUIRED_TIMES:
        out[column] = aware_utc(out[column], column)
        masks[column] = out[column].notna() & out[column].le(cutoff)
    out['return_time_usable'] = np.logical_and.reduce([masks[c] for c in RETURN_TIMES])
    out['return_available_at'] = out[list(RETURN_TIMES)].max(axis=1).where(
        out[list(RETURN_TIMES)].notna().all(axis=1))
    out['membership_time_usable'] = masks['membership_available_at']
    out['industry_time_usable'] = masks['industry_available_at']
    if 'available_at' in out:
        generic = aware_utc(out['available_at'], 'available_at')
        out['return_time_usable'] &= generic.notna() & generic.le(cutoff)
    out.loc[~out.industry_time_usable, 'industry_code'] = pd.NA
    eligible = out.return_time_usable & out.membership_time_usable
    # Added tech membership depends on knowing its technology classification, even for market LOO.
    eligible &= out.pool_origin.eq('original_csi300') | (
        out.industry_time_usable & out.industry_code.notna())
    out['reference_input_usable'] = eligible
    out.loc[~eligible, 'r_1d'] = np.nan
    return out
