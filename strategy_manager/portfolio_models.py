"""Causal return calibration and risk estimation for reviewed offline portfolios.

Input histories are explicit, versioned research artifacts, never discovered
market files. OOF provenance is validated structurally, not proved by a flag.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf


RULE_VERSION = "PORTFOLIO_RULES_20260918_V3"


class AllocationError(ValueError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def require(condition, error_code, message, **details):
    if not condition:
        raise AllocationError(error_code, message, details)


def stamp(value):
    try:
        value = pd.Timestamp(value)
        assert not pd.isna(value) and value.tzinfo is not None
        return value.tz_convert("UTC")
    except (ValueError, TypeError, AssertionError, OverflowError) as exc:
        raise AllocationError("INVALID_INPUT_TIME", "配置数据时间必须显式包含时区") from exc


def matrix(value, shape, label):
    try:
        raw = np.asarray(value, dtype=object)
        assert all(x is None or isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw.flat)
        result = np.asarray(value, dtype=float)
        assert result.shape == shape
        assert not np.isinf(result).any()
    except (ValueError, TypeError, AssertionError, OverflowError) as exc:
        raise AllocationError("INVALID_RESEARCH_MATRIX", f"{label}维度或数值无效") from exc
    return result


def percentile(values):
    """Midrank percentile in [-.5,.5]; current selection never changes ranks."""
    return (rankdata(values, method="average") - .5) / len(values) - .5


def fit_calibrator(history: dict, frozen_at, *, draws=2000, seed=20260917):
    times = [stamp(t) for t in history.get("signal_at", [])]
    codes = history.get("codes", [])
    require(252 <= len(times) <= 5000 and len(set(times)) == len(times) and times == sorted(times),
            "INSUFFICIENT_CALIBRATION", "校准至少需要252个有序、已成熟的样本外信号日")
    require(1 < len(codes) <= 2000 and len(set(codes)) == len(codes),
            "INVALID_CALIBRATION_UNIVERSE", "校准股票全集无效")
    ends = [stamp(t) for t in history.get("label_end_at", [])]
    fits = [stamp(t) for t in history.get("forecast_fit_end_at", [])]
    flags = history.get("oof", [])
    require(len(ends) == len(fits) == len(flags) == len(times),
            "INVALID_CALIBRATION_PROVENANCE", "校准时间与样本外证据不完整")
    require(all(flag is True and fit < signal < end < frozen_at
                for flag, fit, signal, end in zip(flags, fits, times, ends)),
            "CALIBRATION_TIME_LEAKAGE", "校准必须使用拟合结束后的预测和冻结时点之前已到期的标签")
    shape = len(times), len(codes)
    scores = matrix(history.get("scores"), shape, "历史评分")
    returns = matrix(history.get("forward_excess_returns"), shape, "已到期收益")
    raw_members = np.asarray(history.get("members"))
    require(raw_members.shape == shape and raw_members.dtype == np.bool_,
            "INVALID_CALIBRATION_MEMBERS", "必须显式提供逐日布尔成员矩阵")
    require(np.all(raw_members.sum(axis=1) == 300), "INCOMPLETE_CALIBRATION_UNIVERSE",
            "收益校准要求历史每期完整300成员，不能只使用当前白名单")
    stats = []
    for s, y, members in zip(scores[-504:], returns[-504:], raw_members[-504:]):
        valid = members & np.isfinite(s) & np.isfinite(y)
        require(valid.sum() >= 285, "INSUFFICIENT_CALIBRATION_COVERAGE", "校准日期有效成员不足95%")
        require(np.all(y[valid] > -1) and np.all(y[valid] < 10),
                "INVALID_FORWARD_RETURN", "校准收益必须是小数比例的毛超额收益")
        available = members & np.isfinite(s)
        all_q = np.full(len(s), np.nan)
        all_q[available] = percentile(s[available])
        q = all_q[valid]
        v = y[valid]
        stats.append([q.mean(), v.mean(), np.mean(q*q), np.mean(q*v)])
    stats = np.asarray(stats)

    def coefficients(averages):
        x, y, xx, xy = np.moveaxis(averages, -1, 0)
        b = np.maximum(0., (xy-x*y)/np.maximum(xx-x*x, 1e-12))
        return np.stack([y-b*x, b], axis=-1)

    ab = coefficients(stats.mean(axis=0))
    rng = np.random.default_rng(seed)
    block = 20
    starts = rng.integers(0, len(stats)-block+1, size=(draws, math.ceil(len(stats)/block)))
    indices = (starts[..., None]+np.arange(block)).reshape(draws, -1)[:, :len(stats)]
    boot = coefficients(stats[indices].mean(axis=1))
    coef_cov = np.cov(boot, rowvar=False, ddof=1)
    require(ab[1] > 0 and np.quantile(boot[:, 1], .05) > 0,
            "CALIBRATION_HAS_NO_DIRECTIONAL_EVIDENCE", "样本外评分到收益的正向校准证据不足")
    return {"method": "DATE_BALANCED_MONOTONE_LINEAR_OOF", "intercept": float(ab[0]),
            "slope": float(ab[1]), "coefficient_covariance": coef_cov.tolist(),
            "date_count": len(stats), "first_signal_at": times[-len(stats)].isoformat(),
            "last_label_end_at": max(ends).isoformat(), "frozen_at": frozen_at.isoformat(),
            "bootstrap_block": block, "bootstrap_draws": draws, "seed": seed,
            "oof_provenance": "DECLARED_AND_TIME_CHECKED_NOT_AN_UPSTREAM_MODEL_AUDIT"}


def prepare_inputs(payload: dict, snapshot: dict):
    require(payload.get("schema_version") == "portfolio_research_inputs_v1",
            "INVALID_RESEARCH_INPUT_VERSION", "配置输入包版本不支持")
    require(payload.get("data_role") == snapshot["data_role"] and payload.get("production_allowed") is False
            and payload.get("holdout_read") is False,
            "INVALID_RESEARCH_INPUT_ROLE", "配置输入必须与评分包同类型且明确为非生产数据")
    require(payload.get("fingerprints") == snapshot["fingerprints"]
            and payload.get("date") == snapshot["date"] and payload.get("fold_id") == snapshot["fold_id"],
            "RESEARCH_INPUT_SNAPSHOT_MISMATCH", "配置证据不属于当前评分、日期或验证折")
    as_of = stamp(snapshot["available_at"])
    frozen = stamp(payload.get("calibration_frozen_at"))
    require(frozen <= as_of and as_of < stamp("2024-08-26T00:00:00+08:00"),
            "RESEARCH_INPUT_TIME_LEAKAGE", "配置输入不能晚于评分时点或进入Holdout")
    horizon = payload.get("horizon")
    require(type(horizon) is int and horizon in (1, 3, 5), "INVALID_HORIZON", "配置周期必须为冻结的1、3或5日")
    require(payload.get("return_semantics") == "GROSS_TOTAL_RETURN_MINUS_CASH_SAME_EXECUTION_HORIZON",
            "INVALID_RETURN_SEMANTICS", "必须提供与执行周期一致的毛总收益减现金收益")
    codes = payload.get("codes", [])
    require(isinstance(codes, list) and 1 < len(codes) <= 1300 and len(set(codes)) == len(codes)
            and all(isinstance(c, str) for c in codes), "INVALID_RISK_UNIVERSE", "风险股票全集无效")
    score_rows = {r["code"]: r for r in snapshot["rows"]}
    require(set(score_rows).issubset(codes), "INCOMPLETE_RISK_UNIVERSE", "风险输入必须覆盖当前全部评分成员")
    require(payload.get("current_scores") == {c: r["score"] for c, r in score_rows.items()},
            "RESEARCH_INPUT_SCORE_MISMATCH", "配置证据与当前原始股票评分不一致")
    calibrator = fit_calibrator(payload.get("calibration", {}), frozen)
    risk = payload.get("risk", {})
    calendar = [stamp(t) for t in risk.get("trading_at", [])]
    require(61 <= len(calendar) <= 505 and calendar == sorted(set(calendar)) and calendar[-1] <= as_of,
            "INVALID_RISK_CALENDAR", "风险日历必须为不晚于评分的最近最多504个交易日区间")
    loc = {t: i for i, t in enumerate(calendar)}
    starts = [stamp(t) for t in risk.get("start_at", [])]
    ends = [stamp(t) for t in risk.get("end_at", [])]
    require(len(starts) == len(ends) and 60 <= len(starts) <= 504,
            "INSUFFICIENT_RISK_HISTORY", "至少需要60个非重叠H周期收益")
    require(all(s in loc and e in loc and loc[e]-loc[s] == horizon for s, e in zip(starts, ends))
            and all(ends[i] <= starts[i+1] for i in range(len(starts)-1)),
            "INVALID_RISK_PERIODS", "风险收益周期必须匹配H且不重叠")
    require(calendar[-1].date() == as_of.date() and loc[ends[-1]] >= len(calendar)-horizon,
            "STALE_RISK_HISTORY", "风险历史必须覆盖当前交易日，不能用过期协方差支持当期配置")
    values = matrix(risk.get("returns"), (len(ends), len(codes)), "风险收益")
    usable = np.isfinite(values).all(axis=0) & (values > -1).all(axis=0) & (values < 10).all(axis=0)
    good_codes = [c for c, good in zip(codes, usable) if good]
    require(bool(good_codes), "INSUFFICIENT_RISK_HISTORY", "没有可用的完整风险历史")
    estimator = LedoitWolf().fit(values[:, usable])
    covariance = estimator.covariance_
    valid_codes = [c for c, r in score_rows.items() if r["signal_valid"]]
    q = percentile(np.array([score_rows[c]["score"] for c in valid_codes]))
    coeff_cov = np.array(calibrator["coefficient_covariance"])
    expected = {}
    for c, percentile_value in zip(valid_codes, q):
        x = np.array([1., percentile_value])
        expected[c] = {"expected_return": calibrator["intercept"]+calibrator["slope"]*float(percentile_value),
                       "standard_error": float(np.sqrt(max(0., x@coeff_cov@x))),
                       "full_universe_percentile": float(percentile_value+.5)}
    metadata = payload.get("market", {})
    require(isinstance(metadata, dict), "INVALID_MARKET_INPUT", "股票行业与执行元数据格式无效")
    for c, item in metadata.items():
        require(c in codes and isinstance(item, dict) and stamp(item.get("available_at")) <= as_of,
                "MARKET_INPUT_TIME_LEAKAGE", "股票行业／价格／流动性数据不晚于评分时点")
        require(item.get("industry") is None or isinstance(item["industry"], str)
                and 0 < len(item["industry"].strip()) <= 100, "INVALID_MARKET_INPUT", "行业名称无效")
        for key in ("price", "median_amount_20d", "mean_amount_20d"):
            if key in item:
                value = item[key]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        and (key != "price" or value > 0), "INVALID_MARKET_INPUT", "价格或流动性输入无效", field=key)
        for key in ("lot_size", "sellable_quantity"):
            if key in item:
                require(type(item[key]) is int and item[key] >= (1 if key == "lot_size" else 0),
                        "INVALID_MARKET_INPUT", "交易单位或可卖数量无效", field=key)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    identity = canonical+"|"+snapshot["snapshot_id"]+"|"+RULE_VERSION
    return {"analysis_id": "analysis_"+sha256(identity.encode()).hexdigest()[:24],
            "snapshot_id": snapshot["snapshot_id"], "rule_version": RULE_VERSION,
            "fingerprints": snapshot["fingerprints"], "date": snapshot["date"], "fold_id": snapshot["fold_id"],
            "available_at": as_of.isoformat(), "data_role": snapshot["data_role"], "production_allowed": False,
            "horizon": horizon, "calibrator": calibrator, "expected": expected,
            "risk_codes": good_codes, "covariance": covariance.tolist(),
            "risk_method": "LEDOIT_WOLF_NONOVERLAPPING_H_RETURNS", "risk_period_count": len(ends),
            "risk_shrinkage": float(estimator.shrinkage_), "risk_end_at": ends[-1].isoformat(),
            "missing_risk_codes": [c for c, good in zip(codes, usable) if not good], "market": metadata,
            "source_sha256": sha256(canonical.encode()).hexdigest()}
