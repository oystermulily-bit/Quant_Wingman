"""Live AmazingData coverage probe: row counts and dates only, no bulk snapshot."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import sys
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BOOT = Path(__file__).with_name("probe_boot.txt")
BOOT.write_text(f"boot {datetime.now().isoformat()}\npython={sys.executable}\n", encoding="utf-8")

import pandas as pd


INDEX = "000300.SH"
SAMPLE_STOCK = "000001.SZ"
OUT = Path(__file__).with_name("coverage_report.json")
TZ = ZoneInfo("Asia/Shanghai")


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        frames = [item for item in value.values() if isinstance(item, pd.DataFrame)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _columns(value: Any) -> list[str]:
    frame = _frame(value)
    return [str(column) for column in frame.columns]


def _row_count(value: Any) -> int:
    if isinstance(value, pd.DataFrame):
        return int(len(value))
    if isinstance(value, dict):
        return int(sum(len(item) for item in value.values() if isinstance(item, pd.DataFrame)))
    if isinstance(value, list):
        return int(len(value))
    return 0


def _date_bounds(frame: pd.DataFrame, *names: str) -> dict[str, str | None]:
    lookup = {str(column).upper(): column for column in frame.columns}
    chosen = None
    for name in names:
        if name.upper() in lookup:
            chosen = lookup[name.upper()]
            break
    if chosen is None:
        return {"field": None, "min": None, "max": None, "non_null": 0}
    series = pd.to_datetime(frame[chosen], errors="coerce")
    valid = series.dropna()
    return {
        "field": str(chosen),
        "min": None if valid.empty else valid.min().date().isoformat(),
        "max": None if valid.empty else valid.max().date().isoformat(),
        "non_null": int(len(valid)),
    }


def _safe_call(method, *args, **kwargs):
    try:
        signature = inspect.signature(method)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return method(*args, **kwargs)
        supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return method(*args, **supported)
    except (TypeError, ValueError):
        return method(*args, **kwargs)


def _probe(method, *args, **kwargs) -> dict[str, Any]:
    started = datetime.now(TZ).isoformat()
    if not callable(method):
        return {
            "success": False,
            "started_at": started,
            "completed_at": datetime.now(TZ).isoformat(),
            "error_type": "TypeError",
            "error": f"probe target is not callable: {type(method).__name__}",
        }
    try:
        with Path(__file__).with_name("probe_progress.txt").open("a", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.now(TZ).isoformat()} call {getattr(method, '__qualname__', type(method).__name__)}\n"
            )
        value = _safe_call(method, *args, **kwargs)
        frame = _frame(value)
        result = {
            "success": True,
            "started_at": started,
            "completed_at": datetime.now(TZ).isoformat(),
            "kind": type(value).__name__,
            "rows": _row_count(value),
            "columns": _columns(value),
            "dict_keys_sample": sorted(str(k) for k in list(value.keys())[:8])
            if isinstance(value, dict)
            else None,
        }
        if not frame.empty:
            result["date_bounds"] = {
                field: _date_bounds(frame, field)
                for field in (
                    "TRADE_DATE",
                    "KLINE_TIME",
                    "INDATE",
                    "OUTDATE",
                    "LISTDATE",
                    "DELISTDATE",
                    "DATE_EX",
                    "DATE_DVD_ANN",
                    "ANN_DATE",
                    "PUBLISH_TIME",
                )
                if field in {c.upper() for c in frame.columns}
                or field in set(frame.columns)
            }
        return result
    except Exception as exc:
        return {
            "success": False,
            "started_at": started,
            "completed_at": datetime.now(TZ).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _weight_unit(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "WEIGHT" not in frame.columns:
        return {"detected_unit": "MISSING_WEIGHT_COLUMN"}
    date_col = "TRADE_DATE" if "TRADE_DATE" in frame.columns else None
    weights = pd.to_numeric(frame["WEIGHT"], errors="coerce")
    daily = None
    if date_col:
        tmp = frame.copy()
        tmp["_w"] = weights
        daily = tmp.groupby(date_col)["_w"].sum()
    sums = daily if daily is not None and not daily.empty else pd.Series([float(weights.sum())])
    median = float(sums.median())
    minimum = float(sums.min())
    maximum = float(sums.max())
    if 99.0 <= median <= 101.0:
        unit = "percent_0_100"
    elif 0.99 <= median <= 1.01:
        unit = "fraction_0_1"
    else:
        unit = "UNKNOWN_DO_NOT_GUESS"
    return {
        "detected_unit": unit,
        "daily_sum_min": round(minimum, 6),
        "daily_sum_median": round(median, 6),
        "daily_sum_max": round(maximum, 6),
        "trading_days": int(len(sums)),
        "unique_codes": int(frame["CON_CODE"].nunique()) if "CON_CODE" in frame.columns else None,
        "rows_per_day_min": int(frame.groupby(date_col).size().min()) if date_col else None,
        "rows_per_day_max": int(frame.groupby(date_col).size().max()) if date_col else None,
    }


def _latest_completed_session(calendar_values: list[int], now: datetime) -> dict[str, Any]:
    dates = sorted(int(x) for x in calendar_values if int(x) > 0)
    today = int(now.strftime("%Y%m%d"))
    past_or_today = [d for d in dates if d <= today]
    last_listed = past_or_today[-1] if past_or_today else None
    session_complete = True
    end_date = last_listed
    reason = "calendar_last_on_or_before_today"
    if last_listed == today and now.hour < 15:
        prior = [d for d in past_or_today if d < today]
        end_date = prior[-1] if prior else None
        session_complete = False
        reason = "today_is_trading_day_before_15:00_Asia/Shanghai"
    return {
        "now_asia_shanghai": now.isoformat(),
        "today": today,
        "calendar_min": dates[0] if dates else None,
        "calendar_max": dates[-1] if dates else None,
        "calendar_len": len(dates),
        "today_is_trading_day": last_listed == today,
        "session_complete": session_complete,
        "latest_completed_trading_day": end_date,
        "reason": reason,
    }


def main() -> int:
    required = ("AD_USERNAME", "AD_PASSWORD", "AD_HOST", "AD_PORT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    import AmazingData as ad

    username = os.environ["AD_USERNAME"]
    password = os.environ["AD_PASSWORD"]
    host = os.environ["AD_HOST"]
    port = int(os.environ["AD_PORT"])
    report: dict[str, Any] = {
        "audited_at": datetime.now(TZ).isoformat(),
        "sdk_version": version("AmazingData"),
        "python": sys.executable,
        "raw_values_emitted": False,
        "login": {},
        "sdk_methods": {},
        "queries": {},
        "coverage": {},
        "end_date_policy": {},
    }

    logged_in = False
    login_attempts = []
    progress = Path(__file__).with_name("probe_progress.txt")

    def checkpoint(message: str) -> None:
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(TZ).isoformat()} {message}\n")

    try:
        progress.write_text("", encoding="utf-8")
        checkpoint("main_start")
        checkpoint("imported_AmazingData")
        for label, pwd in (("as_provided", password), ("stripped", password.strip())):
            if label == "stripped" and pwd == password:
                continue
            checkpoint(f"login_try_{label}")
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    ok = bool(ad.login(username=username, password=pwd, host=host, port=port))
            except SystemExit as exc:
                checkpoint(f"login_systemexit_{label}_code={exc.code}")
                ok = False
            login_attempts.append({
                "variant": label,
                "success": bool(ok),
                "password_len": len(pwd),
                "stdout_chars": len(stdout_buf.getvalue()),
                "stderr_chars": len(stderr_buf.getvalue()),
            })
            if ok:
                logged_in = True
                report["login"] = {
                    "success": True,
                    "variant": label,
                    "password_had_trailing_whitespace": password != password.strip(),
                    "host": host,
                    "port": port,
                    "username_len": len(username),
                }
                checkpoint("login_ok")
                break
        if not logged_in:
            report["login"] = {
                "success": False,
                "attempts": login_attempts,
                "password_had_trailing_whitespace": password != password.strip(),
            }
            OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"ok": False, "login": report["login"]}, ensure_ascii=False))
            return 2

        base = ad.BaseData()
        info = ad.InfoData()
        report["sdk_methods"] = {
            "InfoData": sorted(name for name in dir(info) if name.startswith("get_")),
            "BaseData": sorted(name for name in dir(base) if name.startswith("get_")),
            "has_get_industry_base_info": hasattr(info, "get_industry_base_info"),
            "has_get_industry_index_info": hasattr(info, "get_industry_index_info"),
            "has_get_index_weight": hasattr(info, "get_index_weight"),
            "has_get_index_constituent": hasattr(info, "get_index_constituent"),
            "industry_base_signature": str(inspect.signature(info.get_industry_base_info))
            if hasattr(info, "get_industry_base_info")
            else None,
            "industry_index_info_signature": str(inspect.signature(info.get_industry_index_info))
            if hasattr(info, "get_industry_index_info")
            else "MISSING",
        }

        calendar = _safe_call(base.get_calendar, market="SH")
        if not isinstance(calendar, list):
            calendar = list(calendar)
        now = datetime.now(TZ)
        end_policy = _latest_completed_session(calendar, now)
        report["end_date_policy"] = end_policy
        end_int = end_policy["latest_completed_trading_day"]

        queries = report["queries"]
        queries["calendar"] = {
            "success": True,
            "rows": len(calendar),
            "min": calendar[0] if calendar else None,
            "max": calendar[-1] if calendar else None,
        }

        if hasattr(info, "get_industry_index_info"):
            queries["industry_index_info_broken_name"] = _probe(info.get_industry_index_info)
        else:
            queries["industry_index_info_broken_name"] = {
                "success": False,
                "error_type": "AttributeError",
                "error": "SDK has no get_industry_index_info",
            }

        queries["industry_base_info"] = _probe(info.get_industry_base_info, is_local=False)

        queries["index_constituent"] = _probe(
            info.get_index_constituent, [INDEX], is_local=False
        )

        windows = [
            ("kline_2010_jan", 20100101, 20100131),
            ("kline_2012_dec", 20121201, 20121231),
            ("kline_2013_jan", 20130101, 20130131),
            ("kline_2016_jan", 20160101, 20160131),
        ]
        market = ad.MarketData(_safe_call(base.get_calendar))
        for name, begin, end in windows:
            queries[name] = _probe(
                market.query_kline,
                [SAMPLE_STOCK, INDEX],
                begin_date=begin,
                end_date=end,
                period=ad.constant.Period.day.value,
                is_local=False,
            )
            queries[name.replace("kline_", "status_")] = _probe(
                info.get_history_stock_status,
                [SAMPLE_STOCK],
                begin_date=begin,
                end_date=end,
                is_local=False,
            )
            queries[name.replace("kline_", "weight_")] = _probe(
                info.get_index_weight,
                [INDEX],
                begin_date=begin,
                end_date=end,
                is_local=False,
            )

        if end_int:
            queries["weight_latest_month"] = _probe(
                info.get_index_weight,
                [INDEX],
                begin_date=int(pd.Timestamp(str(end_int)).replace(day=1).strftime("%Y%m%d")),
                end_date=end_int,
                is_local=False,
            )
            queries["kline_latest_month"] = _probe(
                market.query_kline,
                [SAMPLE_STOCK],
                begin_date=int(pd.Timestamp(str(end_int)).replace(day=1).strftime("%Y%m%d")),
                end_date=end_int,
                period=ad.constant.Period.day.value,
                is_local=False,
            )

        queries["stock_basic_sample"] = _probe(info.get_stock_basic, [SAMPLE_STOCK])
        queries["dividend_sample"] = _probe(
            info.get_dividend,
            [SAMPLE_STOCK],
            begin_date=20130101,
            end_date=end_int or 20260824,
            is_local=False,
        )
        queries["right_issue_sample"] = _probe(
            info.get_right_issue,
            [SAMPLE_STOCK],
            begin_date=20130101,
            end_date=end_int or 20260824,
            is_local=False,
        )
        queries["equity_sample"] = _probe(
            info.get_equity_structure,
            [SAMPLE_STOCK],
            begin_date=20130101,
            end_date=end_int or 20260824,
            is_local=False,
        )
        queries["announcement_2020"] = _probe(
            info.get_announcement_stock_list,
            [SAMPLE_STOCK],
            begin_date=20200101,
            end_date=20200331,
            is_local=False,
        )

        industry_frame = pd.DataFrame()
        industry_query = queries.get("industry_base_info", {})
        if industry_query.get("success"):
            industry_raw = _safe_call(info.get_industry_base_info, is_local=False)
            industry_frame = _frame(industry_raw)
            report["industry_summary"] = {
                "rows": int(len(industry_frame)),
                "columns": [str(c) for c in industry_frame.columns],
            }
            if "LEVEL_TYPE" in industry_frame.columns:
                counts = industry_frame["LEVEL_TYPE"].value_counts(dropna=False).to_dict()
                report["industry_summary"]["level_type_counts"] = {
                    str(k): int(v) for k, v in counts.items()
                }
            if "INDEX_CODE" in industry_frame.columns:
                codes = industry_frame["INDEX_CODE"].astype(str)
                report["industry_summary"]["index_code_suffix_counts"] = (
                    codes.str.extract(r"(\.[A-Z]+)$")[0].value_counts().head(10).to_dict()
                )
                report["industry_summary"]["index_code_prefix_sample"] = sorted(
                    set(codes.str[:3].tolist())
                )[:20]
            level1 = industry_frame
            if "LEVEL_TYPE" in industry_frame.columns:
                level1 = industry_frame[industry_frame["LEVEL_TYPE"] == 1]
            l1_codes = (
                level1["INDEX_CODE"].astype(str).tolist()
                if "INDEX_CODE" in level1.columns
                else []
            )
            report["industry_summary"]["level1_count"] = int(len(l1_codes))
            if l1_codes:
                sample_industry = l1_codes[:1]
                queries["industry_constituent_l1_sample"] = _probe(
                    info.get_industry_constituent, sample_industry, is_local=False
                )
                queries["industry_daily_2010_jan"] = _probe(
                    info.get_industry_daily,
                    sample_industry,
                    begin_date=20100101,
                    end_date=20100131,
                    is_local=False,
                )
                queries["industry_daily_2013_jan"] = _probe(
                    info.get_industry_daily,
                    sample_industry,
                    begin_date=20130101,
                    end_date=20130131,
                    is_local=False,
                )
                queries["industry_weight_2013_jan"] = _probe(
                    info.get_industry_weight,
                    sample_industry,
                    begin_date=20130101,
                    end_date=20130131,
                    is_local=False,
                )

        # Weight unit from 2013-01 if available, else latest month.
        weight_unit_source = None
        for key in ("weight_2013_jan", "weight_latest_month", "weight_2010_jan"):
            q = queries.get(key, {})
            if q.get("success") and q.get("rows", 0) > 0:
                weight_unit_source = key
                raw = _safe_call(
                    info.get_index_weight,
                    [INDEX],
                    begin_date={"weight_2013_jan": 20130101, "weight_latest_month": int(pd.Timestamp(str(end_int)).replace(day=1).strftime("%Y%m%d")), "weight_2010_jan": 20100101}[key],
                    end_date={"weight_2013_jan": 20130131, "weight_latest_month": end_int, "weight_2010_jan": 20100131}[key],
                    is_local=False,
                )
                report["weight_unit"] = {
                    "source_query": key,
                    **_weight_unit(_frame(raw)),
                }
                break
        if weight_unit_source is None:
            report["weight_unit"] = {"detected_unit": "NO_ROWS"}

        def _first_nonzero(*keys: str) -> str | None:
            for key in keys:
                q = queries.get(key, {})
                if q.get("success") and q.get("rows", 0) > 0:
                    return key
            return None

        report["coverage"] = {
            "requested_start": "2010-01-01",
            "vendor_manual_bar_start": "2013-present",
            "kline_2010_jan_rows": queries.get("kline_2010_jan", {}).get("rows"),
            "kline_2012_dec_rows": queries.get("kline_2012_dec", {}).get("rows"),
            "kline_2013_jan_rows": queries.get("kline_2013_jan", {}).get("rows"),
            "weight_2010_jan_rows": queries.get("weight_2010_jan", {}).get("rows"),
            "weight_2012_dec_rows": queries.get("weight_2012_dec", {}).get("rows"),
            "weight_2013_jan_rows": queries.get("weight_2013_jan", {}).get("rows"),
            "status_2010_jan_rows": queries.get("status_2010_jan", {}).get("rows"),
            "status_2013_jan_rows": queries.get("status_2013_jan", {}).get("rows"),
            "industry_daily_2010_jan_rows": queries.get("industry_daily_2010_jan", {}).get("rows"),
            "industry_daily_2013_jan_rows": queries.get("industry_daily_2013_jan", {}).get("rows"),
            "silent_shortening_allowed": False,
            "first_kline_window_with_rows": _first_nonzero(
                "kline_2010_jan", "kline_2012_dec", "kline_2013_jan", "kline_2016_jan"
            ),
            "first_weight_window_with_rows": _first_nonzero(
                "weight_2010_jan", "weight_2012_dec", "weight_2013_jan"
            ),
        }

        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        BOOT.write_text(BOOT.read_text(encoding="utf-8") + f"login_success={logged_in}\n", encoding="utf-8")
        print(json.dumps({"ok": True, "out": str(OUT), "bytes": OUT.stat().st_size}, ensure_ascii=False))
        return 0
    finally:
        if logged_in:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    ad.logout(username)
                except TypeError:
                    ad.logout()


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:
        Path(__file__).with_name("probe_crash.txt").write_text(
            f"SystemExit {exc.code}\n{traceback.format_exc()}", encoding="utf-8"
        )
        raise
    except BaseException:
        Path(__file__).with_name("probe_crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    raise SystemExit(code)
