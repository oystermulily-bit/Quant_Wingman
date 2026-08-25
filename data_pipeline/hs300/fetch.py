"""Fetch and freeze AmazingData raw responses for the CSI 300 PIT snapshot."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from .config import (
    BATCH_SIZE,
    INDEX_CODE,
    RAW_SNAPSHOT_ROOT,
    REQUESTED_START,
    SNAPSHOT_ROOT,
)
from .raw_store import RawStore
from .sdk_client import AmazingDataClient

TZ = ZoneInfo("Asia/Shanghai")
SESSION_CLOSE_HOUR = 15


def latest_completed_trading_day(calendar: pd.DataFrame, now: datetime) -> int:
    dates = sorted(int(value) for value in calendar["date"])
    today = int(now.strftime("%Y%m%d"))
    past = [item for item in dates if item <= today]
    last = past[-1]
    if last == today and now.hour < SESSION_CLOSE_HOUR:
        prior = [item for item in past if item < today]
        if not prior:
            raise RuntimeError("no completed trading day before today")
        return prior[-1]
    return last


def year_window(year: int, end_date: int) -> tuple[int, int]:
    begin = year * 10000 + 101
    end = min(year * 10000 + 1231, end_date)
    return begin, end


def chunks(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield index // size, items[index : index + size]


def _run_part(
    store: RawStore,
    client: AmazingDataClient,
    method: str,
    part_id: str,
    parameters: dict,
    loader: Callable[[], pd.DataFrame],
    log: Callable[[str], None],
) -> None:
    if store.is_complete(method, part_id):
        log(f"skip {method}/{part_id}")
        return
    retry = 0
    last_error = None
    while retry < 3:
        try:
            frame = loader()
            meta = store.save(
                method=method,
                part_id=part_id,
                frame=frame,
                parameters=parameters,
                mcp_server_sha256=client.mcp_server_sha256,
                sdk_version=client.sdk_version,
                retry_count=retry,
            )
            log(f"saved {method}/{part_id} rows={meta['row_count']}")
            return
        except SystemExit as exc:
            last_error = f"SystemExit({exc.code})"
            store.save_failure(
                method=method,
                part_id=part_id,
                parameters=parameters,
                error=last_error,
                retry_count=retry,
                mcp_server_sha256=client.mcp_server_sha256,
                sdk_version=client.sdk_version,
            )
            retry += 1
            log(f"retry {retry} {method}/{part_id}: {last_error}")
            time.sleep(2)
            try:
                client.login()
                log("relogin after SystemExit")
            except Exception as login_exc:
                log(f"relogin failed: {login_exc}")
        except Exception as exc:
            last_error = str(exc)
            store.save_failure(
                method=method,
                part_id=part_id,
                parameters=parameters,
                error=last_error,
                retry_count=retry,
                mcp_server_sha256=client.mcp_server_sha256,
                sdk_version=client.sdk_version,
            )
            retry += 1
            log(f"retry {retry} {method}/{part_id}: {last_error}")
            if "SystemExit" in last_error:
                time.sleep(2)
                try:
                    client.login()
                    log("relogin after session loss")
                except Exception as login_exc:
                    log(f"relogin failed: {login_exc}")
    raise RuntimeError(f"{method}/{part_id} failed after retries: {last_error}")


def fetch_raw_snapshot(root: Path = RAW_SNAPSHOT_ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "fetch.log"
    status_path = root / "STATUS.txt"

    def log(message: str) -> None:
        line = f"{datetime.now(TZ).isoformat()} {message}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    status_path.write_text("RAW_SNAPSHOT_IN_PROGRESS\n", encoding="utf-8")
    store = RawStore(root)
    client = AmazingDataClient()
    report: dict = {"snapshot_root": str(root), "parts": []}
    try:
        client.login()
        log(f"login ok sdk={client.sdk_version} mcp_sha={client.mcp_server_sha256}")

        calendar = client.get_calendar()
        _run_part(
            store,
            client,
            "trading_calendar",
            "SH",
            {"market": "SH"},
            lambda: calendar,
            log,
        )
        end_date = latest_completed_trading_day(calendar, datetime.now(TZ))
        start_year = REQUESTED_START.year
        end_year = end_date // 10000
        report["end_date"] = end_date
        report["mcp_server_sha256"] = client.mcp_server_sha256
        log(f"END_DATE={end_date}")

        _run_part(
            store,
            client,
            "index_constituent",
            INDEX_CODE.replace(".", "_"),
            {"index_code": INDEX_CODE},
            client.get_index_constituent,
            log,
        )

        for year in range(start_year, end_year + 1):
            begin, end = year_window(year, end_date)
            if begin > end:
                continue
            _run_part(
                store,
                client,
                "index_weight",
                f"year={year}",
                {"index_code": INDEX_CODE, "begin_date": begin, "end_date": end},
                lambda begin=begin, end=end: client.get_index_weight(begin, end),
                log,
            )

        industry = client.get_industry_base_info()
        _run_part(
            store,
            client,
            "industry_base",
            "all",
            {"method": "get_industry_base_info", "is_local": False},
            lambda: industry,
            log,
        )
        level1 = industry
        if "LEVEL_TYPE" in industry.columns:
            level1 = industry[industry["LEVEL_TYPE"] == 1]
        if "INDEX_CODE" in level1.columns:
            level1 = level1[level1["INDEX_CODE"].astype(str).str.upper().str.endswith(".SI")]
        sw_l1 = (
            sorted(level1["INDEX_CODE"].astype(str).unique().tolist())
            if "INDEX_CODE" in level1.columns
            else []
        )
        report["sw_level1_count"] = len(sw_l1)
        log(f"SW L1 codes={len(sw_l1)}")

        if sw_l1:
            _run_part(
                store,
                client,
                "industry_constituent",
                "sw_l1",
                {"codes": sw_l1},
                lambda: client.get_industry_constituent(sw_l1),
                log,
            )
            for year in range(start_year, end_year + 1):
                begin, end = year_window(year, end_date)
                if begin > end:
                    continue
                _run_part(
                    store,
                    client,
                    "industry_daily",
                    f"sw_l1_year={year}",
                    {"codes": sw_l1, "begin_date": begin, "end_date": end},
                    lambda begin=begin, end=end: client.get_industry_daily(sw_l1, begin, end),
                    log,
                )

        codes: list[str] = []
        for year in range(start_year, end_year + 1):
            path, _ = store.part_paths("index_weight", f"year={year}")
            if path.exists():
                weight = pd.read_parquet(path)
                col = "CON_CODE" if "CON_CODE" in weight.columns else None
                if col:
                    codes.extend(weight[col].astype(str).tolist())
        codes = sorted(set(codes))
        if not codes:
            raise RuntimeError("no CON_CODE found in frozen index_weight parts")
        report["historical_member_count"] = len(codes)
        log(f"historical members={len(codes)}")
        (root / "historical_member_codes.json").write_text(
            json.dumps(codes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for batch_id, batch in chunks(codes, BATCH_SIZE):
            _run_part(
                store,
                client,
                "stock_basic",
                f"batch={batch_id:04d}",
                {"codes": batch},
                lambda batch=batch: client.get_stock_basic(batch),
                log,
            )

        for year in range(start_year, end_year + 1):
            begin, end = year_window(year, end_date)
            if begin > end:
                continue
            _run_part(
                store,
                client,
                "index_kline",
                f"year={year}",
                {"codes": [INDEX_CODE], "begin_date": begin, "end_date": end},
                lambda begin=begin, end=end: client.query_kline([INDEX_CODE], begin, end),
                log,
            )

        kline_start_year = 2013
        for year in range(kline_start_year, end_year + 1):
            begin, end = year_window(year, end_date)
            if begin > end:
                continue
            for batch_id, batch in chunks(codes, BATCH_SIZE):
                _run_part(
                    store,
                    client,
                    "daily_kline",
                    f"year={year}_batch={batch_id:04d}",
                    {"codes": batch, "begin_date": begin, "end_date": end},
                    lambda batch=batch, begin=begin, end=end: client.query_kline(
                        batch, begin, end
                    ),
                    log,
                )
                _run_part(
                    store,
                    client,
                    "stock_status",
                    f"year={year}_batch={batch_id:04d}",
                    {"codes": batch, "begin_date": begin, "end_date": end},
                    lambda batch=batch, begin=begin, end=end: client.get_history_stock_status(
                        batch, begin, end
                    ),
                    log,
                )

        for batch_id, batch in chunks(codes, BATCH_SIZE):
            _run_part(
                store,
                client,
                "dividends",
                f"batch={batch_id:04d}",
                {"codes": batch, "begin_date": 20100101, "end_date": end_date},
                lambda batch=batch: client.get_dividend(batch, 20100101, end_date),
                log,
            )
            _run_part(
                store,
                client,
                "right_issues",
                f"batch={batch_id:04d}",
                {"codes": batch, "begin_date": 20100101, "end_date": end_date},
                lambda batch=batch: client.get_right_issue(batch, 20100101, end_date),
                log,
            )
            _run_part(
                store,
                client,
                "equity_changes",
                f"batch={batch_id:04d}",
                {"codes": batch, "begin_date": 20100101, "end_date": end_date},
                lambda batch=batch: client.get_equity_structure(batch, 20100101, end_date),
                log,
            )

        coverage = {
            "requested_start": REQUESTED_START.isoformat(),
            "end_date": end_date,
            "kline_vendor_start": "2013-01-04",
            "status_sample_start": "2014-01-02",
            "status_2013_sample_rows": 0,
            "index_weight_available_from": "2010-01-04",
            "gap_2010_2012_kline": "UNAVAILABLE_FROM_VENDOR",
            "notes": (
                "2013 kline exists but PRECLOSE/status for 000001.SZ/600000.SH/000002.SZ "
                "is empty in 2013; OpenTR start is max(kline, status)."
            ),
        }
        (root / "coverage_gap.json").write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "snapshot_id": root.name,
            "requested_start": REQUESTED_START.isoformat(),
            "end_date": end_date,
            "historical_member_count": len(codes),
            "sw_level1_count": report.get("sw_level1_count"),
            "mcp_server_sha256": client.mcp_server_sha256,
            "sdk_version": client.sdk_version,
            "status": "RAW_SNAPSHOT_COMPLETE",
            "research_status": "DATA_NOT_FORMALLY_VALIDATED",
            "enhancement_layer": False,
            "frozen": False,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        status_path.write_text(
            "RAW_SNAPSHOT_COMPLETE\nDATA_NOT_FORMALLY_VALIDATED\n",
            encoding="utf-8",
        )
        log("RAW_SNAPSHOT_COMPLETE; research status remains DATA_NOT_FORMALLY_VALIDATED")
        report["status"] = "RAW_SNAPSHOT_COMPLETE"
        report["research_status"] = "DATA_NOT_FORMALLY_VALIDATED"
        return report
    except SystemExit as exc:
        status_path.write_text("RAW_SNAPSHOT_IN_PROGRESS\n", encoding="utf-8")
        raise RuntimeError(f"AmazingData SystemExit({exc.code})") from exc
    except Exception:
        status_path.write_text("RAW_SNAPSHOT_IN_PROGRESS\n", encoding="utf-8")
        raise
    finally:
        client.logout()
