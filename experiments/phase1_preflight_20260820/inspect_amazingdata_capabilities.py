"""Inspect AmazingData temporal fields without printing credentials or market values."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import pandas as pd


CODE = "000001.SZ"
INDEX_CODE = "000300.SH"


def describe(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return {
            "kind": "DataFrame",
            "rows": int(len(value)),
            "columns": [str(column) for column in value.columns],
            "index_name": None if value.index.name is None else str(value.index.name),
            "index_type": str(value.index.dtype),
        }
    if isinstance(value, dict):
        frames = {
            str(key): describe(frame)
            for key, frame in value.items()
            if isinstance(frame, pd.DataFrame)
        }
        return {
            "kind": "dict",
            "keys": sorted(str(key) for key in value.keys()),
            "frames": frames,
        }
    return {"kind": type(value).__name__}


def call_supported(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(method)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return method(*args, **kwargs)
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return method(*args, **supported)
    except (TypeError, ValueError):
        return method(*args, **kwargs)


def inspect_call(
    output: dict[str, Any], name: str, method: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    try:
        try:
            method_signature = str(inspect.signature(method))
        except (TypeError, ValueError):
            method_signature = "unavailable"
        value = call_supported(method, *args, **kwargs)
        output[name] = {
            "success": True,
            "signature": method_signature,
            **describe(value),
        }
    except Exception as exc:  # capability audit must report every independent failure
        output[name] = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> int:
    required = ("AD_USERNAME", "AD_PASSWORD", "AD_HOST", "AD_PORT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    import AmazingData as ad

    report: dict[str, Any] = {
        "sdk_version": version("AmazingData"),
        "queries": {},
        "raw_values_emitted": False,
    }
    username = os.environ["AD_USERNAME"]
    logged_in = False
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            logged_in = bool(
                ad.login(
                    username=username,
                    password=os.environ["AD_PASSWORD"],
                    host=os.environ["AD_HOST"],
                    port=int(os.environ["AD_PORT"]),
                )
            )
        if not logged_in:
            raise RuntimeError("AmazingData login failed")

        base = ad.BaseData()
        info = ad.InfoData()
        queries: dict[str, Any] = report["queries"]

        inspect_call(queries, "stock_basic", info.get_stock_basic, [CODE])
        inspect_call(
            queries,
            "history_stock_status",
            info.get_history_stock_status,
            [CODE],
            begin_date=20200101,
            end_date=20201231,
            is_local=False,
        )
        inspect_call(
            queries,
            "dividend",
            info.get_dividend,
            [CODE],
            begin_date=20130101,
            end_date=20260820,
            is_local=False,
        )
        inspect_call(
            queries,
            "right_issue",
            info.get_right_issue,
            [CODE],
            begin_date=20130101,
            end_date=20260820,
            is_local=False,
        )
        inspect_call(
            queries,
            "equity_structure",
            info.get_equity_structure,
            [CODE],
            begin_date=20130101,
            end_date=20260820,
            is_local=False,
        )
        inspect_call(
            queries,
            "announcement_stock_list",
            info.get_announcement_stock_list,
            [CODE],
            begin_date=20200101,
            end_date=20201231,
            is_local=False,
        )
        inspect_call(
            queries,
            "index_constituent",
            info.get_index_constituent,
            [INDEX_CODE],
            is_local=False,
        )
        inspect_call(
            queries,
            "index_weight",
            info.get_index_weight,
            [INDEX_CODE],
            begin_date=20200101,
            end_date=20200131,
            is_local=False,
        )

        calendar = base.get_calendar()
        market = ad.MarketData(calendar)
        inspect_call(
            queries,
            "daily_kline",
            market.query_kline,
            [CODE],
            begin_date=20200101,
            end_date=20200131,
            period=ad.constant.Period.day.value,
            is_local=False,
        )

        with tempfile.TemporaryDirectory(prefix="w1ngman_ad_audit_") as cache:
            inspect_call(
                queries,
                "single_adjustment_factor",
                base.get_adj_factor,
                [CODE],
                local_path=str(Path(cache).resolve()),
                is_local=False,
            )

        all_columns: set[str] = set()
        for query in queries.values():
            all_columns.update(str(column).upper() for column in query.get("columns", []))
            for frame in query.get("frames", {}).values():
                all_columns.update(str(column).upper() for column in frame.get("columns", []))
        report["temporal_field_presence"] = {
            "literal_known_at": "KNOWN_AT" in all_columns,
            "literal_available_at": "AVAILABLE_AT" in all_columns,
            "announcement_fields": sorted(
                column
                for column in all_columns
                if "ANN" in column or "PUBLISH" in column or "PREPLAN" in column
            ),
            "effective_date_fields": sorted(
                column
                for column in all_columns
                if column in {
                    "INDATE",
                    "OUTDATE",
                    "DATE_EX",
                    "CHANGE_DATE",
                    "EX_CHANGE_DATE",
                    "EX_DIVIDEND_DATE",
                    "DELISTDATE",
                    "TRADE_DATE",
                }
            ),
            "native_open_tr": "OPEN_TR" in all_columns,
            "delisting_fields": sorted(column for column in all_columns if "DELIST" in column),
            "disposition_or_compensation_fields": sorted(
                column
                for column in all_columns
                if any(token in column for token in ("DISPOSITION", "COMPENSATION", "CASH_SETTLE"))
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        if logged_in:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    ad.logout(username)
                except TypeError:
                    ad.logout()


if __name__ == "__main__":
    raise SystemExit(main())
