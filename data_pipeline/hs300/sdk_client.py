"""AmazingData SDK client for CSI 300 snapshot builds. Not used during training."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import os
from typing import Any

import pandas as pd

from .config import MCP_SERVER_PATH


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        frames = []
        for key, frame in value.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            item = frame.copy()
            if "MARKET_CODE" not in item.columns and "INDEX_CODE" not in item.columns:
                item["CODE_KEY"] = str(key)
            frames.append(item)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if isinstance(value, list):
        return pd.DataFrame({"value": value})
    return pd.DataFrame()


def _login_hint(text: str) -> str:
    lower = text.lower()
    if "login success" in lower:
        return "login success"
    if "already" in lower:
        return "already logged in"
    if "force" in lower:
        return "force logout"
    return "no login success marker"


class AmazingDataClient:
    def __init__(self) -> None:
        import AmazingData as ad
        from importlib.metadata import version

        self.ad = ad
        self.sdk_version = version("AmazingData")
        self.mcp_server_sha256 = hashlib.sha256(MCP_SERVER_PATH.read_bytes()).hexdigest().upper()
        self._logged_in = False
        self._username = os.environ["AD_USERNAME"].strip()
        self._base = None
        self._info = None
        self._market = None

    def login(self) -> None:
        password = os.environ["AD_PASSWORD"].strip()
        host = os.environ["AD_HOST"].strip()
        port = int(os.environ["AD_PORT"].strip())
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                result = self.ad.login(
                    username=self._username,
                    password=password,
                    host=host,
                    port=port,
                )
        except SystemExit as exc:
            hint = _login_hint(stdout_buf.getvalue() + stderr_buf.getvalue())
            raise RuntimeError(
                f"AmazingData login SystemExit({exc.code}) hint={hint}"
            ) from exc
        text = stdout_buf.getvalue() + stderr_buf.getvalue()
        ok = bool(result) or ("login success" in text.lower())
        if not ok:
            raise RuntimeError(
                f"AmazingData login failed result={result!r} hint={_login_hint(text)}"
            )
        self._logged_in = True
        self._base = self.ad.BaseData()
        self._info = self.ad.InfoData()
        calendar = self._call(self._base.get_calendar, market="SH")
        self._market = self.ad.MarketData(calendar)

    def logout(self) -> None:
        if not self._logged_in:
            return
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                self.ad.logout(self._username)
            except TypeError:
                self.ad.logout()
            except SystemExit:
                pass
        self._logged_in = False

    def _call(self, method, *args, **kwargs):
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    signature = inspect.signature(method)
                    if any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in signature.parameters.values()
                    ):
                        return method(*args, **kwargs)
                    supported = {
                        key: value
                        for key, value in kwargs.items()
                        if key in signature.parameters
                    }
                    return method(*args, **supported)
                except (TypeError, ValueError):
                    return method(*args, **kwargs)
        except SystemExit as exc:
            detail = (stdout_buf.getvalue() + stderr_buf.getvalue()).strip()
            raise RuntimeError(
                f"AmazingData SystemExit({exc.code}) method={getattr(method, '__name__', method)} stdout={detail[:800]!r}"
            ) from exc

    def get_calendar(self) -> pd.DataFrame:
        values = self._call(self._base.get_calendar, market="SH")
        if not isinstance(values, list):
            values = list(values)
        return pd.DataFrame({"date": [int(item) for item in values]})

    def get_index_weight(self, begin_date: int, end_date: int) -> pd.DataFrame:
        from .config import INDEX_CODE

        value = self._call(
            self._info.get_index_weight,
            [INDEX_CODE],
            begin_date=begin_date,
            end_date=end_date,
            is_local=False,
        )
        return _as_frame(value)

    def get_index_constituent(self) -> pd.DataFrame:
        from .config import INDEX_CODE

        return _as_frame(
            self._call(self._info.get_index_constituent, [INDEX_CODE], is_local=False)
        )

    def get_industry_base_info(self) -> pd.DataFrame:
        return _as_frame(self._call(self._info.get_industry_base_info, is_local=False))

    def get_industry_constituent(self, codes: list[str]) -> pd.DataFrame:
        return _as_frame(
            self._call(self._info.get_industry_constituent, codes, is_local=False)
        )

    def get_industry_daily(
        self, codes: list[str], begin_date: int, end_date: int
    ) -> pd.DataFrame:
        return _as_frame(
            self._call(
                self._info.get_industry_daily,
                codes,
                begin_date=begin_date,
                end_date=end_date,
                is_local=False,
            )
        )

    def get_industry_weight(
        self, codes: list[str], begin_date: int, end_date: int
    ) -> pd.DataFrame:
        return _as_frame(
            self._call(
                self._info.get_industry_weight,
                codes,
                begin_date=begin_date,
                end_date=end_date,
                is_local=False,
            )
        )

    def query_kline(self, codes: list[str], begin_date: int, end_date: int) -> pd.DataFrame:
        value = self._call(
            self._market.query_kline,
            codes,
            begin_date=begin_date,
            end_date=end_date,
            period=self.ad.constant.Period.day.value,
            is_local=False,
        )
        frame = _as_frame(value)
        if not frame.empty and "code" in frame.columns:
            frame = frame.rename(columns={"code": "MARKET_CODE"})
        return frame

    def get_history_stock_status(
        self, codes: list[str], begin_date: int, end_date: int
    ) -> pd.DataFrame:
        return _as_frame(
            self._call(
                self._info.get_history_stock_status,
                codes,
                begin_date=begin_date,
                end_date=end_date,
                is_local=False,
            )
        )

    def get_stock_basic(self, codes: list[str]) -> pd.DataFrame:
        return _as_frame(self._call(self._info.get_stock_basic, codes))

    def get_dividend(self, codes: list[str], begin_date: int, end_date: int) -> pd.DataFrame:
        return _as_frame(
            self._call(
                self._info.get_dividend,
                codes,
                begin_date=begin_date,
                end_date=end_date,
                is_local=False,
            )
        )

    def get_right_issue(self, codes: list[str], begin_date: int, end_date: int) -> pd.DataFrame:
        return _as_frame(
            self._call(
                self._info.get_right_issue,
                codes,
                begin_date=begin_date,
                end_date=end_date,
                is_local=False,
            )
        )

    def get_equity_structure(
        self, codes: list[str], begin_date: int, end_date: int
    ) -> pd.DataFrame:
        return _as_frame(
            self._call(
                self._info.get_equity_structure,
                codes,
                begin_date=begin_date,
                end_date=end_date,
                is_local=False,
            )
        )
