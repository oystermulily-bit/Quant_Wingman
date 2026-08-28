"""Point-in-time HS300 panel loading and immutable snapshot validation.

This module is intentionally separate from the legacy single-symbol loaders.
It never fills missing market rows and never derives membership or industry
history from the latest observation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data_pipeline.hs300.config import INDUSTRY_COVERAGE_MIN, RESEARCH_START
from data_pipeline.hs300.lineage import LINEAGE_COLUMNS
from data_pipeline.hs300.panel_tensors import (
    HS300PanelTensors,
    pack_panel_tensors,
    save_panel_tensors,
)


SNAPSHOT_MANIFEST_VERSION = "w1ngman_raw_snapshot_v2"
PANEL_SCHEMA_VERSION = "w1ngman_hs300_panel_v2"


class SnapshotValidationError(ValueError):
    """Raised when an immutable input snapshot cannot pass the data gate."""


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str
    severity: str = "ERROR"
    row_count: int | None = None


@dataclass
class DataGateReport:
    snapshot_id: str | None
    status: str = "DATA_GATE_FAILED"
    checked_files: int = 0
    trading_dates: int = 0
    panel_rows: int = 0
    issues: list[GateIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "DATA_GATE_PASSED"

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: str = "ERROR",
        row_count: int | None = None,
    ) -> None:
        self.issues.append(GateIssue(code, message, severity, row_count))

    def finish(self) -> "DataGateReport":
        self.status = (
            "DATA_GATE_FAILED"
            if any(issue.severity == "ERROR" for issue in self.issues)
            else "DATA_GATE_PASSED"
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_file(root: Path, rel: str) -> Path:
    """Resolve a snapshot-relative path without treating directory junctions as escapes.

    Windows junctions (v2/raw -> v1/raw) resolve to another tree. The logical
    path must still live under the snapshot root and must not contain '..'.
    """
    relative = Path(str(rel))
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"path escapes snapshot: {rel}")
    return root / relative


class SnapshotManifestValidator:
    """Validate paths, hashes and row counts without changing source files."""

    required_tables = {
        "raw_request_manifest",
        "daily_bars",
        "universe_membership",
        "industry_membership",
        "trading_status",
        "trading_calendar",
        "security_master",
        "corporate_actions",
    }
    table_required_columns: dict[str, set[str]] = {
        "daily_bars": {
            "date", "code", "open_raw", "high_raw", "low_raw", "close_raw",
            "volume", "amount", "preclose", "open_tr", "close_tr",
            "adjustment_type", "available_at", "available_at_source", "has_quote",
            "quote_missing_reason", "revision_id", "source_row_hash",
        },
        "universe_membership": {
            "date", "index_code", "code", "is_member", "weight_pct",
            "effective_at", "known_at", "known_at_source", "entry_effective_date",
            "exit_effective_date", "membership_source", "source_request_id",
        },
        "industry_membership": {
            "code", "industry_system", "level", "industry_code", "industry_name",
            "valid_from", "valid_to", "effective_at", "known_at", "known_at_source",
        },
        "trading_status": {
            "date", "code", "is_suspended", "is_st", "is_xr", "is_wd",
            "limit_up_price", "limit_down_price", "can_buy_open", "can_sell_open",
            "buy_block_reason", "sell_block_reason", "available_at",
            "available_at_source",
        },
        "trading_calendar": {
            "date", "market", "is_trading_day", "session_open", "session_close",
            "is_complete_session", "available_at",
        },
        "security_master": {
            "code", "name", "exchange", "list_date", "delist_date",
            "security_type", "board", "valid_from", "valid_to", "known_at",
        },
        "corporate_actions": {
            "action_id", "code", "action_type", "announcement_at", "record_date",
            "ex_date", "payment_date", "effective_at", "cash_dividend",
            "stock_dividend_ratio", "rights_ratio", "rights_price", "known_at",
            "known_at_source", "raw_payload_hash",
        },
    }

    def __init__(self, snapshot_dir: str | Path) -> None:
        self.root = Path(snapshot_dir).resolve()
        self.manifest_path = self.root / "manifest.json"

    def load(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise SnapshotValidationError(f"缺少冻结快照清单: {self.manifest_path}")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotValidationError(f"快照清单无法读取: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SnapshotValidationError("manifest.json 必须是 JSON object")
        return manifest

    def validate(self) -> tuple[dict[str, Any], DataGateReport]:
        manifest = self.load()
        report = DataGateReport(snapshot_id=manifest.get("snapshot_id"))
        if manifest.get("manifest_version") not in {
            SNAPSHOT_MANIFEST_VERSION,
            "w1ngman_raw_snapshot_v1",
        }:
            report.add(
                "MANIFEST_VERSION_MISMATCH",
                f"需要 {SNAPSHOT_MANIFEST_VERSION}，实际 {manifest.get('manifest_version')!r}",
            )
        if not manifest.get("frozen", False):
            report.add("SNAPSHOT_NOT_FROZEN", "manifest.frozen 必须为 true")
        if not manifest.get("snapshot_id"):
            report.add("SNAPSHOT_ID_MISSING", "manifest.snapshot_id 不能为空")
        for field_name in ("data_version", "built_at", "sources"):
            if not manifest.get(field_name):
                report.add(
                    "MANIFEST_PROVENANCE_MISSING",
                    f"manifest.{field_name} 不能为空",
                )
        files = manifest.get("files")
        if not isinstance(files, dict):
            report.add("FILE_MANIFEST_MISSING", "manifest.files 必须是按表名索引的 object")
            return manifest, report.finish()

        missing_tables = sorted(self.required_tables - set(files))
        if missing_tables:
            report.add("REQUIRED_TABLE_MISSING", f"缺少表: {missing_tables}")

        for table_name, entry in files.items():
            if not isinstance(entry, dict):
                report.add("FILE_ENTRY_INVALID", f"{table_name} 的清单项不是 object")
                continue
            rel = entry.get("path")
            expected_hash = str(entry.get("sha256", "")).lower()
            if not rel or len(expected_hash) != 64:
                report.add("FILE_EVIDENCE_MISSING", f"{table_name} 缺 path/sha256")
                continue
            if "row_count" in entry and "rows" in entry:
                try:
                    if int(entry["row_count"]) != int(entry["rows"]):
                        report.add(
                            "ROW_COUNT_CONTRACT",
                            f"{table_name}: row_count={entry['row_count']} rows={entry['rows']}",
                        )
                        continue
                except (TypeError, ValueError):
                    report.add(
                        "ROW_COUNT_CONTRACT",
                        f"{table_name}: row_count/rows 无法比较",
                    )
                    continue
            try:
                path = _snapshot_file(self.root, str(rel))
            except ValueError:
                report.add("PATH_ESCAPES_SNAPSHOT", f"{table_name}: {rel}")
                continue
            if not path.is_file():
                report.add("FILE_NOT_FOUND", f"{table_name}: {path}")
                continue
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                report.add("FILE_HASH_MISMATCH", f"{table_name}: SHA-256不一致")
                continue
            required_columns = self.table_required_columns.get(table_name)
            actual_columns: set[str] | None = None
            if path.suffix.lower() == ".parquet":
                try:
                    import pyarrow.parquet as pq

                    actual_columns = set(pq.ParquetFile(path).schema.names)
                except Exception as exc:
                    report.add(
                        "PARQUET_SCHEMA_UNREADABLE",
                        f"{table_name}: {exc}",
                    )
                    continue
            if required_columns and actual_columns is not None:
                missing_columns = sorted(required_columns - actual_columns)
                if missing_columns:
                    report.add(
                        "TABLE_SCHEMA_MISSING",
                        f"{table_name}缺少字段: {missing_columns}",
                    )
                    continue
            if (
                manifest.get("schema_version") == "hs300_pit_v2"
                and actual_columns is not None
            ):
                from data_pipeline.hs300.lineage import LINEAGE_COLUMNS

                missing_lineage = [name for name in LINEAGE_COLUMNS if name not in actual_columns]
                if missing_lineage:
                    report.add(
                        "LINEAGE_FIELDS_MISSING",
                        f"{table_name}缺少血缘字段: {missing_lineage}",
                    )
                    continue
            expected_rows = entry.get("row_count", entry.get("rows"))
            if expected_rows is not None and path.suffix.lower() == ".parquet":
                try:
                    import pyarrow.parquet as pq

                    actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
                except Exception:
                    actual_rows = len(pd.read_parquet(path))
                if int(expected_rows) != actual_rows:
                    report.add(
                        "ROW_COUNT_MISMATCH",
                        f"{table_name}: 清单={expected_rows} 实际={actual_rows}",
                    )
                    continue
            report.checked_files += 1
        if "raw_request_manifest" in files:
            self._validate_raw_requests(manifest, report)
        return manifest, report.finish()

    def _validate_raw_requests(
        self, manifest: dict[str, Any], report: DataGateReport
    ) -> None:
        try:
            path = self.table_path(manifest, "raw_request_manifest")
            if path.suffix.lower() == ".parquet":
                rows = pd.read_parquet(path).to_dict(orient="records")
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload.get("requests", []) if isinstance(payload, dict) else []
        except Exception as exc:
            report.add("RAW_REQUEST_MANIFEST_INVALID", str(exc))
            return
        if not rows:
            report.add("RAW_REQUEST_EVIDENCE_EMPTY", "原始请求清单不能为空")
            return
        required = {
            "request_id", "method", "parameters", "requested_at", "completed_at",
            "response_path", "response_sha256", "row_count", "success",
        }
        for index, row in enumerate(rows):
            missing = sorted(required - set(row))
            if missing:
                report.add(
                    "RAW_REQUEST_FIELDS_MISSING",
                    f"请求#{index}缺少字段: {missing}",
                )
                continue
            if not bool(row.get("success")):
                # Failed calls are auditable evidence too; they do not need a
                # response payload, but cannot satisfy a required table alone.
                continue
            rel = row.get("response_path")
            expected = str(row.get("response_sha256", "")).lower()
            try:
                response = _snapshot_file(self.root, str(rel))
            except ValueError:
                report.add("RAW_RESPONSE_PATH_ESCAPE", f"请求#{index}: {rel}")
                continue
            if not response.is_file() or len(expected) != 64:
                report.add("RAW_RESPONSE_EVIDENCE_MISSING", f"请求#{index}: {rel}")
                continue
            if _sha256(response) != expected:
                report.add("RAW_RESPONSE_HASH_MISMATCH", f"请求#{index}: {rel}")

    def table_path(self, manifest: dict[str, Any], table_name: str) -> Path:
        entry = manifest["files"][table_name]
        return _snapshot_file(self.root, str(entry["path"]))


def _normalise_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return parsed.dt.normalize()


def _normalise_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def _require_columns(frame: pd.DataFrame, table: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SnapshotValidationError(f"{table} 缺少字段: {missing}")


def _drop_lineage(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [column for column in LINEAGE_COLUMNS if column in frame.columns]
    return frame.drop(columns=drop) if drop else frame


class HS300PanelDataManager:
    """Build a point-in-time long panel from a verified frozen snapshot.

    The panel is membership-led: every member row is retained even when its
    quote is missing. Industry intervals are joined by their historical
    effective dates; no forward/backward fill is used.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        *,
        expected_members: int = 300,
        index_code: str = "000300.SH",
        required_start: str | pd.Timestamp | None = RESEARCH_START.isoformat(),
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.expected_members = int(expected_members)
        self.index_code = index_code.upper()
        self.required_start = (
            pd.Timestamp(required_start).normalize() if required_start is not None else None
        )
        self.manifest: dict[str, Any] | None = None
        self.report: DataGateReport | None = None
        self.panel: pd.DataFrame | None = None
        self.daily_bars: pd.DataFrame | None = None
        self.trading_status: pd.DataFrame | None = None
        self.calendar: pd.DatetimeIndex | None = None

    def load(self, *, raise_on_gate_failure: bool = True) -> "HS300PanelDataManager":
        validator = SnapshotManifestValidator(self.snapshot_dir)
        try:
            manifest, report = validator.validate()
        except SnapshotValidationError as exc:
            manifest = {}
            report = DataGateReport(snapshot_id=None)
            report.add("SNAPSHOT_MANIFEST_INVALID", str(exc))
            report.finish()
        self.manifest = manifest
        self.report = report
        if not report.passed and raise_on_gate_failure:
            raise SnapshotValidationError(json.dumps(report.to_dict(), ensure_ascii=False))

        try:
            bars = pd.read_parquet(validator.table_path(manifest, "daily_bars"))
            members = pd.read_parquet(
                validator.table_path(manifest, "universe_membership")
            )
            industries = pd.read_parquet(
                validator.table_path(manifest, "industry_membership")
            )
            status = pd.read_parquet(validator.table_path(manifest, "trading_status"))
            calendar = pd.read_parquet(validator.table_path(manifest, "trading_calendar"))
            self._build(bars, members, industries, status, calendar)
        except SnapshotValidationError as exc:
            report.add("SCHEMA_OR_CAUSALITY_FAILURE", str(exc))
        except Exception as exc:
            report.add("PANEL_BUILD_FAILURE", f"{type(exc).__name__}: {exc}")

        report.finish()
        if not report.passed and raise_on_gate_failure:
            raise SnapshotValidationError(json.dumps(report.to_dict(), ensure_ascii=False))
        return self

    def _build(
        self,
        bars: pd.DataFrame,
        members: pd.DataFrame,
        industries: pd.DataFrame,
        status: pd.DataFrame,
        calendar: pd.DataFrame,
    ) -> None:
        assert self.report is not None
        bars = _drop_lineage(bars)
        members = _drop_lineage(members)
        industries = _drop_lineage(industries)
        status = _drop_lineage(status)
        calendar = _drop_lineage(calendar)
        _require_columns(
            bars,
            "daily_bars",
            [
                "date", "code", "open_raw", "high_raw", "low_raw", "close_raw",
                "volume", "amount", "open_tr", "close_tr", "available_at",
                "available_at_source", "has_quote", "quote_missing_reason",
                "revision_id", "source_row_hash",
            ],
        )
        _require_columns(
            members,
            "universe_membership",
            [
                "date", "index_code", "code", "is_member", "weight_pct",
                "effective_at", "known_at", "known_at_source", "membership_source",
            ],
        )
        _require_columns(
            industries,
            "industry_membership",
            ["code", "industry_code", "valid_from", "valid_to", "known_at"],
        )
        _require_columns(
            status,
            "trading_status",
            ["date", "code", "can_buy_open", "can_sell_open"],
        )
        _require_columns(
            calendar,
            "trading_calendar",
            ["date", "is_trading_day", "is_complete_session", "available_at"],
        )

        for frame in (bars, members, status, calendar):
            frame["date"] = _normalise_date(frame["date"])
        for frame in (bars, members, industries, status):
            frame["code"] = _normalise_code(frame["code"])
        members["effective_at"] = pd.to_datetime(
            members["effective_at"], errors="coerce", utc=True
        )
        members["known_at"] = pd.to_datetime(
            members["known_at"], errors="coerce", utc=True
        )
        industries["valid_from"] = _normalise_date(industries["valid_from"])
        industries["valid_to"] = _normalise_date(industries["valid_to"])
        industries["known_at"] = pd.to_datetime(
            industries["known_at"], errors="coerce", utc=True
        )
        bars["available_at"] = pd.to_datetime(
            bars["available_at"], errors="coerce", utc=True
        )

        if bars.duplicated(["date", "code"]).any():
            raise SnapshotValidationError("daily_bars 存在 date+code 重复；禁止 keep-last")
        if members.duplicated(["date", "index_code", "code"]).any():
            raise SnapshotValidationError("universe_membership 存在重复成员证据")
        if status.duplicated(["date", "code"]).any():
            raise SnapshotValidationError("trading_status 存在 date+code 重复")

        complete_dates = pd.DatetimeIndex(
            calendar.loc[
                calendar["is_trading_day"].fillna(False).astype(bool)
                & calendar["is_complete_session"].fillna(False).astype(bool),
                "date",
            ].dropna().drop_duplicates().sort_values()
        )
        if complete_dates.empty:
            raise SnapshotValidationError("交易日历没有有效交易日")
        holdout_dates = self._holdout_dates(complete_dates)
        if len(holdout_dates):
            leaked_prices = bars["date"].isin(holdout_dates).any()
            leaked_members = members["date"].isin(holdout_dates).any()
            sealed_lock = self.snapshot_dir / "sealed" / "holdout.lock.json"
            if leaked_prices or leaked_members or not sealed_lock.is_file():
                self.report.add(
                    "HOLDOUT_PRICES_NOT_SEALED",
                    "standardized 层仍含 Holdout 日期的价格或成员行，或缺少 sealed/holdout.lock.json",
                )
            complete_dates = complete_dates.difference(holdout_dates)
        if complete_dates.empty:
            raise SnapshotValidationError("剔除 Holdout 后没有 Development 交易日")
        if self.required_start is not None:
            # Allow the first few New Year holidays but do not silently accept
            # a supplier whose series starts years after the frozen request.
            latest_allowed = self.required_start + pd.Timedelta(days=10)
            if complete_dates[0] > latest_allowed:
                self.report.add(
                    "REQUESTED_START_NOT_COVERED",
                    f"要求起点{self.required_start.date()}，首个交易日为"
                    f"{complete_dates[0].date()}",
                )
        member = members.loc[
            (members["index_code"].astype("string").str.upper() == self.index_code)
            & members["is_member"].fillna(False).astype(bool)
        ].copy()
        member = member[member["date"].isin(complete_dates)]
        effective_date = member["effective_at"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
        if member["effective_at"].isna().any() or (effective_date > member["date"]).any():
            self.report.add(
                "MEMBERSHIP_EFFECTIVE_TIME_INVALID",
                "成员生效时间缺失或晚于被使用的交易日",
            )
        counts = member.groupby("date", observed=True)["code"].nunique()
        bad_counts = counts[counts != self.expected_members]
        if not bad_counts.empty:
            self.report.add(
                "MEMBER_COUNT_NOT_EXACT",
                f"{len(bad_counts)}个交易日成员数不等于{self.expected_members}",
                row_count=len(bad_counts),
            )
        missing_membership_dates = complete_dates.difference(counts.index)
        if len(missing_membership_dates):
            self.report.add(
                "MEMBERSHIP_DATE_MISSING",
                f"{len(missing_membership_dates)}个交易日没有成员数据",
                row_count=len(missing_membership_dates),
            )
        member["weight_pct"] = pd.to_numeric(member["weight_pct"], errors="coerce")
        weight_sums = member.groupby("date", observed=True)["weight_pct"].sum(min_count=1)
        bad_weight = weight_sums[~np.isclose(weight_sums, 100.0, rtol=0.0, atol=0.25)]
        if not bad_weight.empty:
            self.report.add(
                "MEMBER_WEIGHT_SUM_INVALID",
                "每日沪深300权重和必须接近100%（容差0.25个百分点）",
                row_count=len(bad_weight),
            )

        quote = bars.copy()
        numeric = [
            "open_raw", "high_raw", "low_raw", "close_raw", "volume", "amount",
            "open_tr", "close_tr",
        ]
        for column in numeric:
            quote[column] = pd.to_numeric(quote[column], errors="coerce")
        has_quote = quote["has_quote"].fillna(False).astype(bool)
        quoted = quote.loc[has_quote]
        invalid_ohlc = (
            quoted[["open_raw", "high_raw", "low_raw", "close_raw"]].le(0).any(axis=1)
            | (quoted["low_raw"] > quoted[["open_raw", "close_raw"]].min(axis=1))
            | (quoted["high_raw"] < quoted[["open_raw", "close_raw"]].max(axis=1))
            | quoted[["volume", "amount"]].lt(0).any(axis=1)
            | quoted[["open_tr", "close_tr"]].le(0).any(axis=1)
        )
        if invalid_ohlc.any():
            self.report.add(
                "INVALID_OHLCV",
                "行情违反OHLCV基本约束",
                row_count=int(invalid_ohlc.sum()),
            )
        if quote.loc[~has_quote, numeric].notna().any(axis=None):
            self.report.add(
                "SYNTHETIC_MISSING_QUOTE_VALUES",
                "has_quote=false 的记录包含数值行情，疑似填充",
            )
        missing_reason = quote.loc[~has_quote, "quote_missing_reason"].astype("string").str.strip()
        if missing_reason.isna().any() or missing_reason.eq("").any():
            self.report.add(
                "QUOTE_MISSING_REASON_ABSENT",
                "has_quote=false 的记录必须说明停牌/未上市/退市/数据异常原因",
            )
        if quote.loc[has_quote, "available_at"].isna().any():
            self.report.add("AVAILABLE_AT_MISSING", "有效行情缺少 available_at")
        valid_hash = quote["source_row_hash"].astype("string").str.fullmatch(
            r"[0-9a-fA-F]{64}", na=False
        )
        if not valid_hash.all():
            self.report.add(
                "SOURCE_ROW_HASH_INVALID",
                "daily_bars.source_row_hash 必须是64位SHA-256",
                row_count=int((~valid_hash).sum()),
            )
        non_trading_quotes = quote.loc[has_quote & ~quote["date"].isin(complete_dates)]
        if not non_trading_quotes.empty:
            self.report.add(
                "QUOTE_ON_NON_TRADING_DATE",
                "完整交易日历之外出现有效日线",
                row_count=len(non_trading_quotes),
            )

        panel = member.merge(
            quote, on=["date", "code"], how="left", validate="one_to_one", suffixes=("", "_quote")
        )
        panel = panel.merge(
            status,
            on=["date", "code"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_status"),
        )
        if len(panel) != len(member):
            raise SnapshotValidationError("行情/状态连接改变了沪深300成员行数")
        if panel["has_quote"].isna().any():
            self.report.add(
                "MEMBER_QUOTE_EVIDENCE_MISSING",
                "成员行既无有效行情也无带原因的缺失行情记录",
                row_count=int(panel["has_quote"].isna().sum()),
            )

        panel = self._join_industry_intervals(panel, industries)
        unmapped = int(panel["industry_code"].isna().sum())
        coverage = 1.0 - (unmapped / len(panel)) if len(panel) else 0.0
        self.report.metadata["industry_coverage"] = coverage
        if coverage < INDUSTRY_COVERAGE_MIN:
            self.report.add(
                "INDUSTRY_COVERAGE_BELOW_THRESHOLD",
                f"行业覆盖率 {coverage:.4f} 低于正式门槛 {INDUSTRY_COVERAGE_MIN}",
                row_count=unmapped,
            )
        elif unmapped:
            self.report.add(
                "INDUSTRY_UNMAPPED",
                "存在无法按历史有效区间匹配的行业；成员行保留，行业为空",
                severity="WARNING",
                row_count=unmapped,
            )
        if panel[["can_buy_open", "can_sell_open"]].isna().any(axis=None):
            self.report.add("TRADING_STATUS_MISSING", "成员记录缺少方向性交易状态")

        self.panel = panel.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        self.daily_bars = quote.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        self.trading_status = status.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        self.calendar = pd.DatetimeIndex(complete_dates)
        self.report.trading_dates = len(complete_dates)
        self.report.panel_rows = len(panel)
        self.report.metadata.update(
            {
                "schema_version": PANEL_SCHEMA_VERSION,
                "index_code": self.index_code,
                "expected_members": self.expected_members,
                "first_date": str(complete_dates[0].date()),
                "last_date": str(complete_dates[-1].date()),
            }
        )

    def to_tensors(self) -> HS300PanelTensors:
        if self.panel is None or self.calendar is None:
            raise RuntimeError("Call load() first")
        return pack_panel_tensors(
            self.panel,
            self.calendar,
            expected_members=self.expected_members,
        )

    def save_tensors(self, path: str | Path | None = None) -> dict[str, Any]:
        tensors = self.to_tensors()
        target = Path(path) if path is not None else self.snapshot_dir / "panel" / "hs300_tensors.npz"
        meta = save_panel_tensors(tensors, target)
        manifest_path = target.with_name("tensor_manifest.json")
        manifest_path.write_text(
            json.dumps(meta | {"expected_members": self.expected_members}, indent=2),
            encoding="utf-8",
        )
        return meta

    def load_labels(self, *, development_only: bool = True) -> pd.DataFrame:
        if not development_only:
            raise PermissionError("Holdout labels are not written and must not be requested")
        path = self.snapshot_dir / "labels" / "development_labels.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 Development 标签: {path}")
        frame = pd.read_parquet(path)
        lock_path = self.snapshot_dir / "splits" / "holdout.lock.json"
        if lock_path.is_file():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            holdout_start = pd.Timestamp(lock["holdout_start"]).normalize()
            signal = pd.to_datetime(frame["signal_date"]).dt.normalize()
            if (signal >= holdout_start).any():
                raise PermissionError("development labels contain holdout dates")
        return frame

    @staticmethod
    def _join_industry_intervals(
        panel: pd.DataFrame, industries: pd.DataFrame
    ) -> pd.DataFrame:
        intervals = industries.copy().rename(
            columns={
                "known_at": "industry_known_at",
                "known_at_source": "industry_known_at_source",
                "effective_at": "industry_effective_at",
            }
        )
        intervals = intervals.sort_values(
            ["code", "valid_from", "industry_known_at"], kind="stable"
        )
        if intervals.duplicated(["code", "valid_from"]).any():
            raise SnapshotValidationError("industry_membership 同一股票/生效日存在冲突")

        rows: list[pd.DataFrame] = []
        industry_columns = [
            col for col in intervals.columns if col not in {"code"}
        ]
        for code, stock_rows in panel.groupby("code", sort=False, observed=True):
            history = intervals.loc[intervals["code"] == code].copy()
            stock_rows = stock_rows.sort_values("date", kind="stable")
            if history.empty:
                empty = stock_rows.copy()
                for column in industry_columns:
                    empty[column] = pd.NA
                rows.append(empty)
                continue
            joined = pd.merge_asof(
                stock_rows,
                history.drop(columns="code").sort_values("valid_from", kind="stable"),
                left_on="date",
                right_on="valid_from",
                direction="backward",
                allow_exact_matches=True,
            )
            signal_date = joined["date"]
            known = pd.to_datetime(joined["industry_known_at"], utc=True, errors="coerce")
            if getattr(known.dt, "tz", None) is not None:
                known_date = (
                    known.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
                )
            else:
                known_date = known.dt.normalize()
            invalid = (
                joined["valid_from"].isna()
                | (joined["valid_to"].notna() & (joined["date"] > joined["valid_to"]))
                | (joined["industry_code"].notna() & known.isna())
                | (known_date > signal_date)
            )
            joined.loc[invalid, industry_columns] = pd.NA
            rows.append(joined)
        result = pd.concat(rows, ignore_index=True)
        if len(result) != len(panel):
            raise SnapshotValidationError("行业连接改变了成员行数")
        return result

    def development_panel(self, development_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        if self.panel is None:
            raise RuntimeError("Call load() first")
        allowed = pd.DatetimeIndex(development_dates)
        return self.panel[self.panel["date"].isin(allowed)].copy()

    def bars_for_dates(self, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        if self.daily_bars is None:
            raise RuntimeError("Call load() first")
        allowed = pd.DatetimeIndex(dates)
        return self.daily_bars[self.daily_bars["date"].isin(allowed)].copy()

    def _holdout_dates(self, complete_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        dates_path = self.snapshot_dir / "splits" / "holdout_dates.parquet"
        if dates_path.is_file():
            frame = pd.read_parquet(dates_path)
            return pd.DatetimeIndex(
                pd.to_datetime(frame["date"]).dt.normalize().unique()
            ).sort_values()
        for lock_path in (
            self.snapshot_dir / "splits" / "holdout.lock.json",
            self.snapshot_dir / "sealed" / "holdout.lock.json",
        ):
            if lock_path.is_file():
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                start = pd.Timestamp(lock["holdout_start"]).normalize()
                return complete_dates[complete_dates >= start]
        return pd.DatetimeIndex([])
