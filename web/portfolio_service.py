"""Validated score-only imports and immutable local offline portfolio records.

The source bundle is read once through an explicit manifest path. This module
does not discover research data, load executable models, or open Holdout.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATUS = "OFFLINE_DEMO_COMPLETED_NOT_VALIDATED"
SCORE_COLUMNS = ["date", "code", "score", "score_available_at", "fold_id", "is_member", "signal_valid"]
FILES = ("scores.parquet", "selected_features.json", "formulas.json", "model_evidence.json")
FALSE_FLAGS = ("research_go", "holdout_read", "sota_promoted", "production_allowed", "whitelist_applied")
DEFAULT_BUNDLES = (
    "experiments/offline_demo_optimizer_300_20260917/manifest.json",
    "experiments/offline_demo_portfolio_300_20260916/manifest.json",
    "experiments/offline_demo_factors_20260915_r1/manifest.json",
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_BANNED = {"weight", "weights", "target_weight", "target_weights", "allocation", "allocations",
           "recommendation", "recommendations", "order", "orders", "trade_order", "trade_orders"}
_BOUNDARY = pd.Timestamp("2024-08-26T00:00:00+08:00")


class PortfolioError(ValueError):
    """A deliberately path-free public error."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise PortfolioError(message)


def _plain_json(raw: bytes) -> Any:
    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite JSON")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
        _json(value)
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise PortfolioError("评分包 JSON 无效或含非有限数值") from exc
    return value


def _check_score_only(value: Any, depth: int = 0) -> None:
    _check(depth < 80, "评分包 JSON 嵌套过深")
    if isinstance(value, dict):
        for key, child in value.items():
            _check(key.lower() not in _BANNED, "评分包不得包含交易或目标权重字段")
            _check_score_only(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_score_only(child, depth + 1)


class PortfolioService:
    def __init__(self, workspace: str | Path = ROOT, store_dir: str | Path | None = None,
                 default_bundles: tuple[str, ...] = DEFAULT_BUNDLES):
        self.workspace = Path(workspace).resolve()
        self.store_dir = Path(store_dir).resolve() if store_dir else self.workspace / "portfolio_demo_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "portfolio.sqlite3"
        self.default_import_warnings: list[str] = []
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS bundles (
                    bundle_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY, bundle_id TEXT NOT NULL,
                    signal_date TEXT NOT NULL, fold_id TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(bundle_id, signal_date, fold_id));
                CREATE TABLE IF NOT EXISTS whitelists (
                    whitelist_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(whitelist_id, version));
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS forward_events (
                    event_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS validation_reports (
                    validation_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
        # An explicit, fixed registration list; there is intentionally no glob.
        for relative_path in default_bundles:
            if (self.workspace / relative_path).is_file():
                try:
                    self.import_bundle(relative_path)
                except PortfolioError:
                    self.default_import_warnings.append("DEFAULT_BUNDLE_REJECTED")
        evidence = "experiments/offline_demo_optimizer_inputs_20260917/manifest.json"
        if default_bundles == DEFAULT_BUNDLES and (self.workspace / evidence).is_file():
            try:
                self.import_analysis(evidence)
            except (PortfolioError, ValueError):
                self.default_import_warnings.append("DEFAULT_ANALYSIS_REJECTED")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _safe_path(self, supplied: str | Path, *, manifest: bool = False) -> Path:
        candidate = Path(supplied)
        _check(not any("holdout" in part.lower() or "sealed" in part.lower() for part in candidate.parts),
               "禁止读取 Holdout 或 sealed 路径")
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        # Reject lexical escapes/UNC paths before resolving or stat-ing them.
        candidate = Path(os.path.abspath(candidate))
        _check(candidate.is_relative_to(self.workspace), "仅允许工作区内的评分包")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PortfolioError("指定的评分包文件不存在或不可访问") from exc
        _check(resolved.is_relative_to(self.workspace), "仅允许工作区内的评分包")
        _check(not any("holdout" in part.lower() or "sealed" in part.lower() for part in resolved.parts),
               "禁止读取 Holdout 或 sealed 路径")
        _check(resolved.is_file(), "评分包路径必须是文件")
        if manifest:
            _check(candidate.name == "manifest.json" and resolved.name == "manifest.json",
                   "必须显式指定 manifest.json")
            _check(resolved.parent.name.startswith("offline_demo_") and resolved.parent.name != "offline_demo_",
                   "评分包必须位于明确命名的 offline_demo_ 目录")
        return resolved

    def _read(self, path: Path, max_bytes: int) -> bytes:
        safe = self._safe_path(path)
        try:
            _check(safe.stat().st_size <= max_bytes, "评分包文件过大")
            with safe.open("rb") as stream:
                raw = stream.read(max_bytes + 1)
            _check(len(raw) <= max_bytes, "评分包文件过大")
            return raw
        except OSError as exc:
            raise PortfolioError("评分包文件不可读取") from exc

    def import_bundle(self, manifest_path: str) -> dict[str, Any]:
        path = self._safe_path(manifest_path, manifest=True)
        raw_manifest = self._read(path, 16 * 1024 * 1024)
        manifest = _plain_json(raw_manifest)
        _check(isinstance(manifest, dict), "manifest 必须为 JSON 对象")
        _check_score_only(manifest)
        _check(manifest.get("schema_version") == "1.0"
               and manifest.get("artifact_kind") == "OFFLINE_DEMO_SCORE_BUNDLE", "评分包类型或版本不受支持")
        _check(manifest.get("status") == STATUS, "评分包不得声明为已验证策略")
        _check(manifest.get("data_role") in ("synthetic", "development"), "仅允许 synthetic 或 development 数据")
        for key in FALSE_FLAGS:
            _check(manifest.get(key) is False, "评分包必须保留全部非生产及隔离标记")
        _check(manifest.get("formal_sota_written", False) is False, "演示评分包不能写入正式 SOTA")
        _check(isinstance(manifest.get("run_id"), str) and bool(_ID.fullmatch(manifest["run_id"])),
               "评分包运行标识无效")
        fingerprints = manifest.get("fingerprints")
        _check(isinstance(fingerprints, dict) and set(fingerprints) == {"model", "formula", "fold", "data"},
               "缺少模型、公式、折或数据指纹")
        _check(all(isinstance(v, str) and _HASH.fullmatch(v) and len(set(v)) > 1 for v in fingerprints.values()),
               "指纹必须为非占位的小写 SHA-256")
        _check(manifest.get("score_columns") == SCORE_COLUMNS, "七列评分 schema 与合同不符")
        _check(isinstance(manifest.get("warnings", []), list), "评分包警告字段必须为列表")
        files = manifest.get("files")
        _check(isinstance(files, dict) and set(files) == set(FILES), "评分包固定文件清单与合同不符")
        contents = {}
        for name in FILES:
            entry = files[name]
            _check(isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
                   and bool(_HASH.fullmatch(entry["sha256"])), "文件 SHA-256 缺失或无效")
            # Manifest-supplied paths are never used, even if an extra path key exists.
            target = self._safe_path(path.parent / name)
            _check(target.parent == path.parent and target.name == name, "评分包文件不可指向其他目录")
            raw = self._read(target, 64 * 1024 * 1024 if name.endswith(".parquet") else 16 * 1024 * 1024)
            _check(sha256(raw).hexdigest() == entry["sha256"], "评分包文件 SHA-256 校验失败")
            contents[name] = raw
        for name, key in (("selected_features.json", "selected_features"), ("formulas.json", "formulas"),
                          ("model_evidence.json", "model_evidence")):
            evidence = _plain_json(contents[name])
            _check_score_only(evidence)
            _check(evidence == manifest.get(key), "评分包证据文件与 manifest 不一致")
        selected = manifest.get("selected_features")
        _check(isinstance(selected, list) and bool(selected)
               and all(isinstance(v, str) and v.strip() for v in selected)
               and len(set(selected)) == len(selected), "实际拟合特征列表必须非空且唯一")
        _check(isinstance(manifest.get("formulas"), list), "公式证据必须是列表")
        _check(isinstance(manifest.get("model_evidence"), dict) and bool(manifest["model_evidence"]),
               "缺少实际模型证据")
        try:
            frame = pd.read_parquet(BytesIO(contents["scores.parquet"]))
        except Exception as exc:
            raise PortfolioError("评分 Parquet 无法解析") from exc
        frame = self._validate_scores(frame, manifest)
        bundle_id = "bundle_" + sha256(raw_manifest).hexdigest()[:24]
        warnings = ["OFFLINE_DEMO_NOT_A_VALIDATED_STRATEGY", "HISTORICAL_REPLAY_NOT_LIVE_PREDICTION"]
        if manifest["data_role"] == "synthetic":
            warnings.append("SYNTHETIC_DATA_NOT_MARKET_EVIDENCE")
        for warning in manifest.get("warnings", []):
            if isinstance(warning, str) and re.fullmatch(r"[A-Z0-9_]{1,120}", warning) and warning not in warnings:
                warnings.append(warning)
        snapshots = []
        periods = []
        for (date, fold_id), group in frame.groupby(["date", "fold_id"], sort=True):
            members = group[group["is_member"]].copy()
            member_count = len(members)
            if manifest["data_role"] == "development":
                _check(member_count == 300, "Development 每个日期和折必须完整包含300个成员")
            _check(member_count > 0, "评分日期和折缺少成员行")
            members = members.sort_values(["signal_valid", "score", "code"], ascending=[False, False, True],
                                          na_position="last", kind="stable")
            rows = []
            valid_rank = 0
            for row in members.itertuples(index=False):
                valid = bool(row.signal_valid)
                valid_rank += int(valid)
                rows.append({"code": row.code, "score": float(row.score) if valid else None,
                             "rank": valid_rank if valid else None, "signal_valid": valid,
                             "score_available_at": row.score_available_at.isoformat(),
                             "reason_codes": [] if valid else ["INVALID_OR_UNAVAILABLE_SCORE"]})
            day = date.date().isoformat()
            available_at = members["score_available_at"].max().isoformat()
            local_warnings = list(warnings)
            if member_count != 300:
                local_warnings.append("SYNTHETIC_UNIVERSE_NOT_300")
            if valid_rank != member_count:
                local_warnings.append("INVALID_MEMBER_SCORES_PRESERVED")
            snapshot_id = "snapshot_" + sha256(_json([bundle_id, day, fold_id]).encode()).hexdigest()[:24]
            snapshot = {"snapshot_id": snapshot_id, "bundle_id": bundle_id, "date": day, "fold_id": fold_id,
                        "data_role": manifest["data_role"], "expected_member_count": 300,
                        "member_count": member_count, "valid_count": valid_rank,
                        "is_complete_universe": member_count == 300, "rows": rows,
                        "available_at": available_at, "warnings": local_warnings, "status": STATUS,
                        "production_allowed": False, "research_go": False, "holdout_read": False,
                        "sota_promoted": False, "whitelist_applied": False, "fingerprints": fingerprints}
            snapshots.append(snapshot)
            periods.append({"date": day, "fold_id": fold_id, "member_count": member_count,
                            "valid_count": valid_rank, "available_at": available_at})
        metadata = {"bundle_id": bundle_id, "label": path.parent.name, "run_id": manifest["run_id"],
                    "data_role": manifest["data_role"], "status": STATUS, "production_allowed": False,
                    "research_go": False, "holdout_read": False, "sota_promoted": False, "whitelist_applied": False,
                    "periods": periods, "warnings": warnings, "fingerprints": fingerprints,
                    "score_rows": len(frame), "member_rows": int(frame["is_member"].sum()),
                    "valid_member_rows": int(frame["signal_valid"].sum()),
                    "signal_output": manifest.get("signal_output") if manifest.get("signal_output")
                    in ("model_prediction", "equal_weight") else "unspecified",
                    "selected_feature_count": len(selected), "formula_count": len(manifest["formulas"]),
                    "score_semantics": "RAW_MODEL_SCORE_NOT_RETURN_PROBABILITY_OR_ACCOUNT_WEIGHT",
                    "imported_at": _now(), "schema_version": "offline_portfolio_v1"}
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO bundles VALUES (?, ?)", (bundle_id, _json(metadata)))
            for snapshot in snapshots:
                db.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?, ?, ?)",
                           (snapshot["snapshot_id"], bundle_id, snapshot["date"], snapshot["fold_id"], _json(snapshot)))
        return self.get_bundle(bundle_id)

    def _validate_scores(self, frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
        _check(list(frame.columns) == SCORE_COLUMNS and not frame.columns.duplicated().any() and not frame.empty,
               "Parquet 必须仅含七列评分 schema 且非空")
        for count in (manifest.get("expected_score_rows"), manifest.get("score_rows"),
                      manifest["files"]["scores.parquet"].get("rows")):
            _check(type(count) is int and count == len(frame), "评分行数与冻结预测网格不一致")
        for key in ("code", "fold_id"):
            pattern = _CODE if key == "code" else _ID
            _check(frame[key].map(lambda v: isinstance(v, str) and bool(pattern.fullmatch(v))).all(),
                   "股票代码与折标识必须为有效字符串")
        for key in ("is_member", "signal_valid"):
            _check(frame[key].map(lambda v: isinstance(v, (bool, np.bool_))).all(), "成员与有效性标记必须为布尔值")
            frame[key] = frame[key].astype(bool)
        dates, times = [], []
        try:
            for day_value, time_value in zip(frame["date"], frame["score_available_at"]):
                day, stamp = pd.Timestamp(day_value), pd.Timestamp(time_value)
                _check(not pd.isna(day) and day.tzinfo is None and day == day.normalize(), "评分日期必须是无时区完整日期")
                _check(not pd.isna(stamp) and stamp.tzinfo is not None, "评分可用时间必须显式带时区")
                local = stamp.tz_convert("Asia/Shanghai")
                _check(local.date() == day.date(), "评分可用时间与上海信号日期不一致")
                _check(local < _BOUNDARY, "评分日期不能进入固定 Holdout 边界")
                dates.append(day)
                times.append(stamp.tz_convert("UTC"))
        except (ValueError, TypeError, OverflowError) as exc:
            if isinstance(exc, PortfolioError):
                raise
            raise PortfolioError("评分日期或可用时间无效") from exc
        frame["date"], frame["score_available_at"] = pd.DatetimeIndex(dates), pd.DatetimeIndex(times)
        _check(not frame.duplicated(["date", "code", "fold_id"]).any(), "日期、股票、折必须唯一")
        _check(pd.api.types.is_numeric_dtype(frame["score"]) and not pd.api.types.is_bool_dtype(frame["score"]),
               "评分必须为数值或缺失值")
        numeric = frame["score"].astype(float)
        _check(not np.isinf(numeric.to_numpy()).any(), "评分不能是无穷值")
        expected = numeric.notna() & frame["is_member"]
        _check(frame["signal_valid"].equals(expected), "有效性标记与成员有限评分不一致")
        _check(not (numeric.notna() & ~frame["is_member"]).any(), "非成员评分必须为空")
        frame["score"] = numeric
        member_count, valid_count = int(frame["is_member"].sum()), int(frame["signal_valid"].sum())
        _check(type(manifest.get("member_rows")) is int and manifest["member_rows"] == member_count
               and type(manifest.get("valid_member_rows")) is int and manifest["valid_member_rows"] == valid_count,
               "成员覆盖数量与 manifest 不一致")
        coverage = manifest.get("member_coverage")
        _check(member_count > 0 and isinstance(coverage, (int, float)) and not isinstance(coverage, bool)
               and math.isclose(coverage, valid_count / member_count, rel_tol=1e-12, abs_tol=1e-12),
               "成员覆盖率与 manifest 不一致")
        return frame

    def list_bundles(self) -> dict:
        with self._db() as db:
            rows = db.execute("SELECT payload FROM bundles ORDER BY rowid").fetchall()
        bundles = [json.loads(row[0]) for row in rows]
        bundles.sort(key=lambda b: ("optimizer_300_20260917" not in b["label"], b["imported_at"]))
        return {"bundles": [{key: bundle[key] for key in ("bundle_id", "label", "data_role", "status")}
                            for bundle in bundles], "warnings": self.default_import_warnings,
                "production_allowed": False}

    def _get(self, table: str, key: str, value: str) -> dict:
        # Table and column identifiers are internal constants, never request data.
        with self._db() as db:
            row = db.execute(f"SELECT payload FROM {table} WHERE {key}=?", (value,)).fetchone()
        if row is None:
            raise PortfolioError("记录不存在", 404)
        return json.loads(row[0])

    def get_bundle(self, bundle_id: str) -> dict:
        return self._get("bundles", "bundle_id", bundle_id)

    def get_snapshot(self, snapshot_id: str) -> dict:
        return self._get("snapshots", "snapshot_id", snapshot_id)

    def get_scores(self, bundle_id: str, date: str, fold_id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT payload FROM snapshots WHERE bundle_id=? AND signal_date=? AND fold_id=?",
                             (bundle_id, date, fold_id)).fetchone()
        if row is None:
            raise PortfolioError("该评分包中不存在指定日期和折", 404)
        return json.loads(row[0])

    def save_whitelist(self, snapshot_id: str, name: str, codes: list[str],
                       whitelist_id: str | None = None, expected_version: int | None = None) -> dict:
        snapshot = self.get_snapshot(snapshot_id)
        _check(isinstance(name, str) and 0 < len(name.strip()) <= 100, "白名单名称须为1至100个字符")
        _check(isinstance(codes, list) and len(codes) <= 300
               and all(isinstance(code, str) for code in codes) and len(set(codes)) == len(codes),
               "白名单代码必须唯一且不超过300只")
        members = {row["code"] for row in snapshot["rows"]}
        _check(set(codes) <= members, "白名单包含所见快照之外的股票")
        _check(expected_version is None or type(expected_version) is int and expected_version >= 1,
               "白名单版本无效")
        if whitelist_id is None:
            _check(expected_version is None, "创建白名单不能指定旧版本")
            whitelist_id = "whitelist_" + uuid.uuid4().hex
            version = 1
        else:
            _check(isinstance(whitelist_id, str) and bool(_ID.fullmatch(whitelist_id)), "白名单ID无效")
            _check(expected_version is not None, "更新白名单必须指定所见版本")
            version = expected_version + 1
        payload = {"whitelist_id": whitelist_id, "version": version, "name": name.strip(),
                   "codes": sorted(codes), "snapshot_id": snapshot_id, "bundle_id": snapshot["bundle_id"],
                   "date": snapshot["date"], "fold_id": snapshot["fold_id"], "data_role": snapshot["data_role"],
                   "created_at": _now(), "status": STATUS, "production_allowed": False}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT MAX(version) FROM whitelists WHERE whitelist_id=?", (whitelist_id,)).fetchone()[0]
            if expected_version is not None and current is None:
                raise PortfolioError("待更新的白名单不存在", 404)
            if expected_version is not None and current != expected_version:
                raise PortfolioError("白名单已被更新，请重新读取最新版本", 409)
            db.execute("INSERT INTO whitelists VALUES (?, ?, ?)", (whitelist_id, version, _json(payload)))
        return payload

    def list_whitelists(self) -> dict:
        with self._db() as db:
            rows = db.execute("""SELECT w.payload FROM whitelists w JOIN
                (SELECT whitelist_id, MAX(version) v FROM whitelists GROUP BY whitelist_id) latest
                ON w.whitelist_id=latest.whitelist_id AND w.version=latest.v ORDER BY w.rowid DESC""").fetchall()
        return {"whitelists": [json.loads(row[0]) for row in rows], "production_allowed": False}

    def get_whitelist(self, whitelist_id: str, version: int | None = None) -> dict:
        with self._db() as db:
            if version is None:
                row = db.execute("SELECT payload FROM whitelists WHERE whitelist_id=? ORDER BY version DESC LIMIT 1",
                                 (whitelist_id,)).fetchone()
            else:
                row = db.execute("SELECT payload FROM whitelists WHERE whitelist_id=? AND version=?",
                                 (whitelist_id, version)).fetchone()
        if row is None:
            raise PortfolioError("白名单或指定版本不存在", 404)
        return json.loads(row[0])

    def create_plan(self, request: dict) -> dict:
        from strategy_manager.offline_allocation import build_plan
        from strategy_manager.portfolio_models import AllocationError

        whitelist = self.get_whitelist(request["whitelist_id"], request["whitelist_version"])
        snapshot = self.get_snapshot(whitelist["snapshot_id"])
        analysis = self.find_analysis(snapshot["snapshot_id"], request.get("analysis_id"))
        try:
            result = build_plan(snapshot, whitelist, request, analysis)
        except AllocationError:
            raise
        except ValueError as exc:
            # Allocator errors must not leak paths or arbitrary request content.
            raise PortfolioError("配置输入无效，请检查持仓、预算和执行条件", 422) from exc
        result = {**result, "plan_id": "plan_" + uuid.uuid4().hex, "created_at": _now(),
                  "snapshot_id": snapshot["snapshot_id"], "bundle_id": snapshot["bundle_id"],
                  "whitelist_id": whitelist["whitelist_id"], "whitelist_version": whitelist["version"],
                  "data_role": snapshot["data_role"], "production_allowed": False,
                  "research_go": False, "holdout_read": False, "sota_promoted": False, "whitelist_applied": True,
                  "request": request}
        try:
            serialized = _json(result)
        except (ValueError, TypeError) as exc:
            raise PortfolioError("配置结果无效，未保存方案", 500) from exc
        with self._db() as db:
            db.execute("INSERT INTO plans VALUES (?, ?)", (result["plan_id"], serialized))
        return result

    def get_plan(self, plan_id: str) -> dict:
        return self._get("plans", "plan_id", plan_id)

    def import_analysis(self, manifest_path: str, snapshot_id: str | None = None) -> dict:
        from strategy_manager.portfolio_models import prepare_inputs
        path = self._safe_path(manifest_path, manifest=True)
        manifest = _plain_json(self._read(path, 1024 * 1024))
        _check(isinstance(manifest, dict) and manifest.get("artifact_kind") == "OFFLINE_PORTFOLIO_RESEARCH_INPUTS",
               "需要独立的配置证据 manifest")
        _check(manifest.get("production_allowed") is False and manifest.get("holdout_read") is False,
               "配置证据不得声明生产或Holdout访问")
        bundle = self.import_bundle(manifest.get("score_manifest", ""))
        snapshot = self.get_scores(bundle["bundle_id"], manifest.get("date"), manifest.get("fold_id"))
        _check(snapshot_id is None or snapshot_id == snapshot["snapshot_id"], "配置证据与所选快照不一致")
        raw = self._read(path.parent / "research_inputs.json", 80 * 1024 * 1024)
        _check(sha256(raw).hexdigest() == manifest.get("sha256"), "配置证据文件哈希不匹配")
        payload = _plain_json(raw)
        _check(isinstance(payload, dict), "配置证据必须为对象")
        score_manifest = _plain_json(self._read(self._safe_path(manifest["score_manifest"], manifest=True), 16*1024*1024))
        frozen_horizon = score_manifest["model_evidence"].get("horizon")
        _check(type(frozen_horizon) is int and frozen_horizon == payload.get("horizon"),
               "配置周期必须与评分包模型证据中冻结的horizon一致")
        analysis = prepare_inputs(payload, snapshot)
        analysis["imported_at"] = _now()
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO analyses VALUES (?, ?, ?)",
                       (analysis["analysis_id"], snapshot["snapshot_id"], _json(analysis)))
        return self.analysis_summary(self.find_analysis(snapshot["snapshot_id"], analysis["analysis_id"]))

    def find_analysis(self, snapshot_id: str, analysis_id: str | None = None) -> dict | None:
        with self._db() as db:
            if analysis_id:
                row = db.execute("SELECT payload FROM analyses WHERE snapshot_id=? AND analysis_id=?",
                                 (snapshot_id, analysis_id)).fetchone()
                if row is None:
                    raise PortfolioError("指定配置证据不存在或与快照不匹配", 422)
            else:
                row = db.execute("SELECT payload FROM analyses WHERE snapshot_id=? ORDER BY rowid DESC LIMIT 1",
                                 (snapshot_id,)).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def analysis_summary(analysis: dict | None) -> dict:
        if analysis is None:
            return {"available": False, "production_allowed": False,
                    "message": "当前快照缺少成熟OOF校准与风险历史，无法生成优化配置"}
        return {"available": True, **{k: v for k, v in analysis.items() if k != "covariance"}}

    def get_analysis(self, snapshot_id: str) -> dict:
        self.get_snapshot(snapshot_id)
        return self.analysis_summary(self.find_analysis(snapshot_id))

    def record_forward_event(self, plan_id: str, request: dict) -> dict:
        from strategy_manager.portfolio_models import require, stamp
        plan = self.get_plan(plan_id)
        require(plan.get("schema_version") == "offline_portfolio_plan_v2", "LEGACY_PLAN_NOT_TRACKABLE",
                "旧版等权方案不能作为新版配置的前向记录")
        now = _now()
        observed = stamp(request["observed_at"])
        require(stamp(plan["created_at"]) <= observed <= stamp(now), "INVALID_FORWARD_TIME",
                "前向记录时间必须在方案创建后且不能晚于当前时间；历史回放另行验证")
        kind = request["kind"]
        equity = request.get("account_equity")
        require((kind == "valuation") == (equity is not None), "INVALID_FORWARD_VALUE",
                "净值观察须提供账户净值，其他记录不带净值")
        require(equity is None or isinstance(equity, (int, float)) and not isinstance(equity, bool)
                and math.isfinite(equity) and equity > 0, "INVALID_FORWARD_VALUE", "净值必须是有限正数")
        allowed = {p["code"] for p in plan["positions"]}
        require(set(request.get("execution_status", {})) <= allowed, "INVALID_FORWARD_CODES", "执行记录含方案外股票")
        event = {**request, "event_id": "event_" + uuid.uuid4().hex, "plan_id": plan_id,
                 "whitelist_id": plan["whitelist_id"], "whitelist_version": plan["whitelist_version"],
                 "recorded_at": now, "observed_at": observed.isoformat(), "production_allowed": False,
                 "data_role": plan["data_role"], "historical_scores": True, "source": "USER_RECORDED_NOT_BROKER_VERIFIED"}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT payload FROM forward_events WHERE plan_id=? ORDER BY rowid DESC LIMIT 1",
                                  (plan_id,)).fetchone()
            require(not previous or stamp(json.loads(previous[0])["observed_at"]) <= observed,
                    "FORWARD_EVENT_OUT_OF_ORDER", "新增观察不能回填到已有观察之前")
            db.execute("INSERT INTO forward_events VALUES (?, ?, ?)", (event["event_id"], plan_id, _json(event)))
        return event

    def list_forward_events(self, plan_id: str) -> dict:
        self.get_plan(plan_id)
        with self._db() as db:
            rows = db.execute("SELECT payload FROM forward_events WHERE plan_id=? ORDER BY rowid", (plan_id,)).fetchall()
        events = [json.loads(row[0]) for row in rows]
        values = [e["account_equity"] for e in events if e["kind"] == "valuation"]
        return {"plan_id": plan_id, "events": events, "production_allowed": False,
                "observed_account_change": values[-1]/values[0]-1 if len(values) >= 2 else None,
                "performance_note": "人工账户净值变化含出入金与其他操作，不能直接归因于算法收益"}

    def save_validation_report(self, report: dict) -> None:
        # Internal CLI writer only. No HTTP endpoint accepts self-declared passes.
        _check(report.get("production_allowed") is False and report.get("research_go") is False
               and report.get("holdout_read") is False, "回放报告必须保留非生产门禁")
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO validation_reports VALUES (?, ?)", (report["validation_id"], _json(report)))

    def list_validation_reports(self, snapshot_id: str) -> dict:
        self.get_snapshot(snapshot_id)
        with self._db() as db:
            rows=db.execute("SELECT payload FROM validation_reports ORDER BY rowid DESC").fetchall()
        reports=[json.loads(row[0]) for row in rows]
        return {"reports":[r for r in reports if snapshot_id in r["snapshot_ids"]], "production_allowed":False}
