"""Portable training packages without untrusted pickle deserialization."""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from model_core.vocab import FORMULA_VOCAB
from web.progress import (
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    _decode_formula,
    _load_strategy,
    checkpoint_glob,
    invalidate_checkpoint_cache,
)

TRAINING_PACKAGE_FORMAT = "quant_w1ngman_training_v2"
CHECKPOINT_SCHEMA = "quant_w1ngman_checkpoint_json_npz_v1"
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 192 * 1024 * 1024
MAX_MEMBER_COUNT = 8
MAX_COMPRESSION_RATIO = 250
MAX_TENSOR_COUNT = 1024
MAX_JSON_NESTING = 64
MAX_JSON_NODES = 250_000

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_CKPT_NAME_RE = re.compile(
    r"^ckpt_([A-Za-z0-9][A-Za-z0-9._-]{0,63})_rd_step_(\d+)\.pt$",
    re.IGNORECASE,
)
_TENSOR_KEY_RE = re.compile(r"^tensor_\d{6}$")
_ALLOWED_TORCH_DTYPES = {
    str(torch.float16): torch.float16,
    str(torch.float32): torch.float32,
    str(torch.float64): torch.float64,
    str(torch.bfloat16): torch.bfloat16,
    str(torch.int8): torch.int8,
    str(torch.int16): torch.int16,
    str(torch.int32): torch.int32,
    str(torch.int64): torch.int64,
    str(torch.uint8): torch.uint8,
    str(torch.bool): torch.bool,
}


def _history_path(symbol: str) -> Path:
    return PROJECT_ROOT / f"training_history_{symbol}.json"


def _validate_symbol(symbol: Any) -> str:
    value = str(symbol or "").strip()
    if not _SYMBOL_RE.fullmatch(value):
        raise ValueError("训练包品种名称不合法")
    return value


def _symbol_from_ckpt_name(name: str) -> str | None:
    match = _CKPT_NAME_RE.fullmatch(Path(name).name)
    return match.group(1) if match else None


def _load_trusted_local_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint created locally through PyTorch's restricted loader."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"检查点 {path.name} 无法安全读取，请重新训练") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"检查点 {path.name} 不是有效字典")
    return checkpoint


def _validate_checkpoint(checkpoint: dict[str, Any], source: str) -> dict[str, Any]:
    if checkpoint.get("generator_backend") != "rd_agent":
        raise ValueError(f"检查点 {source} 不是 RD-Agent 产物")
    artifact_version = checkpoint.get("vocab_version")
    if artifact_version is None:
        raise ValueError(
            f"检查点 {source} 过旧（无 vocab_version），"
            f"当前词表 {FORMULA_VOCAB.version!r}，请重新训练"
        )
    FORMULA_VOCAB.verify(artifact_version)
    try:
        step = int(checkpoint.get("step", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"检查点 {source} 的 step 不合法") from exc
    if not 0 <= step <= 10_000_000:
        raise ValueError(f"检查点 {source} 的 step 超出允许范围")
    return {
        "step": step,
        "symbol": checkpoint.get("symbol"),
        "best_score": checkpoint.get("best_score"),
    }


def _validate_checkpoint_file(path: Path) -> dict[str, Any]:
    return _validate_checkpoint(_load_trusted_local_checkpoint(path), path.name)


def _encode_checkpoint_value(
    value: Any,
    tensors: dict[str, np.ndarray],
    *,
    depth: int = 0,
    counter: list[int] | None = None,
) -> Any:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if depth > MAX_JSON_NESTING or counter[0] > MAX_JSON_NODES:
        raise ValueError("检查点结构过深或节点过多")

    if isinstance(value, torch.Tensor):
        if len(tensors) >= MAX_TENSOR_COUNT:
            raise ValueError("检查点张量数量超过限制")
        tensor = value.detach().cpu().contiguous()
        dtype_name = str(tensor.dtype)
        if dtype_name not in _ALLOWED_TORCH_DTYPES:
            raise ValueError(f"不支持的张量类型: {dtype_name}")
        key = f"tensor_{len(tensors):06d}"
        if tensor.dtype == torch.bfloat16:
            array = tensor.view(torch.uint16).numpy().copy()
        else:
            array = tensor.numpy().copy()
        tensors[key] = array
        return {"__type__": "tensor", "key": key, "dtype": dtype_name}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        marker = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"__type__": "float", "value": marker}
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [
                [
                    _encode_checkpoint_value(k, tensors, depth=depth + 1, counter=counter),
                    _encode_checkpoint_value(v, tensors, depth=depth + 1, counter=counter),
                ]
                for k, v in value.items()
            ],
        }
    if isinstance(value, list):
        return {
            "__type__": "list",
            "items": [
                _encode_checkpoint_value(v, tensors, depth=depth + 1, counter=counter)
                for v in value
            ],
        }
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "items": [
                _encode_checkpoint_value(v, tensors, depth=depth + 1, counter=counter)
                for v in value
            ],
        }
    raise ValueError(f"检查点包含不允许的对象类型: {type(value).__name__}")


def _decode_checkpoint_value(
    value: Any,
    tensors: dict[str, torch.Tensor],
    used_tensors: set[str],
    *,
    depth: int = 0,
    counter: list[int] | None = None,
) -> Any:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if depth > MAX_JSON_NESTING or counter[0] > MAX_JSON_NODES:
        raise ValueError("检查点元数据结构过深或节点过多")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("检查点元数据包含未编码的非有限浮点数")
        return value
    if not isinstance(value, dict):
        raise ValueError("检查点元数据包含未标记的复合结构")

    kind = value.get("__type__")
    if kind == "float" and set(value) == {"__type__", "value"}:
        markers = {"nan": float("nan"), "inf": float("inf"), "-inf": -float("inf")}
        marker = value.get("value")
        if marker not in markers:
            raise ValueError("检查点浮点标记不合法")
        return markers[marker]
    if kind == "tensor" and set(value) == {"__type__", "key", "dtype"}:
        key = str(value.get("key") or "")
        dtype_name = str(value.get("dtype") or "")
        if not _TENSOR_KEY_RE.fullmatch(key) or dtype_name not in _ALLOWED_TORCH_DTYPES:
            raise ValueError("检查点张量引用不合法")
        if key in used_tensors or key not in tensors:
            raise ValueError("检查点张量引用重复或缺失")
        used_tensors.add(key)
        tensor = tensors[key]
        expected = _ALLOWED_TORCH_DTYPES[dtype_name]
        if expected == torch.bfloat16:
            if tensor.dtype != torch.uint16:
                raise ValueError("bfloat16 张量存储类型不匹配")
            tensor = tensor.view(torch.bfloat16)
        elif tensor.dtype != expected:
            raise ValueError("张量类型与元数据不匹配")
        return tensor
    if kind in {"list", "tuple"} and set(value) == {"__type__", "items"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("检查点序列结构不合法")
        decoded = [
            _decode_checkpoint_value(v, tensors, used_tensors, depth=depth + 1, counter=counter)
            for v in items
        ]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict" and set(value) == {"__type__", "items"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("检查点字典结构不合法")
        result: dict[Any, Any] = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("检查点字典条目不合法")
            key = _decode_checkpoint_value(
                pair[0], tensors, used_tensors, depth=depth + 1, counter=counter
            )
            if not isinstance(key, (str, int, float, bool, type(None))):
                raise ValueError("检查点字典键类型不合法")
            if key in result:
                raise ValueError("检查点字典存在重复键")
            result[key] = _decode_checkpoint_value(
                pair[1], tensors, used_tensors, depth=depth + 1, counter=counter
            )
        return result
    raise ValueError("检查点元数据类型标记不合法")


def _checkpoint_to_safe_files(checkpoint: dict[str, Any]) -> tuple[bytes, bytes]:
    tensors: dict[str, np.ndarray] = {}
    encoded = _encode_checkpoint_value(checkpoint, tensors)
    metadata = json.dumps(
        {"schema": CHECKPOINT_SCHEMA, "checkpoint": encoded},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    tensor_buffer = io.BytesIO()
    np.savez_compressed(tensor_buffer, **tensors)
    return metadata, tensor_buffer.getvalue()


def _validate_inner_npz(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_TENSOR_COUNT:
                raise ValueError("NPZ 张量数量超过限制")
            total = 0
            for info in infos:
                if info.is_dir() or not info.filename.endswith(".npy"):
                    raise ValueError("NPZ 包含非张量成员")
                if not re.fullmatch(r"tensor_\d{6}\.npy", info.filename):
                    raise ValueError("NPZ 张量名称不合法")
                total += info.file_size
                if info.file_size > MAX_MEMBER_BYTES or total > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("NPZ 解压大小超过限制")
                if info.file_size and (
                    info.compress_size <= 0
                    or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise ValueError("NPZ 压缩比异常")
    except zipfile.BadZipFile as exc:
        raise ValueError("张量 NPZ 文件损坏") from exc


def _safe_files_to_checkpoint(metadata: bytes, tensor_payload: bytes) -> dict[str, Any]:
    try:
        document = json.loads(metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("检查点元数据不是有效 JSON") from exc
    if not isinstance(document, dict) or document.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("不支持的安全检查点格式")
    _validate_inner_npz(tensor_payload)

    arrays: dict[str, torch.Tensor] = {}
    try:
        with np.load(io.BytesIO(tensor_payload), allow_pickle=False) as archive:
            for key in archive.files:
                if not _TENSOR_KEY_RE.fullmatch(key):
                    raise ValueError("NPZ 张量键不合法")
                array = archive[key]
                if array.dtype.hasobject:
                    raise ValueError("NPZ 不允许 object 数组")
                arrays[key] = torch.from_numpy(np.array(array, copy=True))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("NPZ"):
            raise
        raise ValueError("无法安全读取张量 NPZ") from exc

    used: set[str] = set()
    checkpoint = _decode_checkpoint_value(document.get("checkpoint"), arrays, used)
    if used != set(arrays):
        raise ValueError("NPZ 包含未被元数据引用的张量")
    if not isinstance(checkpoint, dict):
        raise ValueError("安全检查点根对象不是字典")
    return checkpoint


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_training_export_zip(symbol: str) -> tuple[bytes, str]:
    symbol = _validate_symbol(symbol)
    checkpoints = checkpoint_glob(symbol)
    if not checkpoints:
        raise FileNotFoundError(f"未找到 {symbol} 的训练检查点，请先训练并保存 checkpoint")

    latest = checkpoints[-1]
    checkpoint = _load_trusted_local_checkpoint(latest)
    meta = _validate_checkpoint(checkpoint, latest.name)
    metadata_bytes, tensor_bytes = _checkpoint_to_safe_files(checkpoint)
    files: dict[str, bytes] = {
        "checkpoint/metadata.json": metadata_bytes,
        "checkpoint/tensors.npz": tensor_bytes,
    }

    strategy = _load_strategy(symbol)
    if strategy:
        payload = dict(strategy)
        if payload.get("formula") and not payload.get("formula_decoded"):
            payload["formula_decoded"] = _decode_formula(payload["formula"])
        files[f"strategies/best_{symbol}.json"] = _json_bytes(payload, pretty=True)

    history_path = _history_path(symbol)
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("训练历史 JSON 损坏，拒绝导出") from exc
        files[f"training_history_{symbol}.json"] = _json_bytes(history, pretty=True)

    step = meta["step"]
    checkpoint_name = f"ckpt_{symbol}_rd_step_{step:04d}.pt"
    manifest = {
        "format": TRAINING_PACKAGE_FORMAT,
        "symbol": symbol,
        "step": step,
        "checkpoint_name": checkpoint_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": path, "size": len(payload), "sha256": _sha256(payload)}
            for path, payload in sorted(files.items())
        ],
    }
    files["manifest.json"] = _json_bytes(manifest, pretty=True)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, payload in sorted(files.items()):
            archive.writestr(path, payload)
    result = buffer.getvalue()
    if len(result) > MAX_PACKAGE_BYTES:
        raise ValueError("训练包超过允许大小")
    safe = symbol.replace(".", "_")
    return result, f"training_{safe}_step{step:04d}.zip"


def _validate_member_path(name: str) -> None:
    if (
        "\\" in name
        or not name
        or "\x00" in name
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise ValueError("ZIP 成员路径不合法")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ZIP 成员路径不合法")


def _read_validated_zip(content: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(content) > MAX_PACKAGE_BYTES:
        raise ValueError("训练包超过允许大小")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("训练包 ZIP 已损坏") from exc

    with archive:
        infos = archive.infolist()
        if not 1 <= len(infos) <= MAX_MEMBER_COUNT:
            raise ValueError("ZIP 文件数量超过限制")
        seen: set[str] = set()
        total = 0
        for info in infos:
            _validate_member_path(info.filename)
            if info.is_dir() or info.filename in seen:
                raise ValueError("ZIP 包含目录或重复成员")
            seen.add(info.filename)
            mode = (info.external_attr >> 16) & 0xF000
            if mode == stat.S_IFLNK:
                raise ValueError("ZIP 不允许符号链接")
            total += info.file_size
            if info.file_size > MAX_MEMBER_BYTES or total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP 解压大小超过限制")
            if info.file_size and (
                info.compress_size <= 0
                or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
            ):
                raise ValueError("ZIP 压缩比异常")
        if "manifest.json" not in seen:
            raise ValueError("训练包缺少 manifest.json")
        payloads = {info.filename: archive.read(info) for info in infos}

    try:
        manifest = json.loads(payloads["manifest.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json 无效") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != TRAINING_PACKAGE_FORMAT:
        raise ValueError("仅支持安全的 v2 训练包；旧包请在可信原设备重新导出")
    return manifest, payloads


def _validate_json_artifact(payload: bytes, kind: str, symbol: str) -> None:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} 不是有效 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{kind} 必须是 JSON 对象")
    artifact_symbol = value.get("symbol")
    if artifact_symbol not in (None, "", symbol):
        raise ValueError(f"{kind} 的品种与 manifest 不一致")
    backend = value.get("generator_backend")
    backend_ok = backend in (None, "", "rd_agent") or (
        isinstance(backend, list)
        and all(item == "rd_agent" for item in backend)
    )
    if not backend_ok:
        raise ValueError(f"{kind} 不是 RD-Agent 产物")


def _validated_package(
    content: bytes,
    expected_symbol: str | None,
) -> tuple[str, int, str, dict[str, bytes], dict[str, Any]]:
    manifest, payloads = _read_validated_zip(content)
    symbol = _validate_symbol(manifest.get("symbol"))
    if expected_symbol is not None and symbol != _validate_symbol(expected_symbol):
        raise ValueError(f"训练包品种为 {symbol}，与当前选择的 {expected_symbol} 不一致")
    try:
        step = int(manifest.get("step"))
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest step 不合法") from exc
    if not 0 <= step <= 10_000_000:
        raise ValueError("manifest step 超出允许范围")
    checkpoint_name = str(manifest.get("checkpoint_name") or "")
    expected_name = f"ckpt_{symbol}_rd_step_{step:04d}.pt"
    if checkpoint_name != expected_name or _symbol_from_ckpt_name(checkpoint_name) != symbol:
        raise ValueError("manifest checkpoint_name 不合法")

    records = manifest.get("files")
    if not isinstance(records, list) or not 2 <= len(records) <= MAX_MEMBER_COUNT - 1:
        raise ValueError("manifest 文件清单不合法")
    expected_optional = {
        f"strategies/best_{symbol}.json",
        f"training_history_{symbol}.json",
    }
    allowed = {"checkpoint/metadata.json", "checkpoint/tensors.npz", *expected_optional}
    listed: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("manifest 文件记录不合法")
        path = str(record.get("path") or "")
        _validate_member_path(path)
        if path not in allowed or path in listed:
            raise ValueError("manifest 包含未授权或重复文件")
        listed.add(path)
        payload = payloads.get(path)
        if payload is None:
            raise ValueError("manifest 声明的文件缺失")
        try:
            size = int(record.get("size"))
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest 文件大小不合法") from exc
        digest = str(record.get("sha256") or "")
        if size != len(payload) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("manifest 文件大小或哈希格式不合法")
        if not hmac.compare_digest(_sha256(payload), digest):
            raise ValueError("训练包文件哈希校验失败")
    if listed != set(payloads) - {"manifest.json"}:
        raise ValueError("ZIP 包含 manifest 未声明的文件")
    if not {"checkpoint/metadata.json", "checkpoint/tensors.npz"}.issubset(listed):
        raise ValueError("训练包缺少安全检查点文件")

    checkpoint = _safe_files_to_checkpoint(
        payloads["checkpoint/metadata.json"], payloads["checkpoint/tensors.npz"]
    )
    checkpoint_meta = _validate_checkpoint(checkpoint, "training package")
    if checkpoint_meta["step"] != step:
        raise ValueError("检查点 step 与 manifest 不一致")
    checkpoint_symbol = checkpoint_meta.get("symbol")
    if checkpoint_symbol not in (None, "", symbol):
        raise ValueError("检查点品种与 manifest 不一致")
    strategy_name = f"strategies/best_{symbol}.json"
    history_name = f"training_history_{symbol}.json"
    if strategy_name in payloads:
        _validate_json_artifact(payloads[strategy_name], "策略文件", symbol)
    if history_name in payloads:
        _validate_json_artifact(payloads[history_name], "训练历史", symbol)
    return symbol, step, checkpoint_name, payloads, checkpoint


def _transactional_install(
    symbol: str,
    checkpoint_name: str,
    payloads: dict[str, bytes],
    checkpoint: dict[str, Any],
) -> list[str]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_name = f"strategies/best_{symbol}.json"
    history_name = f"training_history_{symbol}.json"
    destinations: dict[Path, bytes | None] = {CHECKPOINT_DIR / checkpoint_name: None}
    if strategy_name in payloads:
        destinations[PROJECT_ROOT / PurePosixPath(strategy_name)] = payloads[strategy_name]
    if history_name in payloads:
        destinations[PROJECT_ROOT / history_name] = payloads[history_name]

    with tempfile.TemporaryDirectory(prefix=".training-import-", dir=PROJECT_ROOT) as raw_tmp:
        tmp = Path(raw_tmp)
        staged: dict[Path, Path] = {}
        for index, (destination, payload) in enumerate(destinations.items()):
            stage_path = tmp / "staged" / f"{index:03d}_{destination.name}"
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            if payload is None:
                torch.save(checkpoint, stage_path)
                _validate_checkpoint_file(stage_path)
            else:
                stage_path.write_bytes(payload)
            staged[destination] = stage_path

        affected = set(checkpoint_glob(symbol)) | set(destinations)
        if history_name not in payloads:
            affected.add(_history_path(symbol))
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for index, destination in enumerate(sorted(affected, key=str)):
                if not destination.exists():
                    continue
                backup = tmp / "backup" / f"{index:03d}_{destination.name}"
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backups[destination] = backup
            for destination, stage_path in staged.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage_path, destination)
                installed.append(destination)
        except Exception:
            for destination in reversed(installed):
                destination.unlink(missing_ok=True)
            for destination, backup in backups.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
            raise

    return [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in installed]


def import_training_package(
    content: bytes,
    filename: str,
    expected_symbol: str | None = None,
) -> dict[str, Any]:
    name = Path(filename).name.lower()
    if name.endswith(".pt"):
        raise ValueError(
            "出于安全原因不再接受 .pt 上传（pickle 可执行任意代码）；"
            "请在可信原设备导出新的 .zip 安全训练包"
        )
    if not name.endswith(".zip"):
        raise ValueError("仅支持 quant_w1ngman v2 .zip 安全训练包")

    symbol, step, checkpoint_name, payloads, checkpoint = _validated_package(
        content, expected_symbol
    )
    installed = _transactional_install(symbol, checkpoint_name, payloads, checkpoint)
    invalidate_checkpoint_cache()
    return {
        "ok": True,
        "symbol": symbol,
        "step": step,
        "installed": installed,
        "message": f"已安全导入 {symbol} 的训练文件（step {step}），下次训练将从断点续训",
    }
