from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from model_core.vocab import FORMULA_VOCAB


COMMUTATIVE = {"ADD", "MUL", "MAX", "MIN"}


def canonical_formula(tokens: Iterable[int], arity_map: dict[int, int]) -> str:
    """Canonicalise RPN structure without changing non-commutative semantics."""
    stack: list[str] = []
    names = FORMULA_VOCAB.token_names
    for raw in tokens:
        token = int(raw)
        if token < FORMULA_VOCAB.operator_offset:
            stack.append(f"F:{names[token]}")
            continue
        arity = arity_map.get(token)
        if arity is None or len(stack) < arity:
            raise ValueError("invalid RPN formula")
        args = stack[-arity:]
        del stack[-arity:]
        name = names[token]
        if name in COMMUTATIVE:
            args = sorted(args)
        stack.append(f"{name}({','.join(args)})")
    if len(stack) != 1:
        raise ValueError("invalid RPN formula")
    return stack[0]


def feature_implementation_version(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "model_core/features.py",
        "model_core/ops.py",
        "model_core/vm.py",
        "model_core/vocab.py",
    ):
        path = project_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return "impl-" + digest.hexdigest()[:16]


def research_protocol_version(
    config,
    project_root: Path,
    *,
    n_folds: int,
    label_horizon: int,
) -> str:
    """Fingerprint every setting and implementation that can change selection."""
    settings = asdict(config)
    for runtime_only in (
        "enabled",
        "store_root",
        "analysis_enabled",
        "analysis_timeout_seconds",
    ):
        settings.pop(runtime_only, None)
    payload = {
        "settings": settings,
        "n_folds": int(n_folds),
        "label_horizon": int(label_horizon),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for relative in (
        "factor_research/validation_unit.py",
        "factor_research/lgbm_baseline.py",
        "factor_research/retention_policy.py",
        "factor_research/research_loop.py",
        "model_core/backtest.py",
        "model_core/engine.py",
        "model_core/holdout.py",
    ):
        path = project_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"{config.validation_protocol}-{digest.hexdigest()[:16]}"


def data_fingerprint(data_manager) -> str:
    """Hash development tensors only; the protected holdout is never inspected."""
    digest = hashlib.sha256()
    for name in ("open", "high", "low", "close", "volume", "time"):
        value = (getattr(data_manager, "raw_dict", None) or {}).get(name)
        if value is None:
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    if digest.digest() == hashlib.sha256().digest():
        value = data_manager.target_ret.detach().cpu().contiguous().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def factor_identity(
    tokens: list[int],
    *,
    arity_map: dict[int, int],
    vocab_version: str,
    implementation_version: str,
    data_fingerprint_value: str,
    validation_protocol: str,
) -> tuple[str, str]:
    canonical = canonical_formula(tokens, arity_map)
    payload = {
        "canonical_formula": canonical,
        "vocab_version": vocab_version,
        "feature_implementation_version": implementation_version,
        "data_fingerprint": data_fingerprint_value,
        "validation_protocol": validation_protocol,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), canonical


def save_factor_values(
    path: Path,
    factor: torch.Tensor,
    invalid_mask: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values_tensor = factor.detach().cpu().to(torch.float32).clone()
    if invalid_mask is not None:
        values_tensor.masked_fill_(invalid_mask.detach().cpu().bool(), float("nan"))
    values = values_tensor.numpy()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, values=values)
    temporary.replace(path)


def load_factor_values(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["values"], dtype=np.float32)
    return torch.from_numpy(values.copy())
