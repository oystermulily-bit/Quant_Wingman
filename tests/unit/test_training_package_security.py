from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import torch

from model_core.vocab import FORMULA_VOCAB
import web.progress as progress
import web.training_package as packages


@pytest.fixture
def package_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkpoint_dir = tmp_path / "checkpoints"
    strategy_dir = tmp_path / "strategies"
    checkpoint_dir.mkdir()
    strategy_dir.mkdir()
    monkeypatch.setattr(packages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(packages, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(progress, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(progress, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(progress, "STRATEGIES_DIR", strategy_dir)
    return tmp_path


def _write_checkpoint(root: Path, step: int = 7) -> Path:
    path = root / "checkpoints" / f"ckpt_TEST_rd_step_{step:04d}.pt"
    torch.save(
        {
            "generator_backend": "rd_agent",
            "vocab_version": FORMULA_VOCAB.version,
            "step": step,
            "train_steps": 30,
            "best_score": 1.25,
            "best_formula": [0, 1, 2],
            "factor_pool": [(1.0, 0, torch.tensor([[1.0, 2.0]]))],
            "training_history": {"step": [1, step]},
        },
        path,
    )
    return path


def _rewrite_zip(source: bytes, replacements: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as old, zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED
    ) as new:
        for info in old.infolist():
            new.writestr(info.filename, replacements.get(info.filename, old.read(info)))
    return output.getvalue()


def test_safe_package_round_trip_and_replaces_only_after_validation(
    package_project: Path,
) -> None:
    original = _write_checkpoint(package_project)
    body, filename = packages.build_training_export_zip("TEST")
    old = _write_checkpoint(package_project, step=1)
    original.write_bytes(b"not-a-checkpoint-anymore")
    stale_history = package_project / "training_history_TEST.json"
    stale_history.write_text(json.dumps({"step": [999]}), encoding="utf-8")

    result = packages.import_training_package(body, filename, expected_symbol="TEST")

    assert result["step"] == 7
    assert not old.exists()
    assert not stale_history.exists()
    restored = torch.load(original, map_location="cpu", weights_only=True)
    assert restored["step"] == 7
    assert torch.equal(restored["factor_pool"][0][2], torch.tensor([[1.0, 2.0]]))


def test_hash_failure_leaves_existing_checkpoint_untouched(
    package_project: Path,
) -> None:
    existing = _write_checkpoint(package_project)
    before = existing.read_bytes()
    body, filename = packages.build_training_export_zip("TEST")
    tampered = _rewrite_zip(body, {"checkpoint/metadata.json": b"{}"})

    with pytest.raises(ValueError, match="哈希"):
        packages.import_training_package(tampered, filename, expected_symbol="TEST")

    assert existing.read_bytes() == before


def test_zip_traversal_is_rejected_without_writing(package_project: Path) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../../outside.txt", b"owned")
        archive.writestr(
            "manifest.json",
            json.dumps({"format": packages.TRAINING_PACKAGE_FORMAT}),
        )

    with pytest.raises(ValueError, match="路径"):
        packages.import_training_package(output.getvalue(), "attack.zip")

    assert not (package_project.parent / "outside.txt").exists()


def test_direct_pt_upload_is_rejected_before_torch_load(
    package_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("torch.load must not run for uploaded .pt files")

    monkeypatch.setattr(packages.torch, "load", fail_if_called)
    with pytest.raises(ValueError, match="不再接受 .pt"):
        packages.import_training_package(b"malicious pickle", "ckpt_TEST_rd_step_0001.pt")
