import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from model_core.holdout import DevelopmentDataView, evaluate_holdout, save_holdout_report
from factor_research.release_audit import audit_engine_release
from model_core.engine import W1ngmanEngine
from model_core.vocab import VOCAB_VERSION


class DummyManager:
    def __init__(self, bars=100):
        self.symbols = ["TEST"]
        base = torch.linspace(100.0, 120.0, bars).view(1, -1)
        self.raw_dict = {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.25,
            "volume": torch.ones_like(base) * 1000,
            "time": torch.arange(bars).view(1, -1).long() * 86400 + 1_600_000_000,
        }
        self.feat_tensor = torch.randn(1, 65, bars)
        self.target_ret = torch.zeros(1, bars)
        self.target_ret[:, :-2] = torch.log(base[:, 2:] / base[:, 1:-1])


def test_development_view_excludes_last_ten_percent():
    full = DummyManager(100)
    dev = DevelopmentDataView(full, 0.10)
    assert dev.spec.split_index == 90
    assert dev.raw_dict["open"].shape == (1, 90)
    assert dev.feat_tensor.shape[0] == 1
    assert dev.feat_tensor.shape[2] == 90
    assert dev.target_ret.shape == (1, 90)
    assert torch.equal(dev.target_ret[:, -2:], torch.zeros(1, 2))


def test_development_view_is_physically_cloned():
    full = DummyManager(100)
    dev = DevelopmentDataView(full, 0.10)
    full.raw_dict["open"][:, 0] = -999
    full.feat_tensor[:, :, 0] = -999
    assert dev.raw_dict["open"][0, 0] != -999
    assert not torch.all(dev.feat_tensor[:, :, 0] == -999)


def test_holdout_report_is_marked_non_selection(tmp_path, monkeypatch):
    full = DummyManager(100)
    # Feature token 0 is a valid one-token frozen formula.
    dev = DevelopmentDataView(full, 0.10)
    result = evaluate_holdout(full, [0], dev.spec, cost_rate=0.0)
    assert result["selection_use"] is False
    assert result["split"]["holdout_bars"] == 10
    assert result["evaluable_bars"] == 8

    monkeypatch.chdir(tmp_path)
    report = save_holdout_report("TEST", [0], result)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["selection_use"] is False
    assert payload["formula"] == [0]


def test_invalid_holdout_fraction_rejected():
    with pytest.raises(ValueError):
        DevelopmentDataView(DummyManager(), 0.0)


def test_checkpoint_without_holdout_protocol_is_rejected(tmp_path):
    full = DummyManager(100)
    dev = DevelopmentDataView(full, 0.10)
    engine = W1ngmanEngine(data_manager=dev, target_symbol="TEST")
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "step": 1,
            "generator_backend": "rd_agent",
            "vocab_version": VOCAB_VERSION,
        },
        legacy,
    )
    with pytest.raises(ValueError, match="holdout"):
        engine.load_checkpoint(str(legacy))


def test_matching_holdout_checkpoint_can_resume(tmp_path):
    full = DummyManager(100)
    dev = DevelopmentDataView(full, 0.10)
    engine = W1ngmanEngine(data_manager=dev, target_symbol="TEST")
    path = tmp_path / "current.pt"
    engine.save_checkpoint(7, str(path))
    restored = W1ngmanEngine(data_manager=dev, target_symbol="TEST")
    assert restored.load_checkpoint(str(path)) == 7


def test_engine_forces_unprotected_manager_onto_development_prefix():
    full = DummyManager(100)
    engine = W1ngmanEngine(data_manager=full, target_symbol="TEST")
    engine._ensure_holdout_isolation()
    assert engine.data_manager.target_ret.shape[1] == 90
    assert engine.holdout_split_index == 90
    assert engine._full_data_manager is full


def test_release_audit_is_immutable_and_reused(tmp_path):
    full = DummyManager(100)
    dev = DevelopmentDataView(full, 0.10)
    engine = SimpleNamespace(
        best_formula=[0],
        best_factor_id="factor-0",
        factor_research=None,
        generator_backend="rd_agent",
        data_protocol=dev.spec.protocol,
        holdout_fraction=0.10,
    )
    store_root = tmp_path / "store"
    report_root = tmp_path / "reports"

    first = audit_engine_release(
        engine, full, dev.spec, "TEST",
        store_root=store_root, report_root=report_root,
    )
    report = Path(first.report_path)
    original = report.read_bytes()
    second = audit_engine_release(
        engine, full, dev.spec, "TEST",
        store_root=store_root, report_root=report_root,
    )

    assert first.status == "complete" and first.reused is False
    assert second.status == "complete" and second.reused is True
    assert second.release_id == first.release_id
    assert report.read_bytes() == original
    with sqlite3.connect(store_root / "research.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM release_audits").fetchone()[0] == 1
