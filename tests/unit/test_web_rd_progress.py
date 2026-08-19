import json

import torch

import web.progress as progress


def test_checkpoint_discovery_only_returns_rd_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "CHECKPOINT_DIR", tmp_path)
    legacy = tmp_path / "ckpt_TEST_step_0099.pt"
    rd = tmp_path / "ckpt_TEST_rd_step_0002.pt"
    torch.save({"generator_backend": "w1ngman", "step": 99}, legacy)
    torch.save({"generator_backend": "rd_agent", "step": 2}, rd)

    assert progress.checkpoint_glob("TEST") == [rd]


def test_checkpoint_metadata_keeps_custom_round_target(tmp_path):
    path = tmp_path / "ckpt_TEST_rd_step_0002.pt"
    torch.save(
        {
            "generator_backend": "rd_agent",
            "step": 2,
            "train_steps": 77,
        },
        path,
    )
    assert progress._load_checkpoint_meta(path)["train_steps"] == 77


def test_strategy_loader_hides_legacy_generator_results(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "STRATEGIES_DIR", tmp_path)
    path = tmp_path / "best_TEST.json"
    path.write_text(json.dumps({"generator_backend": "w1ngman", "best_score": 9}), encoding="utf-8")
    assert progress._load_strategy("TEST") is None

    path.write_text(json.dumps({"generator_backend": "rd_agent", "best_score": 1}), encoding="utf-8")
    assert progress._load_strategy("TEST")["best_score"] == 1


def test_history_picker_rejects_entropy_only_legacy_history():
    legacy = {"step": [0], "entropy": [3.8], "best_score": [9.0]}
    rd = {
        "step": [0],
        "generator_backend": ["rd_agent"],
        "rd_candidates": [16],
        "best_score": [1.0],
    }
    assert progress._pick_training_history(legacy, rd) is rd
    assert progress._pick_training_history(legacy, None) is None
