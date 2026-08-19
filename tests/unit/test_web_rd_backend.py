import pytest
from pydantic import ValidationError

import web.training_manager as module
import web.app as web_app
from web.app import StartTrainingRequest


class _RunningProcess:
    pid = 4321

    def poll(self):
        return None


def test_web_start_explicitly_locks_child_to_rd_agent(tmp_path, monkeypatch):
    captured = {}
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _RunningProcess()

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "LOG_DIR", log_dir)
    monkeypatch.setattr(module.ModelConfig, "GENERATOR_BACKEND", "rd_agent")
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    manager = module.TrainingManager()
    try:
        job = manager.start("sample.parquet", "TEST", "D1", rounds=77)
    finally:
        if manager._log_fp is not None:
            manager._log_fp.close()

    backend_index = captured["cmd"].index("--generator-backend")
    assert captured["cmd"][backend_index + 1] == "rd_agent"
    rounds_index = captured["cmd"].index("--rounds")
    assert captured["cmd"][rounds_index + 1] == "77"
    assert captured["kwargs"]["env"]["W1NGMAN_GENERATOR_BACKEND"] == "rd_agent"
    assert captured["kwargs"]["env"]["RD_AGENT_ROUNDS"] == "77"
    assert job.to_dict()["generator_backend"] == "rd_agent"
    assert job.to_dict()["rounds"] == 77
    log_text = (log_dir / job.log_path.split("/")[-1]).read_text(encoding="utf-8")
    assert "公式生成器=RD-Agent" in log_text
    assert "REINFORCE=disabled" in log_text
    assert "rounds=77" in log_text


@pytest.mark.parametrize("rounds", [0, 10001])
def test_web_start_rejects_rounds_outside_range(rounds):
    manager = module.TrainingManager()
    with pytest.raises(ValueError, match="1.*10000"):
        manager.start("sample.parquet", "TEST", "D1", rounds=rounds)


def test_web_start_refuses_legacy_reinforce_backend(monkeypatch):
    monkeypatch.setattr(module.ModelConfig, "GENERATOR_BACKEND", "w1ngman")
    manager = module.TrainingManager()

    with pytest.raises(RuntimeError, match="RD-Agent"):
        manager.start("sample.parquet", "TEST", "D1", rounds=30)


def test_training_request_accepts_custom_rounds():
    request = StartTrainingRequest(data_file="sample.parquet", rounds=321)
    assert request.rounds == 321


def test_training_request_requires_rounds():
    with pytest.raises(ValidationError):
        StartTrainingRequest(data_file="sample.parquet")


@pytest.mark.parametrize("rounds", [0, 10001, 1.5])
def test_training_request_validates_rounds(rounds):
    with pytest.raises(ValidationError):
        StartTrainingRequest(data_file="sample.parquet", rounds=rounds)


def test_live_progress_uses_job_round_target(monkeypatch):
    class Progress:
        symbol = "TEST"
        current_step = 15
        train_steps = 30
        best_score = 1.0
        formula_decoded = "RET"
        status = "in_progress"
        history = None
        checkpoint_path = None
        has_strategy = False

    monkeypatch.setattr(web_app, "get_symbol_progress", lambda _symbol: Progress())
    monkeypatch.setattr(web_app.training_manager, "parse_step_from_log", lambda: None)

    row = web_app._progress_with_live_step(
        "TEST", active=True, train_steps_override=75
    )
    assert row["train_steps"] == 75
    assert row["progress_pct"] == 20.0
