import json

import pytest

from model_core.config import ModelConfig
from model_core.engine import W1ngmanEngine
from model_core.rd_agent_generator import (
    FormulaCandidate,
    RDFormulaGenerator,
    RDFormulaLimits,
)


def _generator(tmp_path):
    return RDFormulaGenerator(
        tmp_path,
        model="deepseek-ai/DeepSeek-V4-Pro",
        limits=RDFormulaLimits(max_tokens=48, max_depth=7, max_operators=12),
    )


def test_rd_limits_match_requested_ranges():
    assert 32 <= ModelConfig.RD_AGENT_MAX_TOKENS <= 64
    assert 5 <= ModelConfig.RD_AGENT_MAX_DEPTH <= 9
    assert 8 <= ModelConfig.RD_AGENT_MAX_OPERATORS <= 16
    assert 5 <= ModelConfig.RD_AGENT_FORMULA_TIMEOUT_SECONDS <= 10
    assert 60 <= ModelConfig.RD_AGENT_ROUND_TIMEOUT_SECONDS <= 180


def test_valid_rpn_is_converted_to_native_tokens(tmp_path):
    generator = _generator(tmp_path)
    result = generator._validate(["RET", "TS_MEAN_5", "RET5", "SUB"])
    assert result is not None
    assert result.token_names == ["RET", "TS_MEAN_5", "RET5", "SUB"]


@pytest.mark.parametrize(
    "tokens",
    [
        ["ADD"],
        ["RET", "UNKNOWN_OPERATOR"],
        ["RET", "RET5"],
    ],
)
def test_invalid_rpn_is_rejected(tmp_path, tokens):
    assert _generator(tmp_path)._validate(tokens) is None


def test_feedback_drops_raw_data_fields():
    clean = RDFormulaGenerator._sanitise_feedback(
        [{
            "formula": ["RET", "TS_MEAN_5"],
            "rank": 1,
            "status": "elite",
            "prices": [1.0, 2.0],
            "returns": [0.1],
            "data_file": "secret.parquet",
            "val_score": 9.9,
        }]
    )
    encoded = json.dumps(clean)
    assert "prices" not in encoded
    assert "returns" not in encoded
    assert "data_file" not in encoded
    assert "val_score" not in encoded


def test_generate_calls_pinned_siliconflow_endpoint_without_market_data(
    tmp_path, monkeypatch
):
    import model_core.rd_agent_generator as module

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "formulas": [{
                                "tokens": ["RET", "TS_MEAN_5", "RET5", "SUB"],
                                "rationale": "trend spread",
                            }]
                        })
                    }
                }]
            }).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = request.data.decode()
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    candidates = _generator(tmp_path).generate(
        round_index=0,
        count=1,
        feedback=[{
            "formula": ["RET", "TS_MEAN_5"],
            "rank": 1,
            "prices": [100, 101],
            "returns": [0.01],
            "data_file": "private.parquet",
        }],
    )

    assert captured["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert "private.parquet" not in captured["payload"]
    assert "prices" not in captured["payload"]
    assert "returns" not in captured["payload"]
    payload = json.loads(captured["payload"])
    assert payload["enable_thinking"] is False
    assert payload["max_tokens"] == 2048
    assert candidates[0].token_names[-1] == "SUB"


def test_generate_retries_after_read_timeout(tmp_path, monkeypatch):
    import model_core.rd_agent_generator as module

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "formulas": [{
                                "tokens": ["RET", "TS_MEAN_5", "RET5", "SUB"],
                                "rationale": "trend spread",
                            }]
                        })
                    }
                }]
            }).encode()

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("read operation timed out")
        return Response()

    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    candidates = _generator(tmp_path).generate(round_index=0, count=1)

    assert len(calls) == 2
    assert all(timeout <= 90 for timeout in calls)
    assert candidates[0].token_names[-1] == "SUB"


def test_rd_backend_does_not_instantiate_legacy_w1ngman():
    engine = W1ngmanEngine(data_manager=None, use_lord_regularization=True)
    assert engine.generator_backend == "rd_agent"
    assert engine.model is None
    assert engine.opt is None
    assert engine.use_lord is False


def test_engine_refuses_any_non_rd_generator(monkeypatch):
    monkeypatch.setattr(ModelConfig, "GENERATOR_BACKEND", "w1ngman")
    with pytest.raises(RuntimeError, match="RD-Agent"):
        W1ngmanEngine(data_manager=None)


def test_rd_checkpoint_round_trip_has_backend_guard(tmp_path):
    path = tmp_path / "rd.pt"
    engine = W1ngmanEngine(data_manager=None)
    engine.best_score = 1.25
    engine.best_formula = [0, 77]
    engine.save_checkpoint(3, str(path))

    restored = W1ngmanEngine(data_manager=None)
    completed = restored.load_checkpoint(str(path))
    assert completed == 3
    assert restored.best_score == pytest.approx(1.25)
    assert restored.best_formula == [0, 77]


def test_rd_training_only_promotes_a_factor_accepted_by_research_loop(monkeypatch):
    class DataManager:
        target_ret = __import__("torch").zeros((1, 500))
        feat_tensor = __import__("torch").zeros((1, 65, 500))
        raw_dict = {}

    engine = W1ngmanEngine(data_manager=DataManager(), target_symbol=None)
    candidate = engine.rd_generator._validate(["RET", "TS_MEAN_5"])
    assert candidate is not None
    monkeypatch.setattr(
        engine.rd_generator,
        "generate",
        lambda **kwargs: [FormulaCandidate(candidate.tokens, candidate.token_names)],
    )
    factor = __import__("torch").ones((1, 450))
    monkeypatch.setattr(
        engine,
        "_eval_formula_task",
        lambda idx, fml, *args: {
            "idx": idx,
            "status": "ok",
            "reward": 1.0,
            "val_score": 1.1,
            "ic_full": 0.1,
            "ic_stab": 0.1,
            "ic_i": 0.1,
            "res": factor,
            "fml": fml,
        },
    )
    monkeypatch.setattr(engine, "_save_strategy_live", lambda: None)
    monkeypatch.setattr(engine, "save_checkpoint", lambda step: f"mock-{step}.pt")
    from factor_research.schemas import RetentionDecision
    from types import SimpleNamespace

    accepted = SimpleNamespace(
        factor_id="factor-1",
        formula_tokens=candidate.tokens,
        decision=RetentionDecision(True, [], 0.75),
        validation=SimpleNamespace(rank_icir=0.6),
        lgbm=SimpleNamespace(delta=0.02),
    )

    def accept_round(_self, _round, _candidates, results):
        results[0]["factor_id"] = "factor-1"
        return SimpleNamespace(
            accepted=[accepted], assessments=[accepted], duplicate_count=0,
            sota_status={"accepted_count": 1},
        )

    monkeypatch.setattr(
        "factor_research.FactorResearchLoop.assess_round", accept_round
    )

    engine.train(start_step=0, end_step=1, verbose_header=False)

    assert engine.best_formula == candidate.tokens
    assert engine.best_score == pytest.approx(0.75)
    assert engine.training_history["generator_backend"] == ["rd_agent"]
