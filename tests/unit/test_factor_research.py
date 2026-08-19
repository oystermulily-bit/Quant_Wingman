import json
from types import SimpleNamespace

import pytest
import torch

from factor_research.analysis_unit import AnalysisUnit
from factor_research.config import ResearchConfig
from factor_research.factor_artifact import canonical_formula, factor_identity
from factor_research.retention_policy import RetentionPolicy
from factor_research.schemas import LGBMIncrementResult, ValidationResult
from factor_research.sota_store import SOTAStore
from factor_research.lgbm_baseline import FixedLightGBMBaseline
from factor_research.schemas import CandidateAssessment, RetentionDecision
from factor_research.validation_unit import _rank_ic
from factor_research.research_loop import FactorResearchLoop
from factor_research.schemas import AnalysisReport
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION


def _validation(**overrides):
    values = dict(
        valid=True, coverage=1.0, ic=0.02, icir=0.5, rank_ic=0.03,
        rank_icir=0.6, direction_ratio=0.8, bootstrap_ci_low=0.001,
        bootstrap_ci_high=0.05, net_return=0.2, annual_return=0.1,
        net_sharpe=0.8, sortino=1.0, calmar=0.9, max_drawdown=0.1,
        turnover=0.2, long_exposure=0.3, short_exposure=0.3,
        max_correlation=0.2, portfolio_delta=0.1, fold_results=[],
    )
    values.update(overrides)
    return ValidationResult(**values)


def test_factor_identity_includes_protocol_and_commutative_normalisation():
    vm = StackVM()
    names = FORMULA_VOCAB.token_names
    ret, ret5 = names.index("RET"), names.index("RET5")
    add = names.index("ADD")
    left = canonical_formula([ret, ret5, add], vm.arity_map)
    right = canonical_formula([ret5, ret, add], vm.arity_map)
    assert left == right
    one, _ = factor_identity(
        [ret, ret5, add], arity_map=vm.arity_map, vocab_version=VOCAB_VERSION,
        implementation_version="impl", data_fingerprint_value="data",
        validation_protocol="v1",
    )
    two, _ = factor_identity(
        [ret, ret5, add], arity_map=vm.arity_map, vocab_version=VOCAB_VERSION,
        implementation_version="impl", data_fingerprint_value="data",
        validation_protocol="v2",
    )
    assert one != two


def test_sota_store_persists_metadata_and_factor_matrix(tmp_path):
    store = SOTAStore(tmp_path)
    record = {
        "factor_id": "f1", "canonical_formula": "F:RET", "formula_tokens": [0],
        "formula_decoded": "RET", "vocab_version": VOCAB_VERSION,
        "feature_implementation_version": "impl", "data_fingerprint": "data",
        "validation_protocol": "v1", "llm_model": "test", "created_round": 1,
        "status": "accepted", "metrics": {"rank_ic": 0.02},
        "max_correlation": 0.1, "ranking_score": 0.7,
    }
    store.save_result(record, torch.arange(20).view(1, 20).float())
    matrix, rows = store.matrix("data", "v1")
    assert matrix.shape == (1, 1, 20)
    assert rows[0]["formula_tokens"] == [0]
    assert store.status("data", "v1")["accepted_count"] == 1


def test_retention_requires_stable_positive_lgbm_increment():
    policy = RetentionPolicy(ResearchConfig())
    rejected = policy.decide(
        _validation(),
        LGBMIncrementResult(0.01, 0.0, -0.01, 0.25, [-0.01] * 4, 100),
    )
    assert not rejected.accepted
    assert any("LightGBM" in reason for reason in rejected.reasons)
    accepted = policy.decide(
        _validation(),
        LGBMIncrementResult(0.01, 0.03, 0.02, 0.75, [0.01] * 4, 100),
    )
    assert accepted.accepted


def test_analysis_payload_contains_aggregates_not_market_rows():
    clean = {
        "candidate": {"metrics": _validation().to_dict()},
        "sota": {"accepted_count": 2},
    }
    encoded = json.dumps(clean)
    for forbidden in ("prices", "data_file", "timestamps"):
        assert forbidden not in encoded


def test_fixed_lightgbm_detects_oof_signal_increment():
    pytest.importorskip("lightgbm")
    generator = torch.Generator().manual_seed(7)
    candidate = torch.randn((1, 320), generator=generator)
    target = 0.25 * candidate + 0.02 * torch.randn((1, 320), generator=generator)
    folds = [
        {"train_start": 0, "train_end": 100, "val_start": 120, "val_end": 180},
        {"train_start": 80, "train_end": 180, "val_start": 200, "val_end": 260},
        {"train_start": 140, "train_end": 240, "val_start": 260, "val_end": 320},
    ]
    config = ResearchConfig(
        purge_gap=20, embargo_bars=0, lgbm_num_boost_round=30,
        lgbm_min_child_samples=10,
    )
    result = FixedLightGBMBaseline(config).compare(None, candidate, target, folds)
    assert result.delta > 0.5
    assert result.positive_fold_ratio == 1.0
    assert result.oof_samples == 180


def test_rank_ic_uses_average_ranks_for_ties():
    actual = _rank_ic(
        torch.tensor([0.0, 0.0, 1.0]).numpy(),
        torch.tensor([0.0, 1.0, 2.0]).numpy(),
    )
    assert actual == pytest.approx(0.8660254038)


def test_ranking_score_is_independent_of_current_round_peers():
    policy = RetentionPolicy(ResearchConfig())

    def assessment(factor_id: str, rank_icir: float) -> CandidateAssessment:
        validation = _validation(rank_icir=rank_icir)
        lgbm = LGBMIncrementResult(0.0, 0.02, 0.02, 1.0, [0.02] * 4, 100)
        return CandidateAssessment(
            factor_id=factor_id,
            formula_tokens=[0],
            formula_decoded="RET",
            hypothesis="test",
            rationale="test",
            validation=validation,
            lgbm=lgbm,
            decision=RetentionDecision(True, []),
            operator_count=0,
        )

    candidate = assessment("same", 0.6)
    policy.rank([candidate])
    alone = candidate.decision.ranking_score
    candidate_again = assessment("same", 0.6)
    policy.rank([candidate_again, assessment("strong", 2.0)])
    assert candidate_again.decision.ranking_score == pytest.approx(alone)


def test_same_round_candidates_are_forward_selected_against_updated_sota():
    class Store:
        def __init__(self):
            self.saved = []

        def matrix(self, *_):
            return None, []

        def has(self, _):
            return False

        def save_result(self, record, factor, invalid_mask=None):
            self.saved.append(record)

        def status(self, *_):
            accepted = [row for row in self.saved if row["status"] == "accepted"]
            return {"accepted_count": len(accepted), "factor_ids": [r["factor_id"] for r in accepted]}

    class Knowledge:
        def record_failure(self, *args, **kwargs):
            pass

        def add_hypothesis(self, *args, **kwargs):
            return "hypothesis"

        def record_relations(self, *args, **kwargs):
            pass

        def record_assessment(self, *args, **kwargs):
            pass

        def record_analysis(self, *args, **kwargs):
            pass

    class Validation:
        def evaluate(self, factor, target, folds, sota_matrix=None, invalid_mask=None):
            return _validation(max_correlation=0.95 if sota_matrix is not None else 0.0)

        def correlations(self, factor, sota_matrix):
            return [] if sota_matrix is None else [0.95] * sota_matrix.shape[1]

    class LGBM:
        def __init__(self):
            self.sota_sizes = []

        def compare(self, sota, candidate, target, folds, candidate_invalid_mask=None):
            size = 0 if sota is None else sota.shape[1]
            self.sota_sizes.append(size)
            delta = 0.02 if size == 0 else 0.0
            return LGBMIncrementResult(0.0, delta, delta, 1.0 if delta else 0.0, [delta] * 4, 100)

    loop = FactorResearchLoop.__new__(FactorResearchLoop)
    loop.config = ResearchConfig(max_accept_per_round=3)
    loop.data_manager = SimpleNamespace(target_ret=torch.randn(1, 80))
    loop.vm = StackVM()
    loop.generator = SimpleNamespace(model="test")
    loop.folds = [{"train_start": 0, "train_end": 40, "val_start": 60, "val_end": 80}]
    loop.symbol = "test"
    loop.implementation_version = "impl"
    loop.data_fingerprint = "data"
    loop.validation_protocol = "protocol"
    loop.store = Store()
    loop.knowledge = Knowledge()
    loop.validation = Validation()
    loop.lgbm = LGBM()
    loop.retention = RetentionPolicy(loop.config)
    loop.analysis = SimpleNamespace(
        analyse=lambda *args: AnalysisReport("done", [], [], [], [], 1.0, "test")
    )

    candidates = [
        SimpleNamespace(tokens=[0], token_names=["RET"], hypothesis="a", rationale="a"),
        SimpleNamespace(tokens=[1], token_names=["RET5"], hypothesis="b", rationale="b"),
    ]
    results = [
        {"idx": 0, "status": "ok", "res": torch.randn(1, 80)},
        {"idx": 1, "status": "ok", "res": torch.randn(1, 80)},
    ]
    outcome = loop.assess_round(1, candidates, results)

    assert len(outcome.accepted) == 1
    assert 1 in loop.lgbm.sota_sizes


def test_lgbm_baseline_and_challenge_share_samples_and_keep_validation_start(monkeypatch):
    candidate = torch.randn(1, 160)
    candidate[:, 25] = float("nan")
    target = torch.randn(1, 160)
    invalid = torch.zeros_like(candidate, dtype=torch.bool)
    invalid[:, 35] = True
    folds = [
        {"train_start": 0, "train_end": 60, "val_start": 80, "val_end": 110},
        {"train_start": 0, "train_end": 100, "val_start": 120, "val_end": 150},
    ]
    baseline = FixedLightGBMBaseline(
        ResearchConfig(purge_gap=20, embargo_bars=5)
    )
    calls = []

    def fake_fit(features, labels, train, valid):
        calls.append((train.copy(), valid.copy()))
        return labels.reshape(-1)[valid]

    monkeypatch.setattr(baseline, "_fit_predict", fake_fit)
    result = baseline.compare(
        None, candidate, target, folds, candidate_invalid_mask=invalid
    )

    assert result.oof_samples == 60
    assert len(calls) == 4
    for base_call, challenge_call in zip(calls[::2], calls[1::2]):
        assert torch.equal(torch.from_numpy(base_call[0]), torch.from_numpy(challenge_call[0]))
        assert torch.equal(torch.from_numpy(base_call[1]), torch.from_numpy(challenge_call[1]))
    assert calls[0][1][0] == 80
    assert calls[2][1][0] == 120
