from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION

from .analysis_unit import AnalysisUnit
from .config import ResearchConfig
from .factor_artifact import (
    data_fingerprint,
    factor_identity,
    feature_implementation_version,
    research_protocol_version,
)
from .knowledge_forest import KnowledgeForest
from .lgbm_baseline import FixedLightGBMBaseline
from .retention_policy import RetentionPolicy
from .schemas import CandidateAssessment
from .sota_store import SOTAStore
from .validation_unit import ValidationUnit


@dataclass
class RoundOutcome:
    assessments: list[CandidateAssessment]
    accepted: list[CandidateAssessment]
    duplicate_count: int
    sota_status: dict
    analysis: dict

    @property
    def champion(self) -> CandidateAssessment | None:
        return max(self.accepted, key=lambda row: row.decision.ranking_score, default=None)


class FactorResearchLoop:
    """Local authority for validation, SOTA membership and research memory."""

    def __init__(
        self,
        *,
        data_manager,
        vm,
        generator,
        folds: list[dict],
        periods_per_year: int,
        symbol: str,
        config: ResearchConfig | None = None,
        project_root: Path | None = None,
    ):
        self.config = config or ResearchConfig()
        self.data_manager = data_manager
        self.vm = vm
        self.generator = generator
        self.folds = folds
        self.symbol = symbol or "multi_symbol"
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.implementation_version = (
            feature_implementation_version(self.project_root)
            if self.config.feature_implementation_version == "auto"
            else self.config.feature_implementation_version
        )
        self.data_fingerprint = data_fingerprint(data_manager)
        self.validation_protocol = research_protocol_version(
            self.config,
            self.project_root,
            n_folds=len(folds) + 1,
            label_horizon=2,
        )
        self.store = SOTAStore(self.config.for_symbol(self.symbol))
        self.knowledge = KnowledgeForest(self.store.connect)
        raw = getattr(data_manager, "raw_dict", None) or {}
        self.validation = ValidationUnit(
            self.config,
            periods_per_year,
            timestamps=raw.get("time"),
            symbols=getattr(data_manager, "symbols", None),
            close=raw.get("close"),
        )
        self.lgbm = FixedLightGBMBaseline(self.config)
        self.retention = RetentionPolicy(self.config)
        self.analysis = AnalysisUnit(self.config, generator)

    def guidance(self) -> dict:
        return self.knowledge.guidance()

    def status(self) -> dict:
        status = self.store.status(self.data_fingerprint, self.validation_protocol)
        status.update({
            "data_fingerprint": self.data_fingerprint,
            "validation_protocol": self.validation_protocol,
            "holdout_used_for_selection": False,
        })
        return status

    def _identity(self, tokens: list[int]) -> tuple[str, str]:
        return factor_identity(
            tokens,
            arity_map=self.vm.arity_map,
            vocab_version=VOCAB_VERSION,
            implementation_version=self.implementation_version,
            data_fingerprint_value=self.data_fingerprint,
            validation_protocol=self.validation_protocol,
        )

    @staticmethod
    def _operator_count(tokens: list[int]) -> int:
        return sum(int(token) >= FORMULA_VOCAB.operator_offset for token in tokens)

    def assess_round(self, round_index: int, candidates, eval_results: list[dict]) -> RoundOutcome:
        current_matrix, current_rows = self.store.matrix(
            self.data_fingerprint, self.validation_protocol
        )
        target = self.data_manager.target_ret.detach().cpu()
        results_by_index = {int(row.get("idx", -1)): row for row in eval_results}
        pending: list[dict] = []
        duplicate_count = 0
        seen_ids: set[str] = set()

        for index, candidate in enumerate(candidates):
            result = results_by_index.get(index)
            if not result or result.get("status") != "ok" or "res" not in result:
                status = "missing" if result is None else str(result.get("status", "invalid"))
                self.knowledge.record_failure(
                    None,
                    round_index,
                    f"formula_{status}",
                    {"index": index, "error": (result or {}).get("error")},
                )
                continue
            try:
                factor_id, canonical = self._identity(candidate.tokens)
            except (TypeError, ValueError) as exc:
                self.knowledge.record_failure(
                    None, round_index, "invalid_formula", {"index": index, "error": str(exc)}
                )
                continue
            if factor_id in seen_ids or self.store.has(factor_id):
                duplicate_count += 1
                self.knowledge.record_failure(
                    factor_id, round_index, "duplicate_formula", {"index": index}
                )
                continue
            seen_ids.add(factor_id)
            factor = result["res"].detach().cpu()
            common = min(factor.shape[1], target.shape[1])
            factor = factor[:, :common]
            invalid_mask = result.get("invalid_mask")
            if invalid_mask is not None:
                invalid_mask = invalid_mask.detach().cpu()[:, :common]
            current_folds = [
                {
                    **fold,
                    "train_end": min(int(fold["train_end"]), common),
                    "val_start": min(int(fold["val_start"]), common),
                    "val_end": min(int(fold["val_end"]), common),
                }
                for fold in self.folds
                if min(int(fold["val_end"]), common)
                > min(int(fold["val_start"]), common)
            ]
            result.update(
                factor_id=factor_id,
                canonical_formula=canonical,
                res=factor,
                invalid_mask=invalid_mask,
            )
            pending.append(
                {
                    "candidate": candidate,
                    "result": result,
                    "factor": factor,
                    "invalid_mask": invalid_mask,
                    "target": target[:, :common],
                    "folds": current_folds,
                    "relation_rows": [],
                }
            )

        accepted_items: list[dict] = []
        final_items: list[dict] = []

        def evaluate(item: dict) -> CandidateAssessment:
            factor = item["factor"]
            local_target = item["target"]
            local_matrix = current_matrix
            if local_matrix is not None and local_matrix.shape[2] != factor.shape[1]:
                local_matrix = local_matrix[:, :, : factor.shape[1]]
            validation = self.validation.evaluate(
                factor,
                local_target,
                item["folds"],
                local_matrix,
                invalid_mask=item["invalid_mask"],
            )
            if local_matrix is None:
                validation.portfolio_delta = 0.0
            else:
                baseline = self.validation.evaluate(
                    torch.nanmean(local_matrix, dim=1), local_target, item["folds"]
                ).net_sharpe
                combined = torch.cat(
                    (local_matrix, factor.unsqueeze(1)), dim=1
                )
                combined = torch.nanmean(combined, dim=1)
                validation.portfolio_delta = self.validation.evaluate(
                    combined, local_target, item["folds"]
                ).net_sharpe - baseline
            lgbm = self.lgbm.compare(
                local_matrix,
                factor,
                local_target,
                item["folds"],
                candidate_invalid_mask=item["invalid_mask"],
            )
            candidate = item["candidate"]
            hypothesis = candidate.hypothesis or candidate.rationale or "未命名公式假设"
            return CandidateAssessment(
                factor_id=item["result"]["factor_id"],
                formula_tokens=list(candidate.tokens),
                formula_decoded=" -> ".join(candidate.token_names),
                hypothesis=hypothesis,
                rationale=candidate.rationale,
                validation=validation,
                lgbm=lgbm,
                decision=self.retention.decide(validation, lgbm),
                operator_count=self._operator_count(candidate.tokens),
            )

        while pending and len(accepted_items) < self.config.max_accept_per_round:
            evaluated: list[dict] = []
            for item in pending:
                try:
                    item["assessment"] = evaluate(item)
                    evaluated.append(item)
                except Exception as exc:
                    self.knowledge.record_failure(
                        item["result"]["factor_id"],
                        round_index,
                        "validation_error",
                        {"error": f"{type(exc).__name__}: {exc}"},
                    )
            pending = evaluated
            self.retention.rank([item["assessment"] for item in pending])
            eligible = [item for item in pending if item["assessment"].decision.accepted]
            if not eligible:
                for item in pending:
                    item["relation_rows"] = list(current_rows)
                    item["relation_values"] = self.validation.correlations(
                        item["factor"], current_matrix
                    )
                final_items.extend(pending)
                pending = []
                break
            winner = max(
                eligible,
                key=lambda item: item["assessment"].decision.ranking_score,
            )
            winner["relation_rows"] = list(current_rows)
            winner["relation_values"] = self.validation.correlations(
                winner["factor"], current_matrix
            )
            accepted_items.append(winner)
            pending.remove(winner)
            accepted_factor = winner["factor"].clone()
            if winner["invalid_mask"] is not None:
                accepted_factor.masked_fill_(winner["invalid_mask"], float("nan"))
            factor_column = accepted_factor.unsqueeze(1)
            current_matrix = (
                factor_column
                if current_matrix is None
                else torch.cat((current_matrix, factor_column), dim=1)
            )
            current_rows = [
                *current_rows,
                {"factor_id": winner["assessment"].factor_id},
            ]

        if pending:
            # Re-evaluate after the final accepted factor so every remaining
            # candidate is measured against the actual updated SOTA matrix.
            for item in pending:
                try:
                    item["assessment"] = evaluate(item)
                    self.retention.rank([item["assessment"]])
                    item["assessment"].decision.accepted = False
                    item["assessment"].decision.reasons.append("同轮接纳配额限制")
                    item["relation_rows"] = list(current_rows)
                    item["relation_values"] = self.validation.correlations(
                        item["factor"], current_matrix
                    )
                    final_items.append(item)
                except Exception as exc:
                    self.knowledge.record_failure(
                        item["result"]["factor_id"], round_index,
                        "validation_error", {"error": f"{type(exc).__name__}: {exc}"},
                    )

        for item in accepted_items:
            item["assessment"].decision.accepted = True
        final_items = [*accepted_items, *final_items]
        assessments = [item["assessment"] for item in final_items]

        for item in final_items:
            row = item["assessment"]
            result = item["result"]
            hypothesis_id = self.knowledge.add_hypothesis(round_index, row.hypothesis)
            metrics = {
                **row.validation.to_dict(),
                "lgbm": row.lgbm.to_dict(),
                "decision": row.decision.to_dict(),
                "legacy_reward": float(result.get("reward", 0.0)),
                "legacy_val_score": float(result.get("val_score", 0.0)),
            }
            self.store.save_result(
                {
                    "factor_id": row.factor_id,
                    "canonical_formula": result["canonical_formula"],
                    "formula_tokens": row.formula_tokens,
                    "formula_decoded": row.formula_decoded,
                    "formula_version": 1,
                    "vocab_version": VOCAB_VERSION,
                    "feature_implementation_version": self.implementation_version,
                    "data_fingerprint": self.data_fingerprint,
                    "validation_protocol": self.validation_protocol,
                    "generator": "rd_agent",
                    "llm_model": self.generator.model,
                    "hypothesis_id": hypothesis_id,
                    "created_round": round_index,
                    "status": "accepted" if row.decision.accepted else "rejected",
                    "metrics": metrics,
                    "max_correlation": row.validation.max_correlation,
                    "ranking_score": row.decision.ranking_score,
                },
                result["res"] if row.decision.accepted else None,
                result.get("invalid_mask") if row.decision.accepted else None,
            )
            relations = item.get("relation_values", [])
            relation_rows = item["relation_rows"]
            self.knowledge.record_relations(
                row.factor_id,
                [
                    (relation_rows[index]["factor_id"], corr)
                    for index, corr in enumerate(relations[: len(relation_rows)])
                ],
            )
            self.knowledge.record_assessment(row, round_index, hypothesis_id)

        status = self.status()
        report = self.analysis.analyse(round_index, assessments, status)
        self.knowledge.record_analysis(round_index, report.to_dict())
        return RoundOutcome(
            assessments=assessments,
            accepted=[row for row in assessments if row.decision.accepted],
            duplicate_count=duplicate_count,
            sota_status=status,
            analysis=report.to_dict(),
        )
