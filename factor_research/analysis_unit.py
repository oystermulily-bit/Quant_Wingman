from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from model_core.rd_agent_generator import RDFormulaGenerator, SILICONFLOW_CHAT_URL

from .config import ResearchConfig
from .schemas import AnalysisReport, CandidateAssessment


class AnalysisUnit:
    """Aggregate-metric diagnosis only; it cannot accept factors or see market rows."""

    def __init__(self, config: ResearchConfig, generator: RDFormulaGenerator):
        self.config = config
        self.generator = generator

    @staticmethod
    def _fallback(assessments: list[CandidateAssessment], error: str = "") -> AnalysisReport:
        accepted = [row.hypothesis for row in assessments if row.decision.accepted and row.hypothesis]
        rejected = [row.hypothesis for row in assessments if not row.decision.accepted and row.hypothesis]
        common: list[str] = []
        for row in assessments:
            for reason in row.decision.reasons:
                if reason not in common:
                    common.append(reason)
        diagnosis = "；".join(common[:4]) or "本轮指标已完成本地验证"
        if error:
            diagnosis += f"；远程诊断不可用，已使用本地诊断({error[:80]})"
        return AnalysisReport(
            diagnosis=diagnosis,
            accepted_hypotheses=accepted[:8],
            rejected_hypotheses=rejected[:8],
            next_hypotheses=["优先探索低相关且低换手的不同算子结构"],
            formula_mutations=[], confidence=0.35, source="local_fallback",
        )

    @staticmethod
    def _parse(value: Any) -> AnalysisReport:
        if not isinstance(value, dict):
            raise ValueError("analysis response must be an object")

        def strings(name: str) -> list[str]:
            rows = value.get(name, [])
            return [str(item)[:300] for item in rows[:12]] if isinstance(rows, list) else []

        mutations = value.get("formula_mutations", [])
        if not isinstance(mutations, list):
            mutations = []
        return AnalysisReport(
            diagnosis=str(value.get("diagnosis", ""))[:1000],
            accepted_hypotheses=strings("accepted_hypotheses"),
            rejected_hypotheses=strings("rejected_hypotheses"),
            next_hypotheses=strings("next_hypotheses"),
            formula_mutations=[item for item in mutations[:12] if isinstance(item, dict)],
            confidence=max(0.0, min(1.0, float(value.get("confidence", 0.0)))),
        )

    def analyse(self, round_index: int, assessments: list[CandidateAssessment], sota_status: dict) -> AnalysisReport:
        if not self.config.analysis_enabled or not assessments:
            return self._fallback(assessments)
        payload = {
            "round": round_index,
            "candidates": [row.public_summary() for row in assessments],
            "sota_baseline": {
                "accepted_count": int(sota_status.get("accepted_count", 0)),
                "best_ranking_score": sota_status.get("best_ranking_score"),
            },
        }
        system = (
            "You are the diagnostic Analysis Unit of a quantitative factor research loop. "
            "You receive aggregate metrics only. Diagnose failures and propose next formula hypotheses. "
            "Never request raw prices, returns, timestamps, files or code. You cannot accept factors. "
            "Return JSON only with diagnosis, accepted_hypotheses, rejected_hypotheses, "
            "next_hypotheses, formula_mutations, confidence."
        )
        request_body = {
            "model": self.generator.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        }
        try:
            request = Request(
                SILICONFLOW_CHAT_URL,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.generator._api_key()}", "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.config.analysis_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return self._parse(self.generator._extract_json(str(content)))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            return self._fallback(assessments, f"{type(exc).__name__}: {exc}")
