"""RD-Agent style formula proposal adapter for quant_w1ngman.

The upstream RD-Agent project uses an LLM-driven propose/evaluate/feedback loop.
This adapter keeps that loop while emitting quant_w1ngman's native RPN token lists.
Only vocabulary metadata and anonymised formula feedback leave the machine; raw
market rows, targets, file paths and timestamps are never included in prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .vocab import FORMULA_VOCAB
from .vm import StackVM, validate_formula_structure


SILICONFLOW_CHAT_URL = "https://api.siliconflow.cn/v1/chat/completions"


class RDFormulaError(RuntimeError):
    """Raised when the RD-Agent proposal service cannot produce valid formulas."""


@dataclass(frozen=True)
class RDFormulaLimits:
    max_tokens: int = 48
    max_depth: int = 7
    max_operators: int = 12
    api_timeout_seconds: float = 90.0
    round_timeout_seconds: float = 180.0
    api_retries: int = 2

    def __post_init__(self) -> None:
        if not 32 <= self.max_tokens <= 64:
            raise ValueError("RD-Agent max_tokens must be in [32, 64]")
        if not 5 <= self.max_depth <= 9:
            raise ValueError("RD-Agent max_depth must be in [5, 9]")
        if not 8 <= self.max_operators <= 16:
            raise ValueError("RD-Agent max_operators must be in [8, 16]")
        if self.api_retries < 0:
            raise ValueError("RD-Agent api_retries must be non-negative")


@dataclass(frozen=True)
class FormulaCandidate:
    tokens: list[int]
    token_names: list[str]
    rationale: str = ""


class RDFormulaGenerator:
    """Generate constrained RPN formulas through SiliconFlow DeepSeek."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        model: str,
        limits: RDFormulaLimits,
    ) -> None:
        self.project_root = Path(project_root)
        self.model = model
        self.limits = limits
        self.vm = StackVM()
        self.names = FORMULA_VOCAB.token_names
        self.name_to_id = {name: idx for idx, name in enumerate(self.names)}
        self.operator_rows = [
            (self.names[token_id], arity)
            for token_id, arity in sorted(self.vm.arity_map.items())
        ]

    def _api_key(self) -> str:
        key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if key:
            return key

        settings_path = self.project_root / "web_settings.json"
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            settings = {}
        provider = str(settings.get("ai_provider") or "").strip().lower()
        if provider == "siliconflow":
            key = str(settings.get("ai_api_key") or "").strip()
        if not key:
            raise RDFormulaError(
                "未配置硅基流动 API Key。请在网页 AI 设置中选择“硅基流动”并保存 Key，"
                "或设置环境变量 SILICONFLOW_API_KEY。"
            )
        return key

    def _system_prompt(self) -> str:
        features = ", ".join(FORMULA_VOCAB.feature_names)
        operators = ", ".join(f"{name}/{arity}" for name, arity in self.operator_rows)
        return (
            "You are the factor-proposal component of RD-Agent, adapted for quant_w1ngman. "
            "Propose diverse quantitative factor expressions in reverse Polish notation (RPN). "
            "Use only the exact feature and operator names supplied below. Never emit Python, "
            "SQL, file paths, data requests, executable code, or unknown identifiers.\n"
            f"FEATURES: {features}\n"
            f"OPERATORS(name/arity): {operators}\n"
            "RPN rule: features push one value; an operator pops its arity values and pushes one. "
            "A valid formula finishes with exactly one value.\n"
            f"Hard limits: at most {self.limits.max_tokens} tokens, "
            f"expression depth at most {self.limits.max_depth}, "
            f"operators at most {self.limits.max_operators}.\n"
            "Keep each rationale under 12 words. Return JSON only: "
            "{\"formulas\":[{\"tokens\":[\"RET\",\"TS_MEAN_5\"],"
            "\"rationale\":\"short explanation\"}]}."
        )

    @staticmethod
    def _sanitise_feedback(feedback: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Allow formula structure/rank only; block all raw-data-shaped fields."""
        clean: list[dict[str, Any]] = []
        for item in (feedback or [])[:12]:
            formula = item.get("formula")
            if not isinstance(formula, list):
                continue
            clean.append({
                "formula": [str(x) for x in formula[:64]],
                "rank": int(item.get("rank", len(clean) + 1)),
                "status": str(item.get("status", "elite"))[:32],
            })
        return clean

    def _user_prompt(
        self,
        *,
        round_index: int,
        count: int,
        feedback: list[dict[str, Any]] | None,
    ) -> str:
        safe_feedback = self._sanitise_feedback(feedback)
        return json.dumps(
            {
                "round": round_index,
                "requested_formula_count": count,
                "previous_elites": safe_feedback,
                "instructions": (
                    "Generate structurally different candidates. Previous elites are provided "
                    "only as anonymised structural feedback; improve or diversify them without "
                    "requesting or inferring raw market data."
                ),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise RDFormulaError("硅基流动返回内容不是 JSON")
            value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise RDFormulaError("硅基流动返回 JSON 顶层必须是对象")
        return value

    def _validate(self, raw_tokens: Any) -> FormulaCandidate | None:
        if not isinstance(raw_tokens, list) or not raw_tokens:
            return None
        token_names = [str(name).strip().upper() for name in raw_tokens]
        if len(token_names) > self.limits.max_tokens:
            return None
        if any(name not in self.name_to_id for name in token_names):
            return None

        token_ids = [self.name_to_id[name] for name in token_names]
        depths: list[int] = []
        op_count = 0
        for token_id in token_ids:
            if token_id < FORMULA_VOCAB.operator_offset:
                depths.append(1)
                continue
            op_count += 1
            if op_count > self.limits.max_operators:
                return None
            arity = self.vm.arity_map.get(token_id)
            if arity is None or len(depths) < arity:
                return None
            args = [depths.pop() for _ in range(arity)]
            depth = 1 + max(args)
            if depth > self.limits.max_depth:
                return None
            depths.append(depth)

        if len(depths) != 1 or op_count == 0:
            return None
        if validate_formula_structure(token_ids, self.names):
            return None
        return FormulaCandidate(tokens=token_ids, token_names=token_names)

    def generate(
        self,
        *,
        round_index: int,
        count: int,
        feedback: list[dict[str, Any]] | None = None,
    ) -> list[FormulaCandidate]:
        started = time.monotonic()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_prompt(
                        round_index=round_index, count=count, feedback=feedback
                    ),
                },
            ],
            "temperature": 0.75,
            "top_p": 0.9,
            # Formula proposals need compact JSON, not a long chain of thought.
            # Disabling thinking materially reduces first-token latency on V4 Pro.
            "enable_thinking": False,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }
        api_key = self._api_key()
        deadline = started + self.limits.round_timeout_seconds
        body: dict[str, Any] | None = None
        errors: list[str] = []
        max_attempts = self.limits.api_retries + 1

        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                break
            timeout = min(self.limits.api_timeout_seconds, remaining)
            request = Request(
                SILICONFLOW_CHAT_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                errors.append(f"第{attempt}次 HTTP {exc.code}: {detail}")
                # Some deployments may not expose the thinking switch yet. Retry
                # once without it instead of failing an otherwise valid model call.
                if exc.code == 400 and "enable_thinking" in payload:
                    payload.pop("enable_thinking", None)
                elif exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise RDFormulaError(
                        f"硅基流动请求失败 HTTP {exc.code}: {detail}"
                    ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                errors.append(f"第{attempt}次: {type(exc).__name__}: {exc}")

            if attempt < max_attempts:
                remaining = deadline - time.monotonic()
                if remaining <= 1.0:
                    break
                time.sleep(min(2.0 * attempt, max(0.0, remaining - 1.0)))

        if body is None:
            detail = "; ".join(errors) or "单轮时间预算已耗尽"
            raise RDFormulaError(
                f"硅基流动请求在 {max_attempts} 次尝试内未成功: {detail}"
            )

        if time.monotonic() - started > self.limits.round_timeout_seconds:
            raise RDFormulaError("RD-Agent 单轮提案超过时间上限")
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RDFormulaError("硅基流动响应缺少 choices[0].message.content") from exc

        result = self._extract_json(str(content))
        rows = result.get("formulas")
        if not isinstance(rows, list):
            raise RDFormulaError("硅基流动响应缺少 formulas 数组")

        candidates: list[FormulaCandidate] = []
        seen: set[tuple[int, ...]] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate = self._validate(row.get("tokens"))
            if candidate is None:
                continue
            key = tuple(candidate.tokens)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                FormulaCandidate(
                    tokens=candidate.tokens,
                    token_names=candidate.token_names,
                    rationale=str(row.get("rationale") or "")[:300],
                )
            )
            if len(candidates) >= count:
                break
        if not candidates:
            raise RDFormulaError("RD-Agent 本轮没有生成满足约束的公式")
        return candidates
