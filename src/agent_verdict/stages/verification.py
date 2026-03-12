from __future__ import annotations

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import LLMMessage, Verdict, VerdictConfig

from .base import Stage, parse_llm_json

VERIFICATION_PROMPT = """\
You are independently verifying an agent's result. Without being influenced by the \
existing answer, consider the task and determine if you would arrive at the same conclusion.

Task context: {task_context}
Agent result: {result}
Agent justification: {justification}

Respond with JSON only:
{{
  "verified": <true/false>,
  "verification_reason": "<your independent reasoning>",
  "adjusted_confidence": <float 0.0-1.0>
}}"""

CONSERVATIVE_DEFAULTS = {
    "verified": False,
    "verification_reason": "Failed to parse LLM verification",
    "adjusted_confidence": 0.0,
}


class VerificationStage(Stage):
    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        prompt = VERIFICATION_PROMPT.format(
            task_context=task_context,
            result=verdict.result,
            justification=verdict.justification,
        )
        response = await llm.complete([LLMMessage(role="user", content=prompt)])
        data = parse_llm_json(response.content, CONSERVATIVE_DEFAULTS)

        verified = bool(data.get("verified", False))
        adjusted = float(data.get("adjusted_confidence", verdict.confidence))
        adjusted = max(0.0, min(1.0, adjusted))

        updates: dict = {
            "confidence": adjusted,
            "confidence_reason": str(
                data.get("verification_reason", verdict.confidence_reason)
            ),
        }

        if not verified:
            updates["dropped"] = True
            updates["drop_reason"] = "Independent verification failed"

        return verdict.model_copy(update=updates)
