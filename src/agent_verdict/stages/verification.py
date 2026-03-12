from __future__ import annotations

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import (
    LLMMessage,
    Verdict,
    VerdictConfig,
    VerificationOutput,
)

from .base import Stage

VERIFICATION_PROMPT = """\
You are independently verifying an agent's result. Without being influenced by the \
existing answer, consider the task and determine if you would arrive at the same conclusion.

Task context: {task_context}
Agent result: {result}
Agent justification: {justification}

Determine if the result is verified (true/false), provide your independent reasoning, \
and give an adjusted confidence score (0.0-1.0)."""


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
        data = await llm.complete_structured(
            [LLMMessage(role="user", content=prompt)],
            VerificationOutput,
        )

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
