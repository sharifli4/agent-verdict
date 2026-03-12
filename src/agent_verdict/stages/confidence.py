from __future__ import annotations

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import (
    ConfidenceOutput,
    LLMMessage,
    Verdict,
    VerdictConfig,
)

from .base import DATA_BOUNDARY_INSTRUCTION, Stage, sanitize_for_prompt

CONFIDENCE_PROMPT = """\
You are evaluating an agent's output. Score its confidence and relevance to the task.

{data_boundary}

{task_context}

{result}

Evaluate the confidence (0.0-1.0), context relevance (0.0-1.0), \
provide a reason for the confidence score, and a brief justification of the result."""


class ConfidenceStage(Stage):
    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        prompt = CONFIDENCE_PROMPT.format(
            data_boundary=DATA_BOUNDARY_INSTRUCTION,
            task_context=sanitize_for_prompt(task_context, "task_context"),
            result=sanitize_for_prompt(verdict.result, "agent_result"),
        )
        data = await llm.complete_structured(
            [LLMMessage(role="user", content=prompt)],
            ConfidenceOutput,
        )

        confidence = float(data.get("confidence", 0.0))
        relevance = float(data.get("context_relevance", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        relevance = max(0.0, min(1.0, relevance))

        updates: dict = {
            "confidence": confidence,
            "confidence_reason": str(data.get("confidence_reason", "")),
            "context_relevance": relevance,
            "justification": str(data.get("justification", "")),
        }

        if confidence < config.confidence_threshold:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Confidence {confidence:.2f} below threshold {config.confidence_threshold}"
            )
        elif relevance < config.relevance_threshold:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Relevance {relevance:.2f} below threshold {config.relevance_threshold}"
            )

        return verdict.model_copy(update=updates)
