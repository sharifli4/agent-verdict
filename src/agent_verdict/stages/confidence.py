from __future__ import annotations

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import LLMMessage, Verdict, VerdictConfig

from .base import Stage, parse_llm_json

CONFIDENCE_PROMPT = """\
You are evaluating an agent's output. Score its confidence and relevance to the task.

Task context: {task_context}
Agent result: {result}

Respond with JSON only:
{{
  "confidence": <float 0.0-1.0>,
  "confidence_reason": "<why this confidence level>",
  "context_relevance": <float 0.0-1.0>,
  "justification": "<brief justification of the result>"
}}"""

CONSERVATIVE_DEFAULTS = {
    "confidence": 0.0,
    "confidence_reason": "Failed to parse LLM evaluation",
    "context_relevance": 0.0,
    "justification": "",
}


class ConfidenceStage(Stage):
    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        prompt = CONFIDENCE_PROMPT.format(
            task_context=task_context, result=verdict.result
        )
        response = await llm.complete([LLMMessage(role="user", content=prompt)])
        data = parse_llm_json(response.content, CONSERVATIVE_DEFAULTS)

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
