"""
Logprob Calibration stage.

Asks the LLM to re-state its answer and reads token-level
log probabilities. High entropy = the model is uncertain.
Low entropy = the model is confident in its own words.

This is a real confidence signal from the model's internals,
not a self-reported "I'm 90% sure."

Requires the provider to support logprobs (OpenAI does,
Anthropic does not currently). Falls back to LLM-based
confidence if logprobs are unavailable.
"""
from __future__ import annotations

import math

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import LLMMessage, Verdict, VerdictConfig

from .base import DATA_BOUNDARY_INSTRUCTION, Stage, sanitize_for_prompt

RESTATE_PROMPT = """\
Restate the following answer in your own words. Be concise.

{data_boundary}

{task_context}

{result}

Your restatement:"""


class LogprobStage(Stage):
    """Measure model certainty via token log probabilities."""

    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        prompt = RESTATE_PROMPT.format(
            data_boundary=DATA_BOUNDARY_INSTRUCTION,
            task_context=sanitize_for_prompt(task_context, "task_context"),
            result=sanitize_for_prompt(verdict.result, "agent_result"),
        )

        logprob_confidence = await self._get_logprob_confidence(
            llm, prompt
        )

        if logprob_confidence is not None:
            # Blend: average of existing confidence and logprob confidence
            blended = (verdict.confidence + logprob_confidence) / 2
            method = "logprob"
        else:
            blended = verdict.confidence
            method = "logprob unavailable, keeping existing"

        lp_str = f"{logprob_confidence:.2f}" if logprob_confidence is not None else "N/A"
        updates: dict = {
            "confidence": round(max(0.0, min(1.0, blended)), 4),
            "confidence_reason": (
                f"Logprob confidence: {lp_str} ({method}). "
                + verdict.confidence_reason
            ),
        }

        if blended < config.confidence_threshold:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Logprob-blended confidence {blended:.2f} below threshold "
                f"{config.confidence_threshold}"
            )

        return verdict.model_copy(update=updates)

    async def _get_logprob_confidence(
        self, llm: LLMProvider, prompt: str
    ) -> float | None:
        """Try to get logprob-based confidence. Returns None if not supported."""
        # Check if this is an OpenAI provider that supports logprobs
        if not hasattr(llm, "client"):
            return None

        try:
            import openai

            if not isinstance(llm.client, (openai.AsyncOpenAI,)):
                return None
        except (ImportError, AttributeError):
            return None

        try:
            response = await llm.client.chat.completions.create(
                model=llm.model,
                max_tokens=llm.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                top_logprobs=1,
            )

            logprobs = response.choices[0].logprobs
            if logprobs is None or not logprobs.content:
                return None

            # Average token probability = exp(mean logprob)
            token_logprobs = [t.logprob for t in logprobs.content]
            if not token_logprobs:
                return None

            mean_logprob = sum(token_logprobs) / len(token_logprobs)
            # Convert to 0-1 scale. logprob of 0 = certain, -inf = uncertain
            # exp(-0) = 1.0, exp(-3) ≈ 0.05
            confidence = math.exp(mean_logprob)
            return max(0.0, min(1.0, confidence))

        except Exception:
            return None
