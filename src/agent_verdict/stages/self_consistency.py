"""
Self-Consistency stage (Wang et al. 2022).

Asks the LLM to independently solve the same task N times,
then measures how much the agent's answer agrees with the
sampled answers. High agreement = high confidence.
No judge bias — it's pure voting.
"""
from __future__ import annotations

import asyncio

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import LLMMessage, Verdict, VerdictConfig

from .base import Stage

REANSWER_PROMPT = """\
Answer the following task independently. Be concise and direct.

Task: {task_context}

Provide your answer in one or two sentences."""

AGREEMENT_PROMPT = """\
Do these two answers agree on the same core conclusion? \
Ignore wording differences, focus on whether the substance matches.

Answer A: {answer_a}
Answer B: {answer_b}

Reply with exactly "yes" or "no"."""

DEFAULT_SAMPLES = 3


class SelfConsistencyStage(Stage):
    """Sample N independent answers, measure agreement with the agent's result."""

    def __init__(self, num_samples: int = DEFAULT_SAMPLES):
        self.num_samples = num_samples

    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        # Step 1: Generate N independent answers in parallel
        prompt = REANSWER_PROMPT.format(task_context=task_context)
        sample_tasks = [
            llm.complete([LLMMessage(role="user", content=prompt)])
            for _ in range(self.num_samples)
        ]
        samples = await asyncio.gather(*sample_tasks)

        # Step 2: Check agreement of each sample with the agent's answer
        agree_tasks = []
        for sample in samples:
            agree_prompt = AGREEMENT_PROMPT.format(
                answer_a=str(verdict.result),
                answer_b=sample.content,
            )
            agree_tasks.append(
                llm.complete([LLMMessage(role="user", content=agree_prompt)])
            )
        agreements = await asyncio.gather(*agree_tasks)

        # Step 3: Count agreements
        agree_count = sum(
            1 for a in agreements if a.content.strip().lower().startswith("yes")
        )
        agreement_rate = agree_count / self.num_samples if self.num_samples > 0 else 0.0

        # Blend with existing confidence: take the lower of the two
        blended = min(verdict.confidence, agreement_rate) if verdict.confidence > 0 else agreement_rate

        updates: dict = {
            "confidence": round(blended, 4),
            "confidence_reason": (
                f"Self-consistency: {agree_count}/{self.num_samples} independent answers agree "
                f"(agreement rate: {agreement_rate:.0%}). "
                + verdict.confidence_reason
            ),
        }

        if agreement_rate < config.confidence_threshold:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Self-consistency {agreement_rate:.0%} below threshold "
                f"{config.confidence_threshold:.0%} "
                f"({agree_count}/{self.num_samples} answers agree)"
            )

        return verdict.model_copy(update=updates)
