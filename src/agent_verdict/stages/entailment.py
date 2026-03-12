"""
NLI Entailment stage.

Uses a Natural Language Inference model to check whether the
task context actually supports (entails) the agent's answer.
Catches hallucinations with a completely different model —
breaks the "same brain" problem.

Requires: pip install transformers torch
Falls back to LLM-based check if not installed.
"""
from __future__ import annotations

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import LLMMessage, Verdict, VerdictConfig

from .base import DATA_BOUNDARY_INSTRUCTION, Stage, sanitize_for_prompt

# DeBERTa-v3 fine-tuned on NLI — fast, accurate, ~180MB
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

FALLBACK_PROMPT = """\
Does the premise logically support or entail the hypothesis? \
Consider only whether the premise provides evidence for the hypothesis.

{data_boundary}

{premise}

{hypothesis}

Reply with exactly one of: "entailment", "neutral", or "contradiction"."""

# Minimum entailment probability to pass
DEFAULT_ENTAILMENT_THRESHOLD = 0.3


class EntailmentStage(Stage):
    """Check if the task context entails the agent's answer using NLI."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    ):
        self._model_name = model_name
        self._entailment_threshold = entailment_threshold
        self._classifier = None
        self._use_nli = True
        try:
            from transformers import pipeline as hf_pipeline
            self._classifier = hf_pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
            )
        except (ImportError, OSError):
            self._use_nli = False

    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        if self._use_nli and self._classifier is not None:
            scores = self._classify(task_context, str(verdict.result))
            method = f"NLI ({self._model_name})"
        else:
            scores = await self._llm_entailment(llm, task_context, verdict)
            method = "llm-based"

        entailment_score = scores.get("entailment", 0.0)
        contradiction_score = scores.get("contradiction", 0.0)

        updates: dict = {}

        if contradiction_score > 0.5:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Answer contradicts task context "
                f"(contradiction: {contradiction_score:.2f}, {method})"
            )
        elif entailment_score < self._entailment_threshold:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Weak entailment {entailment_score:.2f} below threshold "
                f"{self._entailment_threshold} ({method})"
            )

        # Adjust confidence based on entailment
        entailment_factor = entailment_score - contradiction_score
        adjusted = verdict.confidence * (0.5 + 0.5 * max(0, entailment_factor))
        updates["confidence"] = round(max(0.0, min(1.0, adjusted)), 4)
        updates["confidence_reason"] = (
            f"Entailment: {entailment_score:.2f}, "
            f"contradiction: {contradiction_score:.2f} ({method}). "
            + verdict.confidence_reason
        )

        return verdict.model_copy(update=updates)

    def _classify(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Run NLI classification, return {entailment, neutral, contradiction} scores."""
        result = self._classifier(f"{premise}</s></s>{hypothesis}")
        scores = {}
        for item in result[0]:
            label = item["label"].lower()
            # Handle different label formats
            if "entail" in label:
                scores["entailment"] = item["score"]
            elif "contra" in label:
                scores["contradiction"] = item["score"]
            else:
                scores["neutral"] = item["score"]
        return scores

    async def _llm_entailment(
        self, llm: LLMProvider, task_context: str, verdict: Verdict
    ) -> dict[str, float]:
        prompt = FALLBACK_PROMPT.format(
            data_boundary=DATA_BOUNDARY_INSTRUCTION,
            premise=sanitize_for_prompt(task_context, "premise"),
            hypothesis=sanitize_for_prompt(verdict.result, "hypothesis"),
        )
        response = await llm.complete([LLMMessage(role="user", content=prompt)])
        text = response.content.strip().lower()
        if "entailment" in text:
            return {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1}
        elif "contradiction" in text:
            return {"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8}
        else:
            return {"entailment": 0.3, "neutral": 0.5, "contradiction": 0.2}
