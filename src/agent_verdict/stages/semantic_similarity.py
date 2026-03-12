"""
Semantic Similarity stage.

Uses sentence embeddings to check if the agent's answer is
actually about the task. Catches irrelevant / off-topic answers
without any LLM call — just cosine similarity between embeddings.

Requires: pip install sentence-transformers
Falls back to LLM-based check if sentence-transformers is not installed.
"""
from __future__ import annotations

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import LLMMessage, Verdict, VerdictConfig

from .base import DATA_BOUNDARY_INSTRUCTION, Stage, sanitize_for_prompt

# Light model, ~80MB, fast on CPU
DEFAULT_MODEL = "all-MiniLM-L6-v2"

FALLBACK_PROMPT = """\
On a scale of 0.0 to 1.0, how semantically relevant is the following answer \
to the given task? Consider only topical relevance, not correctness.

{data_boundary}

{task_context}

{result}

Reply with a single number between 0.0 and 1.0."""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticSimilarityStage(Stage):
    """Check topical relevance using embeddings or LLM fallback."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._use_embeddings = True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError:
            self._use_embeddings = False

    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        if self._use_embeddings and self._model is not None:
            similarity = self._compute_embedding_similarity(
                task_context, str(verdict.result)
            )
            method = f"embedding ({self._model_name})"
        else:
            similarity = await self._llm_similarity(llm, task_context, verdict)
            method = "llm-based"

        updates: dict = {
            "context_relevance": round(max(0.0, min(1.0, similarity)), 4),
        }

        if similarity < config.relevance_threshold:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Semantic relevance {similarity:.2f} below threshold "
                f"{config.relevance_threshold} ({method})"
            )

        return verdict.model_copy(update=updates)

    def _compute_embedding_similarity(self, task: str, result: str) -> float:
        embeddings = self._model.encode([task, result])
        return float(_cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist()))

    async def _llm_similarity(
        self, llm: LLMProvider, task_context: str, verdict: Verdict
    ) -> float:
        prompt = FALLBACK_PROMPT.format(
            data_boundary=DATA_BOUNDARY_INSTRUCTION,
            task_context=sanitize_for_prompt(task_context, "task_context"),
            result=sanitize_for_prompt(verdict.result, "agent_result"),
        )
        response = await llm.complete([LLMMessage(role="user", content=prompt)])
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.0
