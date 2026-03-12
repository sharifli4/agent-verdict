import pytest

from agent_verdict import Verdict, VerdictConfig
from agent_verdict.stages.semantic_similarity import SemanticSimilarityStage
from conftest import MockLLMProvider


class FakeSimilarityProvider(MockLLMProvider):
    """Returns a numeric similarity score."""

    def __init__(self, score: float):
        from agent_verdict import LLMResponse

        self._score = score
        self._call_index = 0
        self.calls = []

    async def complete(self, messages):
        from agent_verdict import LLMResponse

        self.calls.append(messages)
        return LLMResponse(content=str(self._score))


class TestSemanticSimilarityStage:
    async def test_llm_fallback_high_relevance(self):
        # Force LLM fallback by not having sentence-transformers
        stage = SemanticSimilarityStage()
        stage._use_embeddings = False
        stage._model = None
        llm = FakeSimilarityProvider(score=0.85)
        verdict = Verdict(result="SQL injection found", confidence=0.8)
        result = await stage.run(verdict, llm, "find security bugs", VerdictConfig())
        assert result.dropped is False
        assert result.context_relevance == 0.85

    async def test_llm_fallback_low_relevance_drops(self):
        stage = SemanticSimilarityStage()
        stage._use_embeddings = False
        stage._model = None
        llm = FakeSimilarityProvider(score=0.1)
        verdict = Verdict(result="nice weather today", confidence=0.8)
        result = await stage.run(verdict, llm, "find security bugs", VerdictConfig())
        assert result.dropped is True
        assert "relevance" in result.drop_reason.lower()

    async def test_clamps_values(self):
        stage = SemanticSimilarityStage()
        stage._use_embeddings = False
        stage._model = None
        llm = FakeSimilarityProvider(score=5.0)
        verdict = Verdict(result="test", confidence=0.8)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.context_relevance <= 1.0

    async def test_invalid_llm_response(self):
        stage = SemanticSimilarityStage()
        stage._use_embeddings = False
        stage._model = None
        llm = MockLLMProvider(responses=[])
        verdict = Verdict(result="test", confidence=0.8)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        # Empty response → 0.0 → dropped
        assert result.context_relevance == 0.0
        assert result.dropped is True
