import pytest

from agent_verdict import Verdict, VerdictConfig
from agent_verdict.stages.confidence import ConfidenceStage
from conftest import MockLLMProvider


@pytest.fixture
def stage():
    return ConfidenceStage()


class TestConfidenceStage:
    async def test_high_confidence_passes(self, stage):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.85,
                    "confidence_reason": "Strong",
                    "context_relevance": 0.9,
                    "justification": "Good answer",
                }
            ]
        )
        verdict = Verdict(result="test answer")
        result = await stage.run(verdict, llm, "test task", VerdictConfig())
        assert result.confidence == 0.85
        assert result.context_relevance == 0.9
        assert result.dropped is False

    async def test_low_confidence_drops(self, stage):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.2,
                    "confidence_reason": "Weak",
                    "context_relevance": 0.8,
                    "justification": "Guess",
                }
            ]
        )
        verdict = Verdict(result="test answer")
        result = await stage.run(verdict, llm, "test task", VerdictConfig())
        assert result.dropped is True
        assert "Confidence" in result.drop_reason

    async def test_low_relevance_drops(self, stage):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.8,
                    "confidence_reason": "Confident",
                    "context_relevance": 0.1,
                    "justification": "Off topic",
                }
            ]
        )
        verdict = Verdict(result="test answer")
        result = await stage.run(verdict, llm, "test task", VerdictConfig())
        assert result.dropped is True
        assert "Relevance" in result.drop_reason

    async def test_malformed_response_conservative(self, stage):
        llm = MockLLMProvider(responses=[])
        # Will return "{}" which has no fields → conservative defaults
        verdict = Verdict(result="test")
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.confidence == 0.0
        assert result.dropped is True

    async def test_clamps_values(self, stage):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 5.0,
                    "context_relevance": -1.0,
                    "confidence_reason": "x",
                    "justification": "y",
                }
            ]
        )
        verdict = Verdict(result="test")
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.confidence == 1.0
        assert result.context_relevance == 0.0
