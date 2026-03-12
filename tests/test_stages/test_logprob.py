import pytest

from agent_verdict import Verdict, VerdictConfig
from agent_verdict.stages.logprob import LogprobStage
from conftest import MockLLMProvider


class TestLogprobStage:
    async def test_no_logprob_support_keeps_existing(self):
        """When provider doesn't support logprobs, confidence stays the same."""
        stage = LogprobStage()
        llm = MockLLMProvider(responses=[])
        verdict = Verdict(result="answer", confidence=0.8)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        # MockLLMProvider has no .client, so logprobs unavailable
        # Should keep existing confidence
        assert result.confidence == 0.8
        assert "unavailable" in result.confidence_reason.lower()
        assert result.dropped is False

    async def test_low_existing_confidence_still_drops(self):
        """Even without logprob data, if existing confidence is below threshold, drop."""
        stage = LogprobStage()
        llm = MockLLMProvider(responses=[])
        verdict = Verdict(result="answer", confidence=0.3)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.dropped is True
        assert "below threshold" in result.drop_reason.lower()
