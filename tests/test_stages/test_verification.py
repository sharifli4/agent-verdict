import pytest

from agent_verdict import Verdict, VerdictConfig
from agent_verdict.stages.verification import VerificationStage
from conftest import MockLLMProvider


@pytest.fixture
def stage():
    return VerificationStage()


class TestVerificationStage:
    async def test_verified_passes(self, stage):
        llm = MockLLMProvider(
            responses=[
                {
                    "verified": True,
                    "verification_reason": "Confirmed",
                    "adjusted_confidence": 0.9,
                }
            ]
        )
        verdict = Verdict(result="answer", confidence=0.8, justification="because")
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.confidence == 0.9
        assert result.dropped is False

    async def test_unverified_drops(self, stage):
        llm = MockLLMProvider(
            responses=[
                {
                    "verified": False,
                    "verification_reason": "Disagree",
                    "adjusted_confidence": 0.3,
                }
            ]
        )
        verdict = Verdict(result="answer", confidence=0.8, justification="because")
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.dropped is True
        assert "verification failed" in result.drop_reason.lower()

    async def test_malformed_response_drops(self, stage):
        llm = MockLLMProvider(responses=[])
        verdict = Verdict(result="answer", confidence=0.8, justification="test")
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.dropped is True
