import pytest

from agent_verdict import Verdict, VerdictConfig
from agent_verdict.stages.entailment import EntailmentStage
from conftest import MockLLMProvider


class FakeEntailmentProvider(MockLLMProvider):
    """Returns a specific entailment label."""

    def __init__(self, label: str):
        self._label = label
        self._call_index = 0
        self.calls = []

    async def complete(self, messages):
        from agent_verdict import LLMResponse

        self.calls.append(messages)
        return LLMResponse(content=self._label)


class TestEntailmentStage:
    async def test_entailment_passes(self):
        stage = EntailmentStage()
        stage._use_nli = False
        stage._classifier = None
        llm = FakeEntailmentProvider("entailment")
        verdict = Verdict(result="answer", confidence=0.8)
        result = await stage.run(verdict, llm, "task context", VerdictConfig())
        assert result.dropped is False
        assert "entailment" in result.confidence_reason.lower()

    async def test_contradiction_drops(self):
        stage = EntailmentStage()
        stage._use_nli = False
        stage._classifier = None
        llm = FakeEntailmentProvider("contradiction")
        verdict = Verdict(result="wrong answer", confidence=0.8)
        result = await stage.run(verdict, llm, "task context", VerdictConfig())
        assert result.dropped is True
        assert "contradict" in result.drop_reason.lower()

    async def test_neutral_with_low_threshold(self):
        stage = EntailmentStage(entailment_threshold=0.2)
        stage._use_nli = False
        stage._classifier = None
        llm = FakeEntailmentProvider("neutral")
        verdict = Verdict(result="maybe", confidence=0.8)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        # neutral gives entailment=0.3 which is >= 0.2 threshold
        assert result.dropped is False

    async def test_neutral_with_high_threshold_drops(self):
        stage = EntailmentStage(entailment_threshold=0.5)
        stage._use_nli = False
        stage._classifier = None
        llm = FakeEntailmentProvider("neutral")
        verdict = Verdict(result="maybe", confidence=0.8)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        # neutral gives entailment=0.3 which is < 0.5
        assert result.dropped is True
        assert "entailment" in result.drop_reason.lower()

    async def test_confidence_adjusted_by_entailment(self):
        stage = EntailmentStage()
        stage._use_nli = False
        stage._classifier = None
        llm = FakeEntailmentProvider("entailment")
        verdict = Verdict(result="good answer", confidence=0.9)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        # entailment boosts, so confidence should stay reasonable
        assert result.confidence > 0
        assert result.confidence <= 1.0

    async def test_contradiction_tanks_confidence(self):
        stage = EntailmentStage()
        stage._use_nli = False
        stage._classifier = None
        llm = FakeEntailmentProvider("contradiction")
        verdict = Verdict(result="wrong", confidence=0.9)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        # contradiction should lower confidence significantly
        assert result.confidence < verdict.confidence
