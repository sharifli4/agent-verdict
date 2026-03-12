import json

import pytest

from agent_verdict import Verdict, VerdictConfig
from agent_verdict.stages.self_consistency import SelfConsistencyStage
from conftest import MockLLMProvider


class FakeConsistencyProvider(MockLLMProvider):
    """Returns N sample answers, then N agreement checks."""

    def __init__(self, sample_answers: list[str], agreements: list[bool]):
        from agent_verdict import LLMResponse

        self._sample_answers = sample_answers
        self._agreements = agreements
        self._call_index = 0
        self.calls = []

    async def complete(self, messages):
        from agent_verdict import LLMMessage, LLMResponse

        self.calls.append(messages)
        idx = self._call_index
        self._call_index += 1

        n = len(self._sample_answers)
        if idx < n:
            return LLMResponse(content=self._sample_answers[idx])
        else:
            agree_idx = idx - n
            if agree_idx < len(self._agreements):
                answer = "yes" if self._agreements[agree_idx] else "no"
                return LLMResponse(content=answer)
            return LLMResponse(content="no")


class TestSelfConsistencyStage:
    async def test_high_agreement_passes(self):
        stage = SelfConsistencyStage(num_samples=3)
        llm = FakeConsistencyProvider(
            sample_answers=["SQL injection on line 14", "SQLi in login", "SQL injection found"],
            agreements=[True, True, True],
        )
        verdict = Verdict(result="SQL injection on line 14", confidence=0.8)
        result = await stage.run(verdict, llm, "find bugs", VerdictConfig())
        assert result.dropped is False
        assert result.confidence > 0
        assert "3/3" in result.confidence_reason

    async def test_low_agreement_drops(self):
        stage = SelfConsistencyStage(num_samples=3)
        llm = FakeConsistencyProvider(
            sample_answers=["XSS attack", "no issues found", "buffer overflow"],
            agreements=[False, False, False],
        )
        verdict = Verdict(result="SQL injection", confidence=0.8)
        result = await stage.run(verdict, llm, "find bugs", VerdictConfig())
        assert result.dropped is True
        assert "0/3" in result.drop_reason

    async def test_partial_agreement(self):
        stage = SelfConsistencyStage(num_samples=4)
        llm = FakeConsistencyProvider(
            sample_answers=["a", "b", "c", "d"],
            agreements=[True, False, True, False],
        )
        verdict = Verdict(result="test", confidence=0.8)
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert "2/4" in result.confidence_reason

    async def test_makes_2n_calls(self):
        stage = SelfConsistencyStage(num_samples=3)
        llm = FakeConsistencyProvider(
            sample_answers=["a", "b", "c"],
            agreements=[True, True, True],
        )
        verdict = Verdict(result="test", confidence=0.8)
        await stage.run(verdict, llm, "task", VerdictConfig())
        # 3 sample calls + 3 agreement calls
        assert len(llm.calls) == 6
