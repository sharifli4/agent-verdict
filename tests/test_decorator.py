import pytest

from agent_verdict import DroppedResultError, Verdict, VerdictConfig, verdict
from conftest import MockLLMProvider


class TestDecorator:
    def test_sync_function_passes(self):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.9,
                    "confidence_reason": "Strong",
                    "context_relevance": 0.9,
                    "justification": "Good",
                },
                {
                    "verified": True,
                    "verification_reason": "Confirmed",
                    "adjusted_confidence": 0.88,
                },
                {"counter_argument": "Edge case"},
                {"defense": "Handled", "defended": True},
            ]
        )

        @verdict(llm=llm, task_context="test")
        def my_agent(x: int) -> str:
            return f"result {x}"

        result = my_agent(42)
        assert isinstance(result, Verdict)
        assert result.result == "result 42"
        assert result.dropped is False

    def test_sync_function_raises_on_drop(self):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.1,
                    "confidence_reason": "Bad",
                    "context_relevance": 0.1,
                    "justification": "Weak",
                }
            ]
        )

        @verdict(llm=llm, task_context="test")
        def my_agent() -> str:
            return "bad result"

        with pytest.raises(DroppedResultError) as exc_info:
            my_agent()
        assert exc_info.value.verdict.dropped is True

    def test_no_raise_on_drop(self):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.1,
                    "confidence_reason": "Bad",
                    "context_relevance": 0.1,
                    "justification": "Weak",
                }
            ]
        )

        @verdict(llm=llm, task_context="test", raise_on_drop=False)
        def my_agent() -> str:
            return "bad result"

        result = my_agent()
        assert isinstance(result, Verdict)
        assert result.dropped is True

    async def test_async_function_passes(self):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.9,
                    "confidence_reason": "Strong",
                    "context_relevance": 0.9,
                    "justification": "Good",
                },
                {
                    "verified": True,
                    "verification_reason": "Confirmed",
                    "adjusted_confidence": 0.88,
                },
                {"counter_argument": "Edge case"},
                {"defense": "Handled", "defended": True},
            ]
        )

        @verdict(llm=llm, task_context="test")
        async def my_agent(x: int) -> str:
            return f"result {x}"

        result = await my_agent(42)
        assert isinstance(result, Verdict)
        assert result.result == "result 42"
        assert result.dropped is False

    async def test_async_function_raises_on_drop(self):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.1,
                    "confidence_reason": "Bad",
                    "context_relevance": 0.1,
                    "justification": "Weak",
                }
            ]
        )

        @verdict(llm=llm, task_context="test")
        async def my_agent() -> str:
            return "bad result"

        with pytest.raises(DroppedResultError):
            await my_agent()
