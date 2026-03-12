from __future__ import annotations

import json
from typing import Any, Type

import pytest
from pydantic import BaseModel

from agent_verdict import LLMMessage, LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """LLM provider that returns pre-configured responses.

    Responses are dicts. complete() returns them as JSON strings,
    complete_structured() returns them as dicts directly — same as
    real providers that use tool_use / json_schema.
    """

    def __init__(self, responses: list[dict[str, Any]] | None = None):
        self._responses = responses or []
        self._call_index = 0
        self.calls: list[list[LLMMessage]] = []

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        self.calls.append(messages)
        if self._call_index < len(self._responses):
            content = json.dumps(self._responses[self._call_index])
            self._call_index += 1
            return LLMResponse(content=content)
        return LLMResponse(content="{}")

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: Type[BaseModel],
    ) -> dict[str, Any]:
        self.calls.append(messages)
        if self._call_index < len(self._responses):
            data = self._responses[self._call_index]
            self._call_index += 1
            return data
        return {}


@pytest.fixture
def mock_llm():
    return MockLLMProvider


@pytest.fixture
def high_confidence_llm():
    return MockLLMProvider(
        responses=[
            {
                "confidence": 0.9,
                "confidence_reason": "Strong result",
                "context_relevance": 0.85,
                "justification": "Well-supported answer",
            },
            {
                "verified": True,
                "verification_reason": "Independently confirmed",
                "adjusted_confidence": 0.88,
            },
            {"counter_argument": "Possible edge case"},
            {"defense": "Edge case handled by X", "defended": True},
        ]
    )


@pytest.fixture
def low_confidence_llm():
    return MockLLMProvider(
        responses=[
            {
                "confidence": 0.2,
                "confidence_reason": "Weak evidence",
                "context_relevance": 0.3,
                "justification": "Uncertain",
            },
        ]
    )
