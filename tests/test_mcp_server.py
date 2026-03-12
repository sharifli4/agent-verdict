"""Test that the MCP server tools call through to the pipeline correctly."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agent_verdict.models import Verdict


class TestMCPTools:
    @patch("agent_verdict.mcp_server._get_provider")
    async def test_evaluate_returns_verdict_json(self, mock_get_provider):
        from conftest import MockLLMProvider

        mock_get_provider.return_value = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.9,
                    "confidence_reason": "Strong",
                    "context_relevance": 0.85,
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

        from agent_verdict.mcp_server import evaluate

        result = await evaluate(
            result="SQL injection found",
            task_context="Find security bugs",
        )
        data = json.loads(result)
        assert data["confidence"] > 0.5
        assert data["defended"] is True
        assert data["dropped"] is False

    @patch("agent_verdict.mcp_server._get_provider")
    async def test_check_confidence_only_runs_one_stage(self, mock_get_provider):
        from conftest import MockLLMProvider

        provider = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.7,
                    "confidence_reason": "Decent",
                    "context_relevance": 0.8,
                    "justification": "OK",
                },
            ]
        )
        mock_get_provider.return_value = provider

        from agent_verdict.mcp_server import check_confidence

        result = await check_confidence(
            result="some result",
            task_context="some task",
        )
        data = json.loads(result)
        assert data["confidence"] == 0.7
        assert data["dropped"] is False
        # Only 1 LLM call (confidence stage only)
        assert len(provider.calls) == 1

    @patch("agent_verdict.mcp_server._get_provider")
    async def test_adversarial_check(self, mock_get_provider):
        from conftest import MockLLMProvider

        mock_get_provider.return_value = MockLLMProvider(
            responses=[
                {"counter_argument": "What about X?"},
                {"defense": "X is covered", "defended": True},
            ]
        )

        from agent_verdict.mcp_server import adversarial_check

        result = await adversarial_check(
            result="my answer",
            task_context="task",
            justification="because reasons",
        )
        data = json.loads(result)
        assert data["counter_argument"] == "What about X?"
        assert data["defended"] is True

    @patch("agent_verdict.mcp_server._get_provider")
    async def test_evaluate_drops_low_confidence(self, mock_get_provider):
        from conftest import MockLLMProvider

        mock_get_provider.return_value = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.1,
                    "confidence_reason": "Weak",
                    "context_relevance": 0.1,
                    "justification": "Bad",
                },
            ]
        )

        from agent_verdict.mcp_server import evaluate

        result = await evaluate(
            result="garbage",
            task_context="anything",
        )
        data = json.loads(result)
        assert data["dropped"] is True
