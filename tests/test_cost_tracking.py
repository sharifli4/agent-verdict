from __future__ import annotations

import json
from typing import Any, Type

import pytest
from pydantic import BaseModel

from agent_verdict import LLMMessage, LLMProvider, LLMResponse, VerdictConfig, VerdictPipeline
from agent_verdict.llm.base import MODEL_PRICING, _estimate_cost
from agent_verdict.models import StageUsage
from agent_verdict.stages import ConfidenceStage, VerificationStage


class MockLLMWithUsage(LLMProvider):
    """Mock provider that simulates token usage."""

    def __init__(self, model: str, responses: list[dict[str, Any]]):
        self.model = model
        self._responses = responses
        self._call_index = 0

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        if self._call_index < len(self._responses):
            data = self._responses[self._call_index]
            self._call_index += 1
            # Simulate token usage
            self._track_usage(150, 50)
            return LLMResponse(
                content=json.dumps(data),
                input_tokens=150,
                output_tokens=50,
            )
        return LLMResponse(content="{}")

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: Type[BaseModel],
    ) -> dict[str, Any]:
        if self._call_index < len(self._responses):
            data = self._responses[self._call_index]
            self._call_index += 1
            self._track_usage(150, 50)
            return data
        return {}


class TestEstimateCost:
    def test_known_model(self):
        cost = _estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == 2.50 + 10.00

    def test_prefix_match(self):
        cost = _estimate_cost("gpt-4o-2024-08-06", 1_000_000, 0)
        assert cost == 2.50

    def test_unknown_model(self):
        cost = _estimate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_zero_tokens(self):
        cost = _estimate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_small_tokens(self):
        # 1000 input tokens of gpt-4o = $0.0025
        cost = _estimate_cost("gpt-4o", 1000, 0)
        assert abs(cost - 0.0025) < 1e-8


class TestProviderUsageTracking:
    def test_tracks_cumulative_usage(self):
        llm = MockLLMWithUsage("gpt-4o", [{"a": 1}, {"b": 2}])
        # Simulate tracking
        llm._track_usage(100, 50)
        llm._track_usage(200, 75)
        usage = llm.get_usage()
        assert usage["input_tokens"] == 300
        assert usage["output_tokens"] == 125
        assert usage["calls"] == 2

    def test_reset_usage(self):
        llm = MockLLMWithUsage("gpt-4o", [])
        llm._track_usage(100, 50)
        llm.reset_usage()
        usage = llm.get_usage()
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["calls"] == 0

    def test_estimate_cost_method(self):
        llm = MockLLMWithUsage("gpt-4o", [])
        llm._track_usage(1_000_000, 500_000)
        cost = llm.estimate_cost()
        assert cost == 2.50 + 5.00


class TestPipelineUsageTracking:
    async def test_pipeline_records_usage_per_stage(self):
        llm = MockLLMWithUsage("gpt-4o", [
            # ConfidenceStage response
            {
                "confidence": 0.9,
                "confidence_reason": "Strong",
                "context_relevance": 0.85,
                "justification": "Good",
            },
            # VerificationStage response
            {
                "verified": True,
                "verification_reason": "Confirmed",
                "adjusted_confidence": 0.88,
            },
        ])
        pipeline = VerdictPipeline(
            llm=llm,
            stages=[ConfidenceStage(), VerificationStage()],
        )
        verdict = await pipeline.evaluate("test result", task_context="test task")

        assert len(verdict.usage) == 2
        assert verdict.usage[0].stage == "ConfidenceStage"
        assert verdict.usage[0].llm_calls == 1
        assert verdict.usage[0].input_tokens == 150
        assert verdict.usage[0].output_tokens == 50
        assert verdict.usage[0].total_tokens == 200

        assert verdict.usage[1].stage == "VerificationStage"
        assert verdict.usage[1].llm_calls == 1

    async def test_total_cost_property(self):
        llm = MockLLMWithUsage("gpt-4o", [
            {
                "confidence": 0.9,
                "confidence_reason": "Strong",
                "context_relevance": 0.85,
                "justification": "Good",
            },
        ])
        pipeline = VerdictPipeline(
            llm=llm,
            stages=[ConfidenceStage()],
        )
        verdict = await pipeline.evaluate("test", task_context="task")

        assert verdict.total_tokens == 200
        assert verdict.total_cost > 0

    async def test_early_exit_only_tracks_run_stages(self):
        llm = MockLLMWithUsage("gpt-4o", [
            # Low confidence → drops, verification never runs
            {
                "confidence": 0.1,
                "confidence_reason": "Weak",
                "context_relevance": 0.1,
                "justification": "Bad",
            },
        ])
        pipeline = VerdictPipeline(
            llm=llm,
            stages=[ConfidenceStage(), VerificationStage()],
        )
        verdict = await pipeline.evaluate("bad result", task_context="task")

        assert verdict.dropped
        assert len(verdict.usage) == 1  # only confidence ran
        assert verdict.usage[0].stage == "ConfidenceStage"

    async def test_usage_in_json_output(self):
        llm = MockLLMWithUsage("gpt-4o", [
            {
                "confidence": 0.9,
                "confidence_reason": "Strong",
                "context_relevance": 0.85,
                "justification": "Good",
            },
        ])
        pipeline = VerdictPipeline(
            llm=llm,
            stages=[ConfidenceStage()],
        )
        verdict = await pipeline.evaluate("test", task_context="task")
        data = verdict.model_dump()

        assert "usage" in data
        assert len(data["usage"]) == 1
        assert data["usage"][0]["stage"] == "ConfidenceStage"
        assert data["usage"][0]["total_tokens"] == 200
        assert data["usage"][0]["cost"] > 0
