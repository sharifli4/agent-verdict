import pytest

from agent_verdict import Verdict, VerdictConfig, VerdictPipeline
from conftest import MockLLMProvider


class TestVerdictPipeline:
    async def test_full_pipeline_passes(self, high_confidence_llm):
        pipeline = VerdictPipeline(llm=high_confidence_llm)
        result = await pipeline.evaluate("good answer", task_context="test task")
        assert result.confidence > 0.5
        assert result.defended is True
        assert result.dropped is False

    async def test_early_exit_on_low_confidence(self, low_confidence_llm):
        pipeline = VerdictPipeline(llm=low_confidence_llm)
        result = await pipeline.evaluate("bad answer", task_context="test task")
        assert result.dropped is True
        # Should have stopped after confidence stage — only 1 LLM call
        assert len(low_confidence_llm.calls) == 1

    async def test_early_exit_skips_stages(self):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.1,
                    "confidence_reason": "bad",
                    "context_relevance": 0.1,
                    "justification": "weak",
                },
            ]
        )
        pipeline = VerdictPipeline(llm=llm)
        result = await pipeline.evaluate("answer", task_context="task")
        assert result.dropped is True
        # Only confidence stage ran
        assert len(llm.calls) == 1

    async def test_custom_stages(self):
        from agent_verdict.stages import ConfidenceStage

        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.9,
                    "confidence_reason": "ok",
                    "context_relevance": 0.9,
                    "justification": "fine",
                }
            ]
        )
        pipeline = VerdictPipeline(llm=llm, stages=[ConfidenceStage()])
        result = await pipeline.evaluate("answer", task_context="task")
        assert result.dropped is False
        assert result.confidence == 0.9
        assert len(llm.calls) == 1

    async def test_custom_config_thresholds(self):
        llm = MockLLMProvider(
            responses=[
                {
                    "confidence": 0.6,
                    "confidence_reason": "ok",
                    "context_relevance": 0.5,
                    "justification": "alright",
                }
            ]
        )
        config = VerdictConfig(confidence_threshold=0.9)
        pipeline = VerdictPipeline(
            llm=llm, config=config, stages=[__import__("agent_verdict").ConfidenceStage()]
        )
        result = await pipeline.evaluate("answer", task_context="task")
        assert result.dropped is True

    def test_evaluate_sync(self, high_confidence_llm):
        pipeline = VerdictPipeline(llm=high_confidence_llm)
        result = pipeline.evaluate_sync("good answer", task_context="test task")
        assert result.dropped is False
        assert result.defended is True
