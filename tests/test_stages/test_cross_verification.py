from __future__ import annotations

import pytest

from agent_verdict.models import Verdict, VerdictConfig
from agent_verdict.stages.cross_verification import CrossVerificationStage


class TestCrossVerificationStage:
    @pytest.fixture
    def support_llm(self, mock_llm):
        """Juror that supports the result in both rounds."""
        return mock_llm(responses=[
            # Phase 1: position
            {
                "vote": "support",
                "argument": "The result is well-reasoned",
                "counter_to_self": "Could be missing edge cases",
                "confidence": 0.85,
            },
            # Phase 2: deliberation
            {
                "final_vote": "support",
                "rebuttal": "Edge cases are covered by the implementation",
                "confidence": 0.9,
            },
        ])

    @pytest.fixture
    def challenge_llm(self, mock_llm):
        """Juror that challenges the result in both rounds."""
        return mock_llm(responses=[
            # Phase 1: position
            {
                "vote": "challenge",
                "argument": "The result overlooks critical issues",
                "counter_to_self": "The main logic is sound",
                "confidence": 0.8,
            },
            # Phase 2: deliberation
            {
                "final_vote": "challenge",
                "rebuttal": "Critical issues remain unaddressed",
                "confidence": 0.85,
            },
        ])

    @pytest.fixture
    def flip_llm(self, mock_llm):
        """Juror that challenges initially, then flips to support."""
        return mock_llm(responses=[
            # Phase 1: position — challenge
            {
                "vote": "challenge",
                "argument": "Seems uncertain",
                "counter_to_self": "But the core approach is valid",
                "confidence": 0.5,
            },
            # Phase 2: deliberation — flips to support
            {
                "final_vote": "support",
                "rebuttal": "Other juror made a good point, changing my vote",
                "confidence": 0.7,
            },
        ])

    async def test_majority_support_passes(self, support_llm, mock_llm):
        """Two supporters = passes."""
        challenger = mock_llm(responses=[
            {"vote": "support", "argument": "Good", "counter_to_self": "Maybe", "confidence": 0.8},
            {"final_vote": "support", "rebuttal": "Agreed", "confidence": 0.85},
        ])
        stage = CrossVerificationStage(challengers=[challenger])
        verdict = Verdict(result="test result", confidence=0.9)
        config = VerdictConfig()

        result = await stage.run(verdict, support_llm, "test task", config)

        assert not result.dropped
        assert len(result.deliberation) == 2
        assert all(p.final_vote == "support" for p in result.deliberation)

    async def test_majority_challenge_drops(self, support_llm, challenge_llm, mock_llm):
        """Two challengers vs one supporter = dropped."""
        challenger2 = mock_llm(responses=[
            {"vote": "challenge", "argument": "Flawed", "counter_to_self": "Partially ok", "confidence": 0.75},
            {"final_vote": "challenge", "rebuttal": "Still flawed", "confidence": 0.8},
        ])
        stage = CrossVerificationStage(challengers=[challenge_llm, challenger2])
        verdict = Verdict(result="test result", confidence=0.9)
        config = VerdictConfig()

        result = await stage.run(verdict, support_llm, "test task", config)

        assert result.dropped
        assert "2/3" in result.drop_reason
        assert len(result.deliberation) == 3

    async def test_juror_can_flip_vote(self, support_llm, flip_llm):
        """A juror that starts challenging can flip to support after deliberation."""
        stage = CrossVerificationStage(challengers=[flip_llm])
        verdict = Verdict(result="test result", confidence=0.9)
        config = VerdictConfig()

        result = await stage.run(verdict, support_llm, "test task", config)

        assert not result.dropped
        # flip_llm started as challenge but ended as support
        flip_pos = [p for p in result.deliberation if p.vote == "challenge"][0]
        assert flip_pos.final_vote == "support"

    async def test_deliberation_records_all_positions(self, support_llm, challenge_llm):
        """Deliberation list contains one entry per juror."""
        stage = CrossVerificationStage(challengers=[challenge_llm])
        verdict = Verdict(result="test result", confidence=0.9)
        config = VerdictConfig()

        result = await stage.run(verdict, support_llm, "test task", config)

        assert len(result.deliberation) == 2
        juror_names = [p.juror for p in result.deliberation]
        assert len(set(juror_names)) == 2  # unique names

    async def test_confidence_blended_with_vote_ratio(self, support_llm, challenge_llm):
        """Confidence reflects the weighted vote ratio."""
        stage = CrossVerificationStage(challengers=[challenge_llm])
        verdict = Verdict(result="test result", confidence=0.9)
        config = VerdictConfig()

        result = await stage.run(verdict, support_llm, "test task", config)

        # 1 support (0.9) vs 1 challenge (0.85), ratio ≈ 0.514
        assert 0.0 < result.confidence < 0.9

    async def test_skips_with_no_challengers(self, support_llm):
        """With no challengers (single juror), stage is a no-op."""
        stage = CrossVerificationStage(challengers=[])
        verdict = Verdict(result="test result", confidence=0.9)
        config = VerdictConfig()

        result = await stage.run(verdict, support_llm, "test task", config)

        assert result.confidence == 0.9
        assert len(result.deliberation) == 0

    async def test_parallel_calls(self, mock_llm):
        """Verify each juror makes exactly 2 calls (position + deliberation)."""
        juror_a = mock_llm(responses=[
            {"vote": "support", "argument": "A", "counter_to_self": "B", "confidence": 0.8},
            {"final_vote": "support", "rebuttal": "C", "confidence": 0.85},
        ])
        juror_b = mock_llm(responses=[
            {"vote": "support", "argument": "D", "counter_to_self": "E", "confidence": 0.7},
            {"final_vote": "support", "rebuttal": "F", "confidence": 0.75},
        ])
        stage = CrossVerificationStage(challengers=[juror_b])
        verdict = Verdict(result="test", confidence=0.9)

        await stage.run(verdict, juror_a, "task", VerdictConfig())

        assert len(juror_a.calls) == 2  # position + deliberation
        assert len(juror_b.calls) == 2
