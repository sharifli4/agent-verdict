import pytest

from agent_verdict import Verdict, VerdictConfig
from agent_verdict.stages.adversarial import AdversarialStage
from conftest import MockLLMProvider


@pytest.fixture
def stage():
    return AdversarialStage()


class TestAdversarialStage:
    async def test_defended_passes(self, stage):
        llm = MockLLMProvider(
            responses=[
                {"counter_argument": "What about edge case X?"},
                {"defense": "X is handled by Y", "defended": True},
            ]
        )
        verdict = Verdict(result="answer", justification="because")
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.counter_argument == "What about edge case X?"
        assert result.defense == "X is handled by Y"
        assert result.defended is True
        assert result.dropped is False

    async def test_undefended_drops(self, stage):
        llm = MockLLMProvider(
            responses=[
                {"counter_argument": "Fatal flaw in reasoning"},
                {"defense": "Cannot defend", "defended": False},
            ]
        )
        verdict = Verdict(result="answer", justification="because")
        result = await stage.run(verdict, llm, "task", VerdictConfig())
        assert result.defended is False
        assert result.dropped is True
        assert "counter-argument" in result.drop_reason.lower()

    async def test_defense_not_required(self, stage):
        llm = MockLLMProvider(
            responses=[
                {"counter_argument": "Flaw"},
                {"defense": "No defense", "defended": False},
            ]
        )
        config = VerdictConfig(require_defense=False)
        verdict = Verdict(result="answer", justification="because")
        result = await stage.run(verdict, llm, "task", config)
        assert result.defended is False
        assert result.dropped is False

    async def test_makes_two_llm_calls(self, stage):
        llm = MockLLMProvider(
            responses=[
                {"counter_argument": "counter"},
                {"defense": "def", "defended": True},
            ]
        )
        verdict = Verdict(result="x", justification="y")
        await stage.run(verdict, llm, "task", VerdictConfig())
        assert len(llm.calls) == 2
