import pytest
from pydantic import ValidationError

from agent_verdict import Verdict, VerdictConfig


class TestVerdict:
    def test_defaults(self):
        v = Verdict()
        assert v.result is None
        assert v.confidence == 0.0
        assert v.dropped is False

    def test_with_result(self):
        v = Verdict(result="hello", confidence=0.8, defended=True)
        assert v.result == "hello"
        assert v.confidence == 0.8
        assert v.defended is True

    def test_model_copy(self):
        v = Verdict(result="test", confidence=0.5)
        v2 = v.model_copy(update={"confidence": 0.9, "dropped": True})
        assert v.confidence == 0.5
        assert v2.confidence == 0.9
        assert v2.dropped is True
        assert v2.result == "test"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Verdict(confidence=1.5)
        with pytest.raises(ValidationError):
            Verdict(confidence=-0.1)


class TestVerdictConfig:
    def test_defaults(self):
        c = VerdictConfig()
        assert c.confidence_threshold == 0.5
        assert c.relevance_threshold == 0.4
        assert c.require_defense is True

    def test_custom(self):
        c = VerdictConfig(confidence_threshold=0.8, require_defense=False)
        assert c.confidence_threshold == 0.8
        assert c.require_defense is False
