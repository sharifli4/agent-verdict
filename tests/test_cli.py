"""Test the CLI by mocking the LLM provider."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent_verdict.cli import main


def _mock_provider(responses):
    from conftest import MockLLMProvider
    return MockLLMProvider(responses=responses)


class TestCLIEvaluate:
    @patch("agent_verdict.cli._get_provider")
    def test_evaluate_passing(self, mock_get, capsys):
        mock_get.return_value = _mock_provider([
            {"confidence": 0.9, "confidence_reason": "Good", "context_relevance": 0.85, "justification": "Solid"},
            {"verified": True, "verification_reason": "Confirmed", "adjusted_confidence": 0.88},
            {"counter_argument": "Edge case"},
            {"defense": "Handled", "defended": True},
        ])
        with pytest.raises(SystemExit) as exc:
            import sys
            with patch.object(sys, "argv", ["agent-verdict", "evaluate", "test result", "-c", "test task"]):
                main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "PASSED" in out

    @patch("agent_verdict.cli._get_provider")
    def test_evaluate_dropped(self, mock_get, capsys):
        mock_get.return_value = _mock_provider([
            {"confidence": 0.1, "confidence_reason": "Bad", "context_relevance": 0.1, "justification": "Weak"},
        ])
        with pytest.raises(SystemExit) as exc:
            import sys
            with patch.object(sys, "argv", ["agent-verdict", "evaluate", "bad result", "-c", "task"]):
                main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "DROPPED" in out

    @patch("agent_verdict.cli._get_provider")
    def test_evaluate_json_output(self, mock_get, capsys):
        mock_get.return_value = _mock_provider([
            {"confidence": 0.9, "confidence_reason": "Good", "context_relevance": 0.85, "justification": "Solid"},
            {"verified": True, "verification_reason": "Confirmed", "adjusted_confidence": 0.88},
            {"counter_argument": "Edge case"},
            {"defense": "Handled", "defended": True},
        ])
        with pytest.raises(SystemExit):
            import sys
            with patch.object(sys, "argv", ["agent-verdict", "--json", "evaluate", "test", "-c", "task"]):
                main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["confidence"] > 0.5
        assert data["dropped"] is False


class TestCLICheck:
    @patch("agent_verdict.cli._get_provider")
    def test_check_passing(self, mock_get, capsys):
        mock_get.return_value = _mock_provider([
            {"confidence": 0.8, "confidence_reason": "OK", "context_relevance": 0.7, "justification": "Fine"},
        ])
        with pytest.raises(SystemExit) as exc:
            import sys
            with patch.object(sys, "argv", ["agent-verdict", "check", "result", "-c", "task"]):
                main()
        assert exc.value.code == 0


class TestCLIAttack:
    @patch("agent_verdict.cli._get_provider")
    def test_attack_defended(self, mock_get, capsys):
        mock_get.return_value = _mock_provider([
            {"counter_argument": "What about X?"},
            {"defense": "X is handled", "defended": True},
        ])
        with pytest.raises(SystemExit) as exc:
            import sys
            with patch.object(sys, "argv", ["agent-verdict", "attack", "answer", "-c", "task"]):
                main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "What about X?" in out

    @patch("agent_verdict.cli._get_provider")
    def test_attack_not_defended(self, mock_get, capsys):
        mock_get.return_value = _mock_provider([
            {"counter_argument": "Fatal flaw"},
            {"defense": "Cannot defend", "defended": False},
        ])
        with pytest.raises(SystemExit) as exc:
            import sys
            with patch.object(sys, "argv", ["agent-verdict", "attack", "answer", "-c", "task"]):
                main()
        assert exc.value.code == 1


class TestCLIStdin:
    @patch("agent_verdict.cli._get_provider")
    def test_reads_from_stdin(self, mock_get, capsys, monkeypatch):
        import io
        mock_get.return_value = _mock_provider([
            {"confidence": 0.8, "confidence_reason": "OK", "context_relevance": 0.7, "justification": "Fine"},
        ])
        monkeypatch.setattr("sys.stdin", io.StringIO("piped result"))
        with pytest.raises(SystemExit) as exc:
            import sys
            with patch.object(sys, "argv", ["agent-verdict", "check", "-c", "task"]):
                main()
        assert exc.value.code == 0
