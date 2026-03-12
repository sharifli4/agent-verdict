from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import Verdict, VerdictConfig


class Stage(ABC):
    @abstractmethod
    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        ...


def parse_llm_json(text: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract JSON from LLM response text, with graceful fallback."""
    defaults = defaults or {}
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return defaults
