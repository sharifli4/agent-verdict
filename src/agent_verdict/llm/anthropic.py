from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel

from agent_verdict.models import LLMMessage, LLMResponse

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Install the anthropic extra: pip install agent-verdict[anthropic]"
            )
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(content=response.content[0].text)

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: Type[BaseModel],
    ) -> dict[str, Any]:
        """Use Anthropic tool_use to force structured JSON output."""
        tool_name = schema.__name__
        tool_schema = schema.model_json_schema()

        # Remove fields that Anthropic doesn't accept in tool input_schema
        tool_schema.pop("title", None)
        tool_schema.pop("$defs", None)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            tools=[
                {
                    "name": tool_name,
                    "description": f"Return structured {tool_name} output",
                    "input_schema": tool_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.input

        # Fallback: shouldn't happen with tool_choice forced, but just in case
        return {}
