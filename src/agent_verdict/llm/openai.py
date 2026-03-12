from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel

from agent_verdict.models import LLMMessage, LLMResponse

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 1024):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Install the openai extra: pip install agent-verdict[openai]"
            )
        self.client = openai.AsyncOpenAI()
        self.model = model
        self.max_tokens = max_tokens

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(content=response.choices[0].message.content or "")

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: Type[BaseModel],
    ) -> dict[str, Any]:
        """Use OpenAI's response_format with json_schema for structured output."""
        json_schema = schema.model_json_schema()
        json_schema.pop("title", None)

        # OpenAI requires all properties to be required for strict mode
        if "properties" in json_schema:
            json_schema["required"] = list(json_schema["properties"].keys())
            json_schema["additionalProperties"] = False

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)
