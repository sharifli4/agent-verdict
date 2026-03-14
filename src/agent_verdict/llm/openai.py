from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel

from agent_verdict.models import LLMMessage, LLMResponse

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        supports_structured_output: bool = True,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI SDK not installed. Run:\n\n"
                "  pip install 'agent-verdict[openai]'\n"
            )

        import os

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise RuntimeError(
                f"{api_key_env} not set. Export it first:\n\n"
                f"  export {api_key_env}=...\n"
            )

        self.client = openai.AsyncOpenAI(api_key=resolved_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self._supports_structured_output = supports_structured_output

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        self._track_usage(input_tokens, output_tokens)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: Type[BaseModel],
    ) -> dict[str, Any]:
        """Use OpenAI's response_format with json_schema for structured output.

        Falls back to base class JSON parsing for providers that don't support
        structured output (e.g. DeepSeek, Kimi).
        """
        if not self._supports_structured_output:
            return await super().complete_structured(messages, schema)

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

        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        self._track_usage(input_tokens, output_tokens)
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
