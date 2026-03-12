from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel

from agent_verdict.models import LLMMessage, LLMResponse


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        ...

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: Type[BaseModel],
    ) -> dict[str, Any]:
        """Request structured output matching the given Pydantic schema.

        Providers should override this to use native structured output
        (Anthropic tool_use, OpenAI json_schema). The default falls back
        to plain complete() + JSON parsing.
        """
        from agent_verdict.stages.base import parse_llm_json

        response = await self.complete(messages)
        return parse_llm_json(response.content, {})

    def complete_sync(self, messages: list[LLMMessage]) -> LLMResponse:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.complete(messages)).result()
        return asyncio.run(self.complete(messages))
