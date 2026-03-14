from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel

from agent_verdict.models import LLMMessage, LLMResponse


# Default pricing per 1M tokens: (input_usd, output_usd)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "kimi-k2.5": (0.50, 2.80),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts and model pricing."""
    # Try exact match, then prefix match
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        for key, val in MODEL_PRICING.items():
            if model.startswith(key):
                pricing = val
                break
    if not pricing:
        return 0.0
    input_price, output_price = pricing
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


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

    def _track_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage for cost tracking."""
        if not hasattr(self, "_usage"):
            self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        self._usage["input_tokens"] += input_tokens
        self._usage["output_tokens"] += output_tokens
        self._usage["calls"] += 1

    def get_usage(self) -> dict[str, int]:
        """Return accumulated token usage since last reset."""
        return getattr(self, "_usage", {"input_tokens": 0, "output_tokens": 0, "calls": 0}).copy()

    def reset_usage(self) -> None:
        """Reset accumulated token usage."""
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def estimate_cost(self) -> float:
        """Estimate accumulated cost in USD."""
        model = getattr(self, "model", "")
        usage = self.get_usage()
        return _estimate_cost(model, usage["input_tokens"], usage["output_tokens"])

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
