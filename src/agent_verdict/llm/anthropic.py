from __future__ import annotations

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
