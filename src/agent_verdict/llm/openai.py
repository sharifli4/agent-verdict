from __future__ import annotations

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
