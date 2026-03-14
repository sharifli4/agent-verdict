from __future__ import annotations

from .openai import OpenAIProvider


class KimiProvider(OpenAIProvider):
    def __init__(self, model: str = "kimi-k2.5", max_tokens: int = 1024):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url="https://api.moonshot.ai/v1",
            api_key_env="MOONSHOT_API_KEY",
            supports_structured_output=False,
        )
