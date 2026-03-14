from __future__ import annotations

from .openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    def __init__(self, model: str = "deepseek-chat", max_tokens: int = 1024):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            supports_structured_output=False,
        )
