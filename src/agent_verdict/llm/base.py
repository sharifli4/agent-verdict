from __future__ import annotations

from abc import ABC, abstractmethod

from agent_verdict.models import LLMMessage, LLMResponse


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        ...

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
