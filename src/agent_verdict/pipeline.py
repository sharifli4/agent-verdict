from __future__ import annotations

from typing import Any

from .llm.base import LLMProvider
from .models import Verdict, VerdictConfig
from .stages import AdversarialStage, ConfidenceStage, Stage, VerificationStage


class VerdictPipeline:
    def __init__(
        self,
        llm: LLMProvider,
        config: VerdictConfig | None = None,
        stages: list[Stage] | None = None,
    ):
        self.llm = llm
        self.config = config or VerdictConfig()
        self.stages = stages or [
            ConfidenceStage(),
            VerificationStage(),
            AdversarialStage(),
        ]

    async def evaluate(
        self,
        result: Any,
        task_context: str = "",
    ) -> Verdict:
        verdict = Verdict(result=result)
        for stage in self.stages:
            if verdict.dropped:
                break
            verdict = await stage.run(verdict, self.llm, task_context, self.config)
        return verdict

    def evaluate_sync(
        self,
        result: Any,
        task_context: str = "",
    ) -> Verdict:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.evaluate(result, task_context)
                ).result()
        return asyncio.run(self.evaluate(result, task_context))
