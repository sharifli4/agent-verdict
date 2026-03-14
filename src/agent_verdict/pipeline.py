from __future__ import annotations

from typing import Any

from .llm.base import LLMProvider, _estimate_cost
from .models import StageUsage, Verdict, VerdictConfig
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
        all_usage: list[StageUsage] = []

        for stage in self.stages:
            if verdict.dropped:
                break
            self.llm.reset_usage()
            verdict = await stage.run(verdict, self.llm, task_context, self.config)

            # Collect usage from primary LLM
            usage = self.llm.get_usage()
            model = getattr(self.llm, "model", "")

            # For CrossVerificationStage, also collect challenger usage
            challengers = getattr(stage, "challengers", None)
            if challengers:
                for challenger in challengers:
                    c_usage = challenger.get_usage()
                    c_model = getattr(challenger, "model", "")
                    usage["input_tokens"] += c_usage["input_tokens"]
                    usage["output_tokens"] += c_usage["output_tokens"]
                    usage["calls"] += c_usage["calls"]
                    # Use challenger model for cost if primary has no tokens
                    if c_usage["input_tokens"] > 0 and not model:
                        model = c_model

            input_t = usage["input_tokens"]
            output_t = usage["output_tokens"]
            cost = _estimate_cost(model, input_t, output_t)

            # Also estimate cost per challenger model separately
            if challengers:
                cost = _estimate_cost(
                    getattr(self.llm, "model", ""),
                    self.llm.get_usage()["input_tokens"],
                    self.llm.get_usage()["output_tokens"],
                )
                for challenger in challengers:
                    c_usage = challenger.get_usage()
                    c_model = getattr(challenger, "model", "")
                    cost += _estimate_cost(c_model, c_usage["input_tokens"], c_usage["output_tokens"])

            all_usage.append(StageUsage(
                stage=type(stage).__name__,
                input_tokens=input_t,
                output_tokens=output_t,
                total_tokens=input_t + output_t,
                llm_calls=usage["calls"],
                cost=round(cost, 6),
            ))

        verdict = verdict.model_copy(update={"usage": all_usage})
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
