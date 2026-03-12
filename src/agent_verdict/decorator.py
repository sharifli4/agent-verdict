from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable

from .llm.base import LLMProvider
from .models import Verdict, VerdictConfig
from .pipeline import VerdictPipeline
from .stages.base import Stage


class DroppedResultError(Exception):
    def __init__(self, verdict: Verdict):
        self.verdict = verdict
        super().__init__(
            f"Result dropped: {verdict.drop_reason} "
            f"(confidence={verdict.confidence:.2f})"
        )


def verdict(
    llm: LLMProvider,
    task_context: str = "",
    config: VerdictConfig | None = None,
    stages: list[Stage] | None = None,
    raise_on_drop: bool = True,
) -> Callable:
    pipeline = VerdictPipeline(llm=llm, config=config, stages=stages)

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Verdict:
                result = await fn(*args, **kwargs)
                v = await pipeline.evaluate(result, task_context)
                if v.dropped and raise_on_drop:
                    raise DroppedResultError(v)
                return v

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Verdict:
                result = fn(*args, **kwargs)
                v = asyncio.run(pipeline.evaluate(result, task_context))
                if v.dropped and raise_on_drop:
                    raise DroppedResultError(v)
                return v

            return sync_wrapper

    return decorator
