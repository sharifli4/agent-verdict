"""Demo: using VerdictPipeline directly."""
import asyncio, json
from agent_verdict import VerdictPipeline, VerdictConfig, LLMProvider, LLMMessage, LLMResponse

class DemoProvider(LLMProvider):
    def __init__(self):
        self._responses = iter([
            json.dumps({
                "confidence": 0.82,
                "confidence_reason": "Race condition diagnosis matches the symptoms described",
                "context_relevance": 0.90,
                "justification": "Connection pool exhaustion under concurrent load points to race condition",
            }),
            json.dumps({
                "verified": True,
                "verification_reason": "Thread-safety analysis confirms: pool.get() lacks synchronization",
                "adjusted_confidence": 0.85,
            }),
            json.dumps({
                "counter_argument": "Timeout could be slow DNS resolution, not a race condition",
            }),
            json.dumps({
                "defense": "DNS is cached after first lookup — timeouts only appear under concurrency, ruling out DNS",
                "defended": True,
            }),
        ])

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        return LLMResponse(content=next(self._responses))


async def main():
    llm = DemoProvider()
    config = VerdictConfig(confidence_threshold=0.6)
    pipeline = VerdictPipeline(llm=llm, config=config)

    print("Evaluating agent result via pipeline...\n")
    v = await pipeline.evaluate(
        "Root cause: race condition in connection pool — pool.get() is not thread-safe",
        task_context="Diagnose why HTTP requests timeout under load",
    )
    print(f"  Confidence:   {v.confidence}")
    print(f"  Relevant:     {v.context_relevance}")
    print(f"  Counter-arg:  {v.counter_argument}")
    print(f"  Defense:      {v.defense}")
    print(f"  Defended:     {v.defended}")
    print(f"  Dropped:      {v.dropped}")


asyncio.run(main())
