# agent-verdict

Your agent returned an answer. But should you trust it?

agent-verdict takes the raw output of any agent and runs it through a gauntlet: score its confidence, independently verify it, then try to tear it apart with counter-arguments. If the result survives, you get a structured `Verdict` with everything that happened along the way. If it doesn't, the result gets dropped before it reaches your users.

## How it works

The pipeline has three stages, run in order:

```
Agent Output → Confidence → Verification → Adversarial → Verdict
```

1. **Confidence** — An LLM scores how confident it is in the result (0-1) and how relevant the result is to the task context. If either score falls below your thresholds, the result is dropped immediately. No point verifying garbage.

2. **Verification** — A separate LLM call looks at the result fresh and asks: "Would I arrive at the same conclusion independently?" If not, dropped.

3. **Adversarial** — Two LLM calls. First, generate the strongest possible counter-argument against the result. Then, try to defend against it. If the defense fails, the result is dropped.

The pipeline exits early. If confidence is too low, verification and adversarial stages never run — no wasted LLM calls.

## Install

```bash
pip install agent-verdict

# with a provider
pip install agent-verdict[anthropic]
pip install agent-verdict[openai]
```

## Quick start

### Decorator

The fastest way. Wrap your agent function, and the return value goes through the full pipeline before coming back:

```python
from agent_verdict import verdict
from agent_verdict.llm.anthropic import AnthropicProvider

llm = AnthropicProvider()

@verdict(llm=llm, task_context="Identify security vulnerabilities in Python code")
def analyze(code: str) -> str:
    # your agent logic here
    return "Found SQL injection in the login handler"

result = analyze(user_code)  # returns a Verdict, not a string
print(result.confidence)       # 0.87
print(result.defended)         # True
print(result.counter_argument) # "The parameterized query on line 42 might..."
print(result.defense)          # "Line 42 uses string formatting, not params..."
```

If the result gets dropped, the decorator raises `DroppedResultError` by default:

```python
from agent_verdict import DroppedResultError

try:
    result = analyze(user_code)
except DroppedResultError as e:
    print(e.verdict.drop_reason)  # "Confidence 0.23 below threshold 0.5"
```

Or suppress the exception and check manually:

```python
@verdict(llm=llm, task_context="...", raise_on_drop=False)
def analyze(code: str) -> str:
    ...

result = analyze(code)
if result.dropped:
    print(result.drop_reason)
```

Works with async functions too — the decorator detects it automatically:

```python
@verdict(llm=llm, task_context="...")
async def analyze(code: str) -> str:
    return await some_async_agent(code)
```

### Pipeline

For more control, use `VerdictPipeline` directly:

```python
from agent_verdict import VerdictPipeline, VerdictConfig
from agent_verdict.llm.openai import OpenAIProvider

llm = OpenAIProvider(model="gpt-4o")
config = VerdictConfig(confidence_threshold=0.7, require_defense=True)
pipeline = VerdictPipeline(llm=llm, config=config)

verdict = await pipeline.evaluate(
    "The root cause is a race condition in the connection pool",
    task_context="Diagnose why requests timeout under load",
)
```

There's `evaluate_sync()` if you're not in an async context.

## The Verdict object

Everything the pipeline produces ends up here:

```python
class Verdict:
    result: Any               # the original agent output
    justification: str        # why the result makes sense
    confidence: float         # 0.0 - 1.0
    confidence_reason: str    # explanation of the score
    context_relevance: float  # 0.0 - 1.0, how on-topic the result is
    counter_argument: str     # best argument against the result
    defense: str              # response to the counter-argument
    defended: bool            # did the defense hold up?
    dropped: bool             # was the result rejected?
    drop_reason: str          # why it was dropped (if it was)
```

## Configuration

`VerdictConfig` controls when results get dropped:

```python
VerdictConfig(
    confidence_threshold=0.5,   # drop below this confidence
    relevance_threshold=0.4,    # drop below this relevance
    require_defense=True,       # drop if adversarial defense fails
)
```

## Custom stages

You don't have to use the default three stages. Subclass `Stage` and pass your own list:

```python
from agent_verdict import Stage, Verdict, VerdictConfig
from agent_verdict.llm.base import LLMProvider

class MyCustomStage(Stage):
    async def run(
        self, verdict: Verdict, llm: LLMProvider,
        task_context: str, config: VerdictConfig,
    ) -> Verdict:
        # do your thing, return a new verdict
        return verdict.model_copy(update={"confidence": 0.99})

pipeline = VerdictPipeline(
    llm=llm,
    stages=[ConfidenceStage(), MyCustomStage()],  # skip verification, skip adversarial
)
```

Stages are pure — they take a `Verdict` and return a new one. They never mutate the input.

## Custom LLM providers

Implement the `LLMProvider` abstract class:

```python
from agent_verdict import LLMProvider, LLMMessage, LLMResponse

class MyProvider(LLMProvider):
    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        text = await call_my_llm(messages[0].content)
        return LLMResponse(content=text)
```

That's the only method you need. `complete_sync()` has a default implementation that wraps `complete()` with `asyncio.run`.

## Design notes

- **No retry logic.** The underlying SDK (anthropic, openai, etc.) handles retries. This library doesn't add another layer.
- **Conservative on failure.** If the LLM returns malformed JSON, the stage falls back to safe defaults (zero confidence, defense failed). The result gets dropped rather than passed through with garbage data.
- **Stages don't know about each other.** Each stage reads from the Verdict and writes a new one. You can reorder them, remove them, or add your own.
- **LLM calls are the minimum needed.** Confidence = 1 call. Verification = 1 call. Adversarial = 2 calls (counter + defense). Full pipeline = 4 calls total, fewer if it exits early.

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests use a `MockLLMProvider` — no API keys needed.
