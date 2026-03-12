# agent-verdict

<p align="center">
  <img src="demo/demo.gif" alt="agent-verdict demo" width="800">
</p>

You have an AI agent. It gives you an answer. But is the answer actually good? You don't know. The agent doesn't know either. It just sounds confident.

This library asks that question properly. It takes your agent's output and does three things:

1. **"How sure are you?"** — Scores the answer 0 to 1. Low score? Thrown out immediately.
2. **"Would you say the same thing if I asked you fresh?"** — A second, independent check. If the answer doesn't hold up on its own, it's gone.
3. **"Here's the best argument against you. Defend yourself."** — Generates a counter-argument, then tries to defend the original answer. If the defense fails, the answer is dropped.

If the answer survives all three, you get it back with all the details — the scores, the reasoning, the counter-argument, the defense. If it doesn't survive, you know exactly why.

That's it. That's the whole thing.

```
Your agent's answer → Confidence check → Verification → Adversarial attack → Verdict
```

If it fails at any step, the rest don't run. No wasted LLM calls.

## Install

```bash
pip install agent-verdict

# pick your LLM provider
pip install agent-verdict[anthropic]
pip install agent-verdict[openai]
```

## Usage

### The simple way — decorator

You already have a function that does something. Add one line:

```python
from agent_verdict import verdict
from agent_verdict.llm.anthropic import AnthropicProvider

llm = AnthropicProvider()

@verdict(llm=llm, task_context="Find security bugs in Python code")
def analyze(code: str) -> str:
    return "Found SQL injection in the login handler"
```

Before: `analyze()` returns a string. After: it returns a `Verdict` object that went through the whole gauntlet.

```python
result = analyze(user_code)

result.confidence        # 0.87 — how sure the LLM is
result.defended          # True — survived the adversarial check
result.counter_argument  # "The query on line 42 uses parameterized..."
result.defense           # "Line 42 uses f-string formatting, not params..."
result.dropped           # False — answer survived
```

If the answer is bad, it blows up by default:

```python
from agent_verdict import DroppedResultError

try:
    result = analyze(bad_code)
except DroppedResultError as e:
    print(e.verdict.drop_reason)  # "Confidence 0.23 below threshold 0.5"
```

Don't want exceptions? Fine:

```python
@verdict(llm=llm, task_context="...", raise_on_drop=False)
def analyze(code: str) -> str:
    ...

result = analyze(code)
if result.dropped:
    print(f"Answer rejected: {result.drop_reason}")
```

Async works too. The decorator figures it out:

```python
@verdict(llm=llm, task_context="...")
async def analyze(code: str) -> str:
    return await some_async_thing(code)
```

### Command line

No Python needed. Pipe output in or pass it as an argument:

```bash
# full pipeline — confidence + verification + adversarial
agent-verdict evaluate "SQL injection found on line 14" -c "Find security bugs"

# pipe from another command
my-agent analyze code.py | agent-verdict evaluate -c "Find security bugs"

# quick confidence check only (1 LLM call)
agent-verdict check "the server crashed due to OOM" -c "Diagnose outage"

# adversarial only — attack and defend
agent-verdict attack "race condition in pool.get()" -c "Find concurrency bugs"
```

Get JSON instead of pretty output:

```bash
agent-verdict --json evaluate "result here" -c "task"
```

Exit code is `0` if the result passes, `1` if it gets dropped. So you can use it in scripts:

```bash
if agent-verdict check "$AGENT_OUTPUT" -c "$TASK" --json > verdict.json; then
    echo "good to go"
else
    echo "agent answer was rejected"
fi
```

Options:

```
-p, --provider        anthropic or openai (default: anthropic)
-m, --model           model name (default: provider default)
--json                output raw JSON
-v, --verbose         show extra details
--confidence-threshold  drop below this (default: 0.5)
--relevance-threshold   drop below this (default: 0.4)
--require-defense / --no-require-defense
```

### The manual way — pipeline

Same thing, more control:

```python
from agent_verdict import VerdictPipeline, VerdictConfig
from agent_verdict.llm.openai import OpenAIProvider

llm = OpenAIProvider(model="gpt-4o")
config = VerdictConfig(confidence_threshold=0.7)
pipeline = VerdictPipeline(llm=llm, config=config)

result = await pipeline.evaluate(
    "The root cause is a race condition in the connection pool",
    task_context="Diagnose why requests timeout under load",
)
```

Not using async? There's `pipeline.evaluate_sync()`.

## What you get back

The `Verdict` object has everything:

```python
result.result              # whatever your agent originally returned
result.confidence          # 0.0 to 1.0
result.confidence_reason   # why it got that score
result.context_relevance   # 0.0 to 1.0, is this even answering the right question?
result.justification       # why the answer makes sense
result.counter_argument    # the best attack against the answer
result.defense             # the response to that attack
result.defended            # True/False, did the defense hold?
result.dropped             # True/False, was the answer thrown out?
result.drop_reason         # if dropped, why
```

## Configuration

Three knobs:

```python
VerdictConfig(
    confidence_threshold=0.5,   # below this confidence → dropped
    relevance_threshold=0.4,    # below this relevance → dropped
    require_defense=True,       # can't defend itself → dropped
)
```

## Bring your own LLM

Implement one method:

```python
from agent_verdict import LLMProvider, LLMMessage, LLMResponse

class MyProvider(LLMProvider):
    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        text = await call_whatever_llm_you_want(messages[0].content)
        return LLMResponse(content=text)
```

Done.

## Bring your own stages

Don't like the default three stages? Swap them out:

```python
from agent_verdict import Stage, Verdict, VerdictConfig, VerdictPipeline, ConfidenceStage
from agent_verdict.llm.base import LLMProvider

class MyStage(Stage):
    async def run(self, verdict, llm, task_context, config):
        # do whatever you want
        return verdict.model_copy(update={"confidence": 0.99})

pipeline = VerdictPipeline(
    llm=llm,
    stages=[ConfidenceStage(), MyStage()],  # only these two run
)
```

Each stage gets a `Verdict` in, returns a new `Verdict` out. They don't touch each other.

## MCP server (Claude Code, Cursor)

Want to use this inside your editor instead of in code? It works as an MCP plugin.

```bash
pip install agent-verdict[mcp-anthropic]
# or
pip install agent-verdict[mcp-openai]
```

### Claude Code

Add to `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "agent-verdict": {
      "command": "agent-verdict-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "your-key"
      }
    }
  }
}
```

Or from the terminal:

```bash
claude mcp add agent-verdict agent-verdict-mcp
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agent-verdict": {
      "command": "agent-verdict-mcp",
      "env": {
        "OPENAI_API_KEY": "your-key",
        "VERDICT_PROVIDER": "openai"
      }
    }
  }
}
```

### What tools you get

Three tools show up in your editor:

- **`evaluate`** — The full pipeline. All three stages. Give it a result and context, get back a verdict.
- **`check_confidence`** — Just the confidence score. Quick and cheap, one LLM call.
- **`adversarial_check`** — Just the attack/defend part. For when you trust the answer but want to stress-test it.

### Environment variables

| Variable | Default | What it does |
|---|---|---|
| `VERDICT_PROVIDER` | `anthropic` | Which LLM to use: `anthropic` or `openai` |
| `VERDICT_MODEL` | provider default | Specific model name |

API keys come from the usual places (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

## How many LLM calls does this make?

- Confidence: 1 call
- Verification: 1 call
- Adversarial: 2 calls (attack + defend)
- **Full pipeline: 4 calls total**

But if the answer fails early (low confidence), the rest don't run. Could be as few as 1 call.

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Everything uses a mock LLM provider. No API keys needed, no network calls.
