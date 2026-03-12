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
pip install git+https://github.com/sharifli4/agent-verdict.git
```

Both Anthropic and OpenAI SDKs are included. Set your API key and you're good:

```bash
export ANTHROPIC_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...
```

The tool auto-detects which provider to use based on which key is set. That's all you need.

### Other ways to install

```bash
# if you use uv
uv pip install git+https://github.com/sharifli4/agent-verdict.git

# if you just want the CLI tool, isolated from everything else
pipx install git+https://github.com/sharifli4/agent-verdict.git

# run it once without installing
uvx --from git+https://github.com/sharifli4/agent-verdict.git agent-verdict --help
```

## Usage

### Command line

The fastest way. No code to write:

```bash
agent-verdict evaluate "SQL injection found on line 14" -c "Find security bugs"
```

That's it. It runs the full pipeline and prints the verdict.

More examples:

```bash
# pipe output from another command
my-agent analyze code.py | agent-verdict eval -c "Find security bugs"

# quick confidence check — 1 LLM call, fast and cheap
agent-verdict check "the server crashed due to OOM" -c "Diagnose outage"

# just the adversarial part — attack the answer, then defend it
agent-verdict attack "race condition in pool.get()" -c "Find concurrency bugs"
```

JSON output for scripts:

```bash
agent-verdict --json eval "result" -c "task" > verdict.json
```

Exit code is `0` if the answer passes, `1` if dropped. Use it in CI, shell scripts, whatever:

```bash
if agent-verdict check "$AGENT_OUTPUT" -c "$TASK"; then
    echo "good to go"
else
    echo "rejected"
fi
```

You don't have to specify the provider. It picks it up from your API key. But if you want to be explicit:

```bash
agent-verdict -p openai eval "result" -c "task"
agent-verdict -p anthropic -m claude-sonnet-4-6 eval "result" -c "task"
```

### Python — decorator

You already have a function. Add one line:

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

If the answer is bad, it raises an error:

```python
from agent_verdict import DroppedResultError

try:
    result = analyze(bad_code)
except DroppedResultError as e:
    print(e.verdict.drop_reason)  # "Confidence 0.23 below threshold 0.5"
```

Don't want exceptions? Set `raise_on_drop=False` and check `result.dropped` yourself.

Async works too. The decorator figures it out:

```python
@verdict(llm=llm, task_context="...")
async def analyze(code: str) -> str:
    return await some_async_agent(code)
```

### Python — pipeline

More control:

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

### MCP plugin (Claude Code, Cursor)

Use it as a tool inside your editor:

```bash
pip install "agent-verdict[mcp] @ git+https://github.com/sharifli4/agent-verdict.git"
```

**Claude Code** — one command:

```bash
claude mcp add agent-verdict agent-verdict-mcp
```

Or add to `.mcp.json`:

```json
{
  "mcpServers": {
    "agent-verdict": {
      "command": "agent-verdict-mcp",
      "env": { "ANTHROPIC_API_KEY": "your-key" }
    }
  }
}
```

**Cursor** — add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agent-verdict": {
      "command": "agent-verdict-mcp",
      "env": { "OPENAI_API_KEY": "your-key" }
    }
  }
}
```

Three tools show up: `evaluate`, `check_confidence`, `adversarial_check`. Provider is auto-detected from whichever API key you set.

## What you get back

The `Verdict` object:

```python
result.result              # whatever your agent originally returned
result.confidence          # 0.0 to 1.0
result.confidence_reason   # why it got that score
result.context_relevance   # 0.0 to 1.0, is this answering the right question?
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

CLI equivalents: `--confidence-threshold`, `--relevance-threshold`, `--no-require-defense`.

## Extend it

### Bring your own LLM

One method:

```python
from agent_verdict import LLMProvider, LLMMessage, LLMResponse

class MyProvider(LLMProvider):
    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        text = await call_whatever_llm(messages[0].content)
        return LLMResponse(content=text)
```

### Bring your own stages

```python
from agent_verdict import Stage, VerdictPipeline, ConfidenceStage

class MyStage(Stage):
    async def run(self, verdict, llm, task_context, config):
        return verdict.model_copy(update={"confidence": 0.99})

pipeline = VerdictPipeline(llm=llm, stages=[ConfidenceStage(), MyStage()])
```

## How many LLM calls?

- Confidence: 1
- Verification: 1
- Adversarial: 2 (attack + defend)
- **Full pipeline: 4 total**

Bad answers fail at step 1 and cost only 1 call.

## Running tests

```bash
git clone https://github.com/sharifli4/agent-verdict.git
cd agent-verdict
pip install -e ".[dev]"
pytest tests/ -v
```

Everything uses a mock provider. No API keys, no network calls.
