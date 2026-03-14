# agent-verdict

<p align="center">
  <img src="demo/demo.gif" alt="agent-verdict demo" width="800">
</p>

Verify your AI agent's answers. Confidence scoring, independent verification, and adversarial stress-testing in one pipeline.

```
Agent's answer → Confidence check → Verification → Adversarial attack → Verdict
```

If it fails at any step, the rest don't run. No wasted LLM calls.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/sharifli4/agent-verdict/main/install.sh | sh
```

Options: `sh -s anthropic`, `sh -s openai`, `sh -s deepseek`, `sh -s kimi`, `sh -s all`, `sh -s mcp`.

## Providers

Works with any LLM. Built-in support for:

| Provider | API key env var | Default model | Install extra |
|----------|----------------|---------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | `agent-verdict[anthropic]` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` | `agent-verdict[openai]` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` | `agent-verdict[deepseek]` |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` | `kimi-k2.5` | `agent-verdict[kimi]` |

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
# or
export DEEPSEEK_API_KEY=sk-...
# or
export MOONSHOT_API_KEY=sk-...
```

**Any OpenAI-compatible API** works too — just pass `--base-url` and `--api-key-env`:

```bash
# use any OpenAI-compatible provider
agent-verdict -p openai --base-url https://api.example.com/v1 --api-key-env MY_API_KEY \
    eval "result" -c "task"
```

## Usage

### CLI

```bash
# full pipeline (auto-detects provider from API key)
agent-verdict evaluate "SQL injection on line 14" -c "Find security bugs"

# pick a specific provider
agent-verdict -p deepseek eval "SQL injection on line 14" -c "Find security bugs"

# pick a specific model
agent-verdict -p openai -m gpt-4o-mini eval "result" -c "task"

# pipe from another tool
my-agent analyze code.py | agent-verdict eval -c "Find security bugs"

# quick confidence check (1 LLM call)
agent-verdict check "server crashed due to OOM" -c "Diagnose outage"

# adversarial only
agent-verdict attack "race condition in pool.get()" -c "Find concurrency bugs"

# JSON output
agent-verdict --json eval "result" -c "task" > verdict.json
```

Exit code `0` = passed, `1` = dropped.

### Python

```python
from agent_verdict import verdict
from agent_verdict.llm.anthropic import AnthropicProvider

llm = AnthropicProvider()

@verdict(llm=llm, task_context="Find security bugs in Python code")
def analyze(code: str) -> str:
    return "Found SQL injection in the login handler"

result = analyze(user_code)
result.confidence        # 0.87
result.defended          # True
result.dropped           # False
```

Use any provider:

```python
from agent_verdict.llm.openai import OpenAIProvider
from agent_verdict.llm.deepseek import DeepSeekProvider
from agent_verdict.llm.kimi import KimiProvider

llm = DeepSeekProvider()                     # uses DEEPSEEK_API_KEY
llm = KimiProvider(model="kimi-k2.5")       # uses MOONSHOT_API_KEY

# any OpenAI-compatible API
llm = OpenAIProvider(
    model="my-model",
    base_url="https://api.example.com/v1",
    api_key_env="MY_API_KEY",
)
```

Or use the pipeline directly:

```python
from agent_verdict import VerdictPipeline, VerdictConfig

pipeline = VerdictPipeline(llm=llm, config=VerdictConfig(confidence_threshold=0.7))
result = await pipeline.evaluate("race condition in pool", task_context="Find concurrency bugs")
```

### MCP (Claude Code / Cursor)

```bash
curl -fsSL https://raw.githubusercontent.com/sharifli4/agent-verdict/main/install.sh | sh -s mcp
claude mcp add agent-verdict -- /path/to/.venv/bin/agent-verdict-mcp
```

Configure via env vars: `VERDICT_PROVIDER`, `VERDICT_MODEL`, `VERDICT_BASE_URL`, `VERDICT_API_KEY_ENV`.

Tools: `evaluate` (customizable via `stages` param), `check_confidence`, `adversarial_check`, `self_consistency_check`, `semantic_similarity_check`, `entailment_check`, `logprob_check`, `cross_verification`.

## Verdict object

```python
result.confidence          # 0.0-1.0
result.context_relevance   # 0.0-1.0
result.justification       # why the answer makes sense
result.counter_argument    # best attack against the answer
result.defense             # response to that attack
result.defended            # did the defense hold?
result.dropped             # was the answer rejected?
result.drop_reason         # why it was rejected
result.deliberation        # list[JurorPosition] — cross-verification jury
```

## Evaluation algorithms

| Algorithm | Technique | Catches |
|-----------|-----------|---------|
| **Confidence Scoring** | LLM rates confidence and relevance 0.0-1.0 | Low-quality or vague answers |
| **Independent Verification** | LLM re-derives answer without seeing the original | Answers that don't hold up independently |
| **Adversarial Dialectic** | Generate counter-argument, then defend against it | Plausible but flawed answers |
| **Self-Consistency** | [Wang et al. 2022](https://arxiv.org/abs/2203.11171) — sample N answers, measure agreement | Unstable/unreliable answers |
| **Cosine Similarity** | Sentence embeddings ([MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) | Off-topic answers |
| **NLI Entailment** | [DeBERTa-v3](https://huggingface.co/cross-encoder/nli-deberta-v3-small) classification | Hallucinated/contradicting answers |
| **Logprob Calibration** | Token log-probabilities via `exp(mean_logprob)` | Internally uncertain answers |
| **Cross-Verification** | Multi-model jury deliberation with position, counter, and rebuttal | Answers that fool one model but not others |

Default pipeline uses the first 3 (4 LLM calls). Stages 5-7 use different models, breaking the "same brain grading itself" problem.

## Stages

| Stage | LLM calls | Install |
|-------|-----------|---------|
| `ConfidenceStage` | 1 | included |
| `VerificationStage` | 1 | included |
| `AdversarialStage` | 2 | included |
| `SelfConsistencyStage(n)` | 2n | included |
| `SemanticSimilarityStage` | 0 | `pip install agent-verdict[embeddings]` |
| `EntailmentStage` | 0 | `pip install agent-verdict[nli]` |
| `LogprobStage` | 1 | OpenAI only |
| `CrossVerificationStage` | 2 per juror | needs 2+ providers |

Custom pipeline:

```python
from agent_verdict import VerdictPipeline, ConfidenceStage, EntailmentStage, AdversarialStage

pipeline = VerdictPipeline(llm=llm, stages=[
    ConfidenceStage(),
    EntailmentStage(),
    AdversarialStage(),
])
```

## Cross-verification (multi-model jury)

Different LLMs deliberate on your agent's answer like a jury. Each juror:

1. **States position** — support or challenge, with argument
2. **Steel-mans the other side** — strongest counter to their own position
3. **Deliberates** — sees all positions, writes rebuttal, casts final vote

All jurors run in parallel. Wall-clock time ≈ one slow LLM call per phase.

```python
from agent_verdict import VerdictPipeline, CrossVerificationStage
from agent_verdict.llm.anthropic import AnthropicProvider
from agent_verdict.llm.openai import OpenAIProvider
from agent_verdict.llm.deepseek import DeepSeekProvider

pipeline = VerdictPipeline(
    llm=AnthropicProvider(),
    stages=[
        CrossVerificationStage(challengers=[
            OpenAIProvider(),
            DeepSeekProvider(),
        ]),
    ],
)

result = await pipeline.evaluate("the agent's answer", task_context="what it should do")

# inspect the jury deliberation
for juror in result.deliberation:
    print(f"{juror.juror}: {juror.vote} → {juror.final_vote} ({juror.confidence:.2f})")
    print(f"  argument: {juror.argument}")
    print(f"  steel-man: {juror.counter_to_self}")
    print(f"  rebuttal: {juror.rebuttal}")
```

Majority vote decides. If more jurors challenge than support, the result is dropped.

## Configuration

```python
VerdictConfig(
    confidence_threshold=0.5,   # below → dropped
    relevance_threshold=0.4,    # below → dropped
    require_defense=True,       # can't defend → dropped
)
```

CLI: `--confidence-threshold`, `--relevance-threshold`, `--no-require-defense`.

## Extend

```python
# Custom LLM provider — implement one method
class MyProvider(LLMProvider):
    async def complete(self, messages):
        return LLMResponse(content=await my_llm(messages[0].content))

# Or subclass OpenAIProvider for any OpenAI-compatible API
class MyProvider(OpenAIProvider):
    def __init__(self):
        super().__init__(
            model="my-model",
            base_url="https://api.example.com/v1",
            api_key_env="MY_API_KEY",
        )

# Custom stage
class MyStage(Stage):
    async def run(self, verdict, llm, task_context, config):
        return verdict.model_copy(update={"confidence": 0.99})
```

## Development

```bash
git clone https://github.com/sharifli4/agent-verdict.git
cd agent-verdict
pip install -e ".[dev]"
pytest tests/ -v    # no API keys needed, uses mock provider
```
