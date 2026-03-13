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

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

Options: `sh -s anthropic`, `sh -s openai`, `sh -s all`, `sh -s mcp` (for Claude Code/Cursor).

## Usage

### CLI

```bash
# full pipeline
agent-verdict evaluate "SQL injection on line 14" -c "Find security bugs"

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

Or use the pipeline directly:

```python
from agent_verdict import VerdictPipeline, VerdictConfig
from agent_verdict.llm.openai import OpenAIProvider

pipeline = VerdictPipeline(llm=OpenAIProvider(), config=VerdictConfig(confidence_threshold=0.7))
result = await pipeline.evaluate("race condition in pool", task_context="Find concurrency bugs")
```

### MCP (Claude Code / Cursor)

```bash
curl -fsSL https://raw.githubusercontent.com/sharifli4/agent-verdict/main/install.sh | sh -s mcp
claude mcp add agent-verdict -- /path/to/.venv/bin/agent-verdict-mcp
```

Tools: `evaluate` (customizable via `stages` param), `check_confidence`, `adversarial_check`, `self_consistency_check`, `semantic_similarity_check`, `entailment_check`, `logprob_check`.

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

Custom pipeline:

```python
from agent_verdict import VerdictPipeline, ConfidenceStage, EntailmentStage, AdversarialStage

pipeline = VerdictPipeline(llm=llm, stages=[
    ConfidenceStage(),
    EntailmentStage(),
    AdversarialStage(),
])
```

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
# Custom LLM provider
class MyProvider(LLMProvider):
    async def complete(self, messages):
        return LLMResponse(content=await my_llm(messages[0].content))

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
