"""
MCP server that exposes agent-verdict as tools for Claude Code, Cursor, etc.

Usage:
    python -m agent_verdict.mcp_server
    # or
    agent-verdict-mcp

Requires an LLM provider. Set environment variables:
    VERDICT_PROVIDER=anthropic|openai  (default: anthropic)
    VERDICT_MODEL=<model name>         (optional, uses provider default)
"""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from .models import VerdictConfig
from .stages import (
    AdversarialStage,
    ConfidenceStage,
    EntailmentStage,
    LogprobStage,
    SelfConsistencyStage,
    SemanticSimilarityStage,
    Stage,
    VerificationStage,
)

STAGE_REGISTRY: dict[str, type[Stage]] = {
    "confidence": ConfidenceStage,
    "verification": VerificationStage,
    "adversarial": AdversarialStage,
    "self_consistency": SelfConsistencyStage,
    "semantic_similarity": SemanticSimilarityStage,
    "entailment": EntailmentStage,
    "logprob": LogprobStage,
}

DEFAULT_STAGES = ["confidence", "verification", "adversarial"]

mcp = FastMCP(
    "agent-verdict",
    instructions=(
        "Tools for evaluating agent outputs with confidence scoring, "
        "independent verification, adversarial self-checking, self-consistency, "
        "semantic similarity, NLI entailment, and logprob calibration. "
        "Use 'evaluate' to run a customizable pipeline (pass 'stages' to pick which). "
        "Use individual tools (check_confidence, adversarial_check, self_consistency_check, "
        "semantic_similarity_check, entailment_check, logprob_check) to run a single stage."
    ),
)


def _detect_provider() -> str:
    """Auto-detect from VERDICT_PROVIDER env var, or from which API key is set."""
    explicit = os.environ.get("VERDICT_PROVIDER", "").lower()
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("MOONSHOT_API_KEY"):
        return "kimi"
    return "anthropic"


def _get_provider():
    provider_name = _detect_provider()
    model = os.environ.get("VERDICT_MODEL")
    base_url = os.environ.get("VERDICT_BASE_URL")
    api_key_env = os.environ.get("VERDICT_API_KEY_ENV")

    try:
        if provider_name == "anthropic":
            from .llm.anthropic import AnthropicProvider

            return AnthropicProvider(model=model) if model else AnthropicProvider()
        elif provider_name == "deepseek":
            from .llm.deepseek import DeepSeekProvider

            return DeepSeekProvider(model=model) if model else DeepSeekProvider()
        elif provider_name == "kimi":
            from .llm.kimi import KimiProvider

            return KimiProvider(model=model) if model else KimiProvider()
        else:
            from .llm.openai import OpenAIProvider

            kwargs: dict = {}
            if model:
                kwargs["model"] = model
            if base_url:
                kwargs["base_url"] = base_url
            if api_key_env:
                kwargs["api_key_env"] = api_key_env
            if base_url and not api_key_env:
                kwargs["supports_structured_output"] = False
            return OpenAIProvider(**kwargs)
    except ImportError as e:
        raise RuntimeError(
            f"Provider '{provider_name}' not installed. "
            f"Run: pip install 'agent-verdict[openai,mcp]'"
        ) from e


def _verdict_to_dict(verdict) -> dict:
    return verdict.model_dump()


def _build_stages(stage_names: list[str]) -> list[Stage]:
    """Build stage instances from a list of stage names."""
    stages = []
    for name in stage_names:
        cls = STAGE_REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown stage '{name}'. "
                f"Available: {', '.join(STAGE_REGISTRY)}"
            )
        stages.append(cls())
    return stages


@mcp.tool(
    name="evaluate",
    description=(
        "Run the verdict pipeline on an agent result. "
        "By default runs: confidence → verification → adversarial. "
        "Pass 'stages' to customize which stages run and in what order. "
        "Available stages: confidence, verification, adversarial, "
        "self_consistency, semantic_similarity, entailment, logprob. "
        "Returns a structured verdict with confidence, justification, "
        "counter-arguments, defense, and whether the result was dropped."
    ),
)
async def evaluate(
    result: str,
    task_context: str,
    stages: list[str] | None = None,
    confidence_threshold: float = 0.5,
    relevance_threshold: float = 0.4,
    require_defense: bool = True,
) -> str:
    from .pipeline import VerdictPipeline

    config = VerdictConfig(
        confidence_threshold=confidence_threshold,
        relevance_threshold=relevance_threshold,
        require_defense=require_defense,
    )
    stage_instances = _build_stages(stages or DEFAULT_STAGES)
    pipeline = VerdictPipeline(llm=_get_provider(), config=config, stages=stage_instances)
    verdict = await pipeline.evaluate(result, task_context=task_context)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


@mcp.tool(
    name="check_confidence",
    description=(
        "Quick confidence check on an agent result without running the full pipeline. "
        "Returns confidence score, relevance score, justification, and whether "
        "the result would be dropped based on thresholds."
    ),
)
async def check_confidence(
    result: str,
    task_context: str,
    confidence_threshold: float = 0.5,
    relevance_threshold: float = 0.4,
) -> str:
    from .pipeline import VerdictPipeline

    config = VerdictConfig(
        confidence_threshold=confidence_threshold,
        relevance_threshold=relevance_threshold,
    )
    pipeline = VerdictPipeline(
        llm=_get_provider(),
        config=config,
        stages=[ConfidenceStage()],
    )
    verdict = await pipeline.evaluate(result, task_context=task_context)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


@mcp.tool(
    name="adversarial_check",
    description=(
        "Run only the adversarial stage: generate the strongest counter-argument "
        "against a result, then attempt to defend it. Useful when you already "
        "trust the result but want to stress-test it."
    ),
)
async def adversarial_check(
    result: str,
    task_context: str,
    justification: str = "",
    require_defense: bool = True,
) -> str:
    from .models import Verdict

    config = VerdictConfig(require_defense=require_defense)
    llm = _get_provider()
    stage = AdversarialStage()
    verdict = Verdict(result=result, justification=justification)
    verdict = await stage.run(verdict, llm, task_context, config)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


@mcp.tool(
    name="self_consistency_check",
    description=(
        "Self-Consistency check (Wang et al. 2022): sample N independent answers "
        "to the same task and measure how many agree with the agent's result. "
        "High agreement = high confidence. Catches unstable/unreliable answers."
    ),
)
async def self_consistency_check(
    result: str,
    task_context: str,
    num_samples: int = 3,
    confidence_threshold: float = 0.5,
) -> str:
    from .models import Verdict

    config = VerdictConfig(confidence_threshold=confidence_threshold)
    llm = _get_provider()
    stage = SelfConsistencyStage(num_samples=num_samples)
    verdict = Verdict(result=result)
    verdict = await stage.run(verdict, llm, task_context, config)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


@mcp.tool(
    name="semantic_similarity_check",
    description=(
        "Check semantic similarity between the agent's result and an independent "
        "re-derivation using sentence embeddings (MiniLM). Catches off-topic answers. "
        "Requires: pip install 'agent-verdict[embeddings]'"
    ),
)
async def semantic_similarity_check(
    result: str,
    task_context: str,
    confidence_threshold: float = 0.5,
) -> str:
    from .models import Verdict

    config = VerdictConfig(confidence_threshold=confidence_threshold)
    llm = _get_provider()
    stage = SemanticSimilarityStage()
    verdict = Verdict(result=result)
    verdict = await stage.run(verdict, llm, task_context, config)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


@mcp.tool(
    name="entailment_check",
    description=(
        "NLI entailment check using DeBERTa-v3: verifies whether an independent "
        "re-derivation entails the agent's answer. Catches hallucinated or "
        "contradicting answers. Requires: pip install 'agent-verdict[nli]'"
    ),
)
async def entailment_check(
    result: str,
    task_context: str,
    confidence_threshold: float = 0.5,
) -> str:
    from .models import Verdict

    config = VerdictConfig(confidence_threshold=confidence_threshold)
    llm = _get_provider()
    stage = EntailmentStage()
    verdict = Verdict(result=result)
    verdict = await stage.run(verdict, llm, task_context, config)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


@mcp.tool(
    name="logprob_check",
    description=(
        "Token log-probability calibration: measures the LLM's internal certainty "
        "about the answer via exp(mean_logprob). Catches internally uncertain answers. "
        "Requires OpenAI provider."
    ),
)
async def logprob_check(
    result: str,
    task_context: str,
    confidence_threshold: float = 0.5,
) -> str:
    from .models import Verdict

    config = VerdictConfig(confidence_threshold=confidence_threshold)
    llm = _get_provider()
    stage = LogprobStage()
    verdict = Verdict(result=result)
    verdict = await stage.run(verdict, llm, task_context, config)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
