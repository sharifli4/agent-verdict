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

mcp = FastMCP(
    "agent-verdict",
    instructions=(
        "Tools for evaluating agent outputs with confidence scoring, "
        "independent verification, and adversarial self-checking. "
        "Use 'evaluate' to run the full pipeline on any result you want to verify. "
        "Use 'check_confidence' for a quick confidence check without the full pipeline."
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
    return "anthropic"


def _get_provider():
    provider_name = _detect_provider()
    model = os.environ.get("VERDICT_MODEL")

    try:
        if provider_name == "openai":
            from .llm.openai import OpenAIProvider

            return OpenAIProvider(model=model) if model else OpenAIProvider()
        else:
            from .llm.anthropic import AnthropicProvider

            return AnthropicProvider(model=model) if model else AnthropicProvider()
    except ImportError as e:
        raise RuntimeError(
            f"Provider '{provider_name}' not installed. "
            f"Run: pip install 'agent-verdict[{provider_name},mcp]'"
        ) from e


def _verdict_to_dict(verdict) -> dict:
    return verdict.model_dump()


@mcp.tool(
    name="evaluate",
    description=(
        "Run the full verdict pipeline on an agent result: confidence scoring, "
        "independent verification, and adversarial counter-argument/defense. "
        "Returns a structured verdict with confidence, justification, "
        "counter-arguments, defense, and whether the result was dropped."
    ),
)
async def evaluate(
    result: str,
    task_context: str,
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
    pipeline = VerdictPipeline(llm=_get_provider(), config=config)
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
    from .stages import ConfidenceStage

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
    from .stages import AdversarialStage

    config = VerdictConfig(require_defense=require_defense)
    llm = _get_provider()
    stage = AdversarialStage()
    verdict = Verdict(result=result, justification=justification)
    verdict = await stage.run(verdict, llm, task_context, config)
    return json.dumps(_verdict_to_dict(verdict), indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
