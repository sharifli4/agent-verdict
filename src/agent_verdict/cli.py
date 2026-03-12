"""
CLI for agent-verdict.

Usage:
    agent-verdict evaluate "your agent's answer" --context "what it was supposed to do"
    echo "agent output" | agent-verdict evaluate --context "task description"
    agent-verdict check "quick confidence test" --context "task"
    agent-verdict attack "stress test this answer" --context "task"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .models import Verdict, VerdictConfig


# --- colors ---
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{RESET}"


def _get_provider(provider_name: str, model: str | None):
    if provider_name == "openai":
        from .llm.openai import OpenAIProvider
        return OpenAIProvider(model=model) if model else OpenAIProvider()
    else:
        from .llm.anthropic import AnthropicProvider
        return AnthropicProvider(model=model) if model else AnthropicProvider()


def _print_verdict(v: Verdict, verbose: bool = False) -> None:
    """Pretty-print a verdict to the terminal."""

    if v.dropped:
        print()
        print(_color("  DROPPED", RED + BOLD))
        print(f"  {_color('reason:', DIM)} {_color(v.drop_reason, RED)}")
        print()
    else:
        print()
        print(_color("  PASSED", GREEN + BOLD))
        print()

    # scores
    conf_color = GREEN if v.confidence >= 0.5 else RED
    rel_color = GREEN if v.context_relevance >= 0.5 else RED
    print(f"  {_color('confidence:', DIM)}   {_color(f'{v.confidence:.2f}', conf_color)}")
    print(f"  {_color('relevance:', DIM)}    {_color(f'{v.context_relevance:.2f}', rel_color)}")

    if v.justification:
        print(f"  {_color('justification:', DIM)} {v.justification}")

    if v.counter_argument:
        print()
        print(f"  {_color('counter-arg:', DIM)}  {_color(v.counter_argument, YELLOW)}")
        print(f"  {_color('defense:', DIM)}      {_color(v.defense, GREEN if v.defended else RED)}")
        defended_str = _color("yes", GREEN) if v.defended else _color("no", RED)
        print(f"  {_color('defended:', DIM)}     {defended_str}")

    if verbose and v.confidence_reason:
        print()
        print(f"  {_color('confidence reason:', DIM)} {v.confidence_reason}")

    print()


def _read_result(args) -> str:
    """Get the result text from args or stdin."""
    if args.result:
        return " ".join(args.result)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Error: provide a result as an argument or pipe it via stdin.", file=sys.stderr)
    sys.exit(1)


async def _run_evaluate(args) -> Verdict:
    from .pipeline import VerdictPipeline

    config = VerdictConfig(
        confidence_threshold=args.confidence_threshold,
        relevance_threshold=args.relevance_threshold,
        require_defense=args.require_defense,
    )
    llm = _get_provider(args.provider, args.model)
    pipeline = VerdictPipeline(llm=llm, config=config)
    return await pipeline.evaluate(_read_result(args), task_context=args.context)


async def _run_check(args) -> Verdict:
    from .pipeline import VerdictPipeline
    from .stages import ConfidenceStage

    config = VerdictConfig(
        confidence_threshold=args.confidence_threshold,
        relevance_threshold=args.relevance_threshold,
    )
    llm = _get_provider(args.provider, args.model)
    pipeline = VerdictPipeline(llm=llm, config=config, stages=[ConfidenceStage()])
    return await pipeline.evaluate(_read_result(args), task_context=args.context)


async def _run_attack(args) -> Verdict:
    from .models import Verdict as VerdictModel
    from .stages import AdversarialStage

    config = VerdictConfig(require_defense=args.require_defense)
    llm = _get_provider(args.provider, args.model)
    stage = AdversarialStage()
    result_text = _read_result(args)
    verdict = VerdictModel(result=result_text, justification=args.justification or "")
    return await stage.run(verdict, llm, args.context, config)


def main():
    parser = argparse.ArgumentParser(
        prog="agent-verdict",
        description="Check if your agent's answer is actually good.",
    )
    parser.add_argument(
        "-p", "--provider", default="anthropic", choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument("-m", "--model", default=None, help="model name")
    parser.add_argument("--json", action="store_true", help="output raw JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="show extra details")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", aliases=["eval", "e"], help="full pipeline: confidence + verification + adversarial")
    p_eval.add_argument("result", nargs="*", help="the agent result to evaluate (or pipe via stdin)")
    p_eval.add_argument("-c", "--context", required=True, help="what the agent was supposed to do")
    p_eval.add_argument("--confidence-threshold", type=float, default=0.5)
    p_eval.add_argument("--relevance-threshold", type=float, default=0.4)
    p_eval.add_argument("--require-defense", action=argparse.BooleanOptionalAction, default=True)

    # --- check ---
    p_check = sub.add_parser("check", aliases=["c"], help="quick confidence check only (1 LLM call)")
    p_check.add_argument("result", nargs="*", help="the agent result to check (or pipe via stdin)")
    p_check.add_argument("-c", "--context", required=True, help="what the agent was supposed to do")
    p_check.add_argument("--confidence-threshold", type=float, default=0.5)
    p_check.add_argument("--relevance-threshold", type=float, default=0.4)

    # --- attack ---
    p_attack = sub.add_parser("attack", aliases=["a"], help="adversarial check only: counter-argue then defend")
    p_attack.add_argument("result", nargs="*", help="the agent result to attack (or pipe via stdin)")
    p_attack.add_argument("-c", "--context", required=True, help="what the agent was supposed to do")
    p_attack.add_argument("-j", "--justification", default="", help="justification to defend against")
    p_attack.add_argument("--require-defense", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    # run
    if args.command in ("evaluate", "eval", "e"):
        verdict = asyncio.run(_run_evaluate(args))
    elif args.command in ("check", "c"):
        verdict = asyncio.run(_run_check(args))
    elif args.command in ("attack", "a"):
        verdict = asyncio.run(_run_attack(args))
    else:
        parser.print_help()
        sys.exit(1)

    # output
    if args.json:
        print(json.dumps(verdict.model_dump(), indent=2))
    else:
        _print_verdict(verdict, verbose=args.verbose)

    sys.exit(1 if verdict.dropped else 0)
