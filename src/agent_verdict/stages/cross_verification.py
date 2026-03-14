from __future__ import annotations

import asyncio

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import (
    JurorDeliberationOutput,
    JurorPosition,
    JurorPositionOutput,
    LLMMessage,
    Verdict,
    VerdictConfig,
)

from .base import DATA_BOUNDARY_INSTRUCTION, Stage, sanitize_for_prompt

POSITION_PROMPT = """\
You are Juror "{juror_name}", an independent evaluator. You must decide whether \
to SUPPORT or CHALLENGE the following agent result.

{data_boundary}

{task_context}

{result}

Instructions:
1. State your vote: "support" or "challenge"
2. Present your argument for your position
3. Steel-man the other side: present the strongest counter-argument against \
your OWN position (be honest, not defensive)
4. Rate your confidence 0.0-1.0"""

DELIBERATION_PROMPT = """\
You are Juror "{juror_name}". You previously voted to {own_vote} the agent's result.

Your argument was:
{own_argument}

Now the other jurors have presented their positions:

{other_positions}

{data_boundary}

{task_context}

{result}

Having heard all arguments, state your FINAL vote ("support" or "challenge"). \
You may change your mind if another juror made a compelling point. \
Write a brief rebuttal addressing the strongest opposing argument. \
Rate your final confidence 0.0-1.0."""


def _juror_name(llm: LLMProvider) -> str:
    """Derive a display name from the provider."""
    model = getattr(llm, "model", None)
    if model:
        return str(model)
    return type(llm).__name__


def _format_other_positions(
    positions: list[JurorPosition], exclude_juror: str
) -> str:
    """Format all other jurors' positions for the deliberation prompt."""
    lines = []
    for p in positions:
        if p.juror == exclude_juror:
            continue
        lines.append(
            f"--- {p.juror} (votes: {p.vote}, confidence: {p.confidence:.2f}) ---\n"
            f"Argument: {p.argument}\n"
            f"Self-counter: {p.counter_to_self}"
        )
    return "\n\n".join(lines)


async def _get_position(
    llm: LLMProvider,
    juror_name: str,
    verdict: Verdict,
    task_context: str,
) -> JurorPosition:
    """Get a single juror's initial position (runs in parallel)."""
    prompt = POSITION_PROMPT.format(
        juror_name=juror_name,
        data_boundary=DATA_BOUNDARY_INSTRUCTION,
        task_context=sanitize_for_prompt(task_context, "task_context"),
        result=sanitize_for_prompt(verdict.result, "agent_result"),
    )
    data = await llm.complete_structured(
        [LLMMessage(role="user", content=prompt)],
        JurorPositionOutput,
    )

    vote = str(data.get("vote", "support")).lower().strip()
    if vote not in ("support", "challenge"):
        vote = "support"

    return JurorPosition(
        juror=juror_name,
        vote=vote,
        argument=str(data.get("argument", "")),
        counter_to_self=str(data.get("counter_to_self", "")),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
    )


async def _get_deliberation(
    llm: LLMProvider,
    juror_name: str,
    position: JurorPosition,
    all_positions: list[JurorPosition],
    verdict: Verdict,
    task_context: str,
) -> JurorPosition:
    """Get a juror's final vote after seeing all positions (runs in parallel)."""
    prompt = DELIBERATION_PROMPT.format(
        juror_name=juror_name,
        own_vote=position.vote,
        own_argument=position.argument,
        other_positions=_format_other_positions(all_positions, juror_name),
        data_boundary=DATA_BOUNDARY_INSTRUCTION,
        task_context=sanitize_for_prompt(task_context, "task_context"),
        result=sanitize_for_prompt(verdict.result, "agent_result"),
    )
    data = await llm.complete_structured(
        [LLMMessage(role="user", content=prompt)],
        JurorDeliberationOutput,
    )

    final_vote = str(data.get("final_vote", position.vote)).lower().strip()
    if final_vote not in ("support", "challenge"):
        final_vote = position.vote

    return position.model_copy(update={
        "final_vote": final_vote,
        "rebuttal": str(data.get("rebuttal", "")),
        "confidence": max(0.0, min(1.0, float(data.get("confidence", position.confidence)))),
    })


class CrossVerificationStage(Stage):
    """Multi-model jury deliberation.

    Each challenger LLM independently evaluates the result, then all jurors
    see each other's positions and cast a final vote. All LLM calls within
    each phase run in parallel.

    Usage:
        CrossVerificationStage(challengers=[DeepSeekProvider(), KimiProvider()])
    """

    def __init__(self, challengers: list[LLMProvider] | None = None):
        self.challengers = challengers or []

    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        # Build jury: primary LLM + all challengers (disambiguate duplicate names)
        jury: list[tuple[str, LLMProvider]] = []
        seen: dict[str, int] = {}
        for provider in [llm, *self.challengers]:
            name = _juror_name(provider)
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                name = f"{name} #{seen[name]}"
            jury.append((name, provider))

        if len(jury) < 2:
            return verdict

        # Phase 1: All jurors state positions in parallel
        position_tasks = [
            _get_position(provider, name, verdict, task_context)
            for name, provider in jury
        ]
        positions: list[JurorPosition] = await asyncio.gather(*position_tasks)

        # Phase 2: All jurors deliberate in parallel (seeing everyone's positions)
        deliberation_tasks = [
            _get_deliberation(provider, name, pos, positions, verdict, task_context)
            for (name, provider), pos in zip(jury, positions)
        ]
        final_positions: list[JurorPosition] = await asyncio.gather(*deliberation_tasks)

        # Synthesize: weighted vote
        support_weight = 0.0
        challenge_weight = 0.0
        for p in final_positions:
            if p.final_vote == "support":
                support_weight += p.confidence
            else:
                challenge_weight += p.confidence

        total_weight = support_weight + challenge_weight
        support_ratio = support_weight / total_weight if total_weight > 0 else 0.5

        support_count = sum(1 for p in final_positions if p.final_vote == "support")
        challenge_count = len(final_positions) - support_count
        jury_size = len(final_positions)

        # Build confidence reason
        voter_summary = ", ".join(
            f"{p.juror}: {p.final_vote} ({p.confidence:.2f})"
            for p in final_positions
        )
        reason = f"Jury vote: {support_count}/{jury_size} support. {voter_summary}"

        # Blend with existing confidence
        blended = min(verdict.confidence, support_ratio) if verdict.confidence > 0 else support_ratio

        updates: dict = {
            "deliberation": final_positions,
            "confidence": blended,
            "confidence_reason": reason,
        }

        if challenge_count > support_count:
            updates["dropped"] = True
            updates["drop_reason"] = (
                f"Jury rejected: {challenge_count}/{jury_size} voted to challenge"
            )

        return verdict.model_copy(update=updates)
