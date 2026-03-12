from __future__ import annotations

from agent_verdict.llm.base import LLMProvider
from agent_verdict.models import LLMMessage, Verdict, VerdictConfig

from .base import Stage, parse_llm_json

COUNTER_PROMPT = """\
You are a critical adversarial reviewer. Find the strongest counter-argument against \
the following agent result. Be rigorous and look for flaws, assumptions, or errors.

Task context: {task_context}
Agent result: {result}
Agent justification: {justification}

Respond with JSON only:
{{
  "counter_argument": "<strongest counter-argument>"
}}"""

DEFENSE_PROMPT = """\
You are defending an agent's result against a counter-argument. Provide a rigorous \
defense if possible. If the counter-argument is valid and the result is wrong, admit it.

Task context: {task_context}
Agent result: {result}
Agent justification: {justification}
Counter-argument: {counter_argument}

Respond with JSON only:
{{
  "defense": "<your defense or admission of fault>",
  "defended": <true/false>
}}"""

COUNTER_DEFAULTS = {
    "counter_argument": "Failed to generate counter-argument",
}

DEFENSE_DEFAULTS = {
    "defense": "Failed to generate defense",
    "defended": False,
}


class AdversarialStage(Stage):
    async def run(
        self,
        verdict: Verdict,
        llm: LLMProvider,
        task_context: str,
        config: VerdictConfig,
    ) -> Verdict:
        # Step 1: Generate counter-argument
        counter_prompt = COUNTER_PROMPT.format(
            task_context=task_context,
            result=verdict.result,
            justification=verdict.justification,
        )
        counter_response = await llm.complete(
            [LLMMessage(role="user", content=counter_prompt)]
        )
        counter_data = parse_llm_json(counter_response.content, COUNTER_DEFAULTS)
        counter_argument = str(counter_data.get("counter_argument", ""))

        # Step 2: Defend against counter-argument
        defense_prompt = DEFENSE_PROMPT.format(
            task_context=task_context,
            result=verdict.result,
            justification=verdict.justification,
            counter_argument=counter_argument,
        )
        defense_response = await llm.complete(
            [LLMMessage(role="user", content=defense_prompt)]
        )
        defense_data = parse_llm_json(defense_response.content, DEFENSE_DEFAULTS)

        defended = bool(defense_data.get("defended", False))
        defense = str(defense_data.get("defense", ""))

        updates: dict = {
            "counter_argument": counter_argument,
            "defense": defense,
            "defended": defended,
        }

        if config.require_defense and not defended:
            updates["dropped"] = True
            updates["drop_reason"] = "Result could not defend against counter-argument"

        return verdict.model_copy(update=updates)
