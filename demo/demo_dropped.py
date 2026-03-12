"""Demo: what happens when an agent's answer gets dropped."""
import json
from agent_verdict import verdict, DroppedResultError, LLMProvider, LLMMessage, LLMResponse

class WeakProvider(LLMProvider):
    def __init__(self):
        self._responses = iter([
            json.dumps({
                "confidence": 0.18,
                "confidence_reason": "The answer is vague and doesn't reference specific code",
                "context_relevance": 0.22,
                "justification": "Generic security advice with no concrete findings",
            }),
        ])

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        return LLMResponse(content=next(self._responses))


llm = WeakProvider()

@verdict(llm=llm, task_context="Find security vulnerabilities in a Python web app")
def bad_agent(code: str) -> str:
    return "There might be some security issues, you should probably check the code"


print("Running a weak agent with @verdict decorator...\n")
try:
    result = bad_agent("def login(user, pw): ...")
except DroppedResultError as e:
    print(f"  DROPPED!")
    print(f"  Reason:      {e.verdict.drop_reason}")
    print(f"  Confidence:  {e.verdict.confidence}")
    print(f"  Relevance:   {e.verdict.context_relevance}")
