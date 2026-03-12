"""Demo: using @verdict decorator to check an agent's answer."""
import json
from agent_verdict import verdict, DroppedResultError, LLMProvider, LLMMessage, LLMResponse

# -- mock provider so the demo runs without API keys --
class DemoProvider(LLMProvider):
    def __init__(self):
        self._responses = iter([
            json.dumps({
                "confidence": 0.91,
                "confidence_reason": "The analysis correctly identifies a classic SQL injection pattern with string concatenation in the query",
                "context_relevance": 0.88,
                "justification": "SQL injection via unsanitized user input in login query",
            }),
            json.dumps({
                "verified": True,
                "verification_reason": "Independently confirmed: the f-string in line 14 passes user input directly into the WHERE clause",
                "adjusted_confidence": 0.89,
            }),
            json.dumps({
                "counter_argument": "The input might be sanitized upstream by middleware before reaching this function",
            }),
            json.dumps({
                "defense": "No sanitization middleware exists — raw input flows directly into the query",
                "defended": True,
            }),
        ])

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        return LLMResponse(content=next(self._responses))


llm = DemoProvider()

@verdict(llm=llm, task_context="Find security vulnerabilities in a Python web app")
def analyze(code: str) -> str:
    return "SQL injection in login handler: user input concatenated into query on line 14"


print("Running agent with @verdict decorator...\n")
result = analyze("def login(user, pw): db.execute(f'SELECT * FROM users WHERE name={user}')")

print(f"  Result:       {result.result}")
print(f"  Confidence:   {result.confidence}")
print(f"  Relevance:    {result.context_relevance}")
print(f"  Justified:    {result.justification}")
print(f"  Counter-arg:  {result.counter_argument}")
print(f"  Defense:      {result.defense}")
print(f"  Defended:     {result.defended}")
print(f"  Dropped:      {result.dropped}")
