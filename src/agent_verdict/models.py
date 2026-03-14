from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


class VerdictConfig(BaseModel):
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    require_defense: bool = True


class JurorPosition(BaseModel):
    """A single juror's position in cross-verification deliberation."""
    juror: str = ""
    vote: str = ""  # "support" or "challenge"
    argument: str = ""
    counter_to_self: str = ""  # steel-man against own position
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_vote: str = ""  # after seeing all positions
    rebuttal: str = ""  # response to other jurors


class StageUsage(BaseModel):
    """Token usage and cost for a single pipeline stage."""
    stage: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cost: float = 0.0  # USD


class Verdict(BaseModel):
    result: Any = None
    justification: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_reason: str = ""
    context_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    counter_argument: str = ""
    defense: str = ""
    defended: bool = False
    dropped: bool = False
    drop_reason: str = ""
    deliberation: list[JurorPosition] = Field(default_factory=list)
    usage: list[StageUsage] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.usage)

    @property
    def total_cost(self) -> float:
        return sum(u.cost for u in self.usage)


# --- Structured output schemas for each stage ---


class ConfidenceOutput(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_reason: str
    context_relevance: float = Field(ge=0.0, le=1.0)
    justification: str


class VerificationOutput(BaseModel):
    verified: bool
    verification_reason: str
    adjusted_confidence: float = Field(ge=0.0, le=1.0)


class CounterArgumentOutput(BaseModel):
    counter_argument: str


class DefenseOutput(BaseModel):
    defense: str
    defended: bool


class JurorPositionOutput(BaseModel):
    vote: str  # "support" or "challenge"
    argument: str
    counter_to_self: str
    confidence: float = Field(ge=0.0, le=1.0)


class JurorDeliberationOutput(BaseModel):
    final_vote: str  # "support" or "challenge"
    rebuttal: str
    confidence: float = Field(ge=0.0, le=1.0)
