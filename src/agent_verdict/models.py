from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str


class VerdictConfig(BaseModel):
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    require_defense: bool = True


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
