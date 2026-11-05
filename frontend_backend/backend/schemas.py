from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

Availability = Literal["available", "unavailable", "not_checked", "error"]

class Envelope(BaseModel):
    availability: Availability
    source: str | None = None
    updated_at: str | None = None
    data: Any = None
    reason: str | None = None

class JobStartRequest(BaseModel):
    preset: str
    run_name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    flags: dict[str, Any] = Field(default_factory=dict)
    base_manifest: str | None = None

class PrepRunRequest(BaseModel):
    data: str = "data/xauusd_h1.csv"
    train_end: str = "2022-01-01"
    window: int = Field(default=64, ge=1, le=4096)
    macro: str | None = None

class RiskConfigRequest(BaseModel):
    changes: dict[str, Any]

class KillSwitchRequest(BaseModel):
    armed: bool
