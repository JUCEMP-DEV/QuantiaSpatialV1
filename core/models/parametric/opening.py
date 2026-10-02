from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ParametricOpening(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_id: str = Field(min_length=1)
    host_wall_id: str = Field(min_length=1)
    opening_type: Literal["OPENING", "DOOR", "WINDOW"]
    offset_px: float = Field(ge=0.0)
    width_px: float = Field(gt=0.0)
    width_m: float | None = Field(default=None, gt=0.0)
    evidence_ids: list[str] = Field(default_factory=list)
