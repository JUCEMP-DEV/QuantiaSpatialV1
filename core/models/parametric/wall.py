from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ParametricPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x_px: float
    y_px: float


class ParametricWall(BaseModel):
    """Muro editable canónico de Quantia."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_id: str = Field(min_length=1)
    role: Literal["PERIMETER", "DIVIDER"]
    reference_path: list[ParametricPoint] = Field(min_length=2)
    anchor_mode: Literal["CENTERLINE", "FACE", "F02_FIXED"]
    length_px: float = Field(gt=0.0)
    length_m: float | None = Field(default=None, gt=0.0)
    thickness_px: float | None = Field(default=None, gt=0.0)
    thickness_m: float | None = Field(default=None, gt=0.0)
    height_m: float | None = Field(default=None, gt=0.0)
    source_candidate_id: str | None = None
    source_f02_wall_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    editable: bool = True
    confirmed: Literal[False] = False
