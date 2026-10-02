from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .wall import ParametricPoint


class ProjectAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str | None = None
    orientation_deg: float
    evidence_ids: list[str] = Field(default_factory=list)


class ViewAxisOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    project_axis_id: str = Field(min_length=1)
    level_id: str = Field(min_length=1)
    start: ParametricPoint
    end: ParametricPoint
    visible: bool = True
    evidence_ids: list[str] = Field(default_factory=list)
