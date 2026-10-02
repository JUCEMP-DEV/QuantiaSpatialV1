from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .axis import ViewAxisOccurrence
from .opening import ParametricOpening
from .wall import ParametricWall


class ParametricSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_id: str = Field(min_length=1)
    name: str | None = None
    boundary_wall_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ParametricStair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_id: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ParametricLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str | None = None
    source_level_view_id: str = Field(min_length=1)
    source_f02_model_id: str = Field(min_length=1)
    walls: list[ParametricWall] = Field(default_factory=list)
    axis_occurrences: list[ViewAxisOccurrence] = Field(default_factory=list)
    openings: list[ParametricOpening] = Field(default_factory=list)
    spaces: list[ParametricSpace] = Field(default_factory=list)
    stairs: list[ParametricStair] = Field(default_factory=list)
