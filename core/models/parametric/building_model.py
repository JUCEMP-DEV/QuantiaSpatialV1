from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .axis import ProjectAxis
from .level import ParametricLevel


class QuantiaParametricModel(BaseModel):
    """Verdad editable del Reconstruction Core; IFC/DXF/SVG serán adaptadores."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    levels: list[ParametricLevel] = Field(default_factory=list)
    project_axes: list[ProjectAxis] = Field(default_factory=list)
    model_revision: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
