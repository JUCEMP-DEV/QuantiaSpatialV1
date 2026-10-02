from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .drawing_model import DrawingPoint


CandidateGenerator = Literal["DOUBLE_FACE", "REGION_CENTERLINE"]
RelationType = Literal[
    "JUNCTION",
    "CONTINUATION",
    "CROSSING",
    "CONFLICT_ALTERNATIVE",
    "DUPLICATE",
]


class WallEvidenceVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    thickness_support: float = Field(default=0.0, ge=0.0, le=1.0)
    vector_support: float = Field(default=0.0, ge=0.0, le=1.0)
    raster_line_support: float = Field(default=0.0, ge=0.0, le=1.0)
    region_support: float = Field(default=0.0, ge=0.0, le=1.0)
    source_consensus: float = Field(default=0.0, ge=0.0, le=1.0)
    perimeter_containment: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_support: float = Field(default=0.0, ge=0.0, le=1.0)
    axis_support: float = Field(default=0.0, ge=0.0, le=1.0)
    dashed_penalty: float = Field(default=0.0, ge=0.0, le=1.0)


class WallCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    generator: CandidateGenerator
    start: DrawingPoint
    end: DrawingPoint
    angle_deg: float = Field(ge=0.0, lt=180.0)
    length_px: float = Field(gt=0.0)
    thickness_px: float = Field(gt=0.0)
    face_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    evidence: WallEvidenceVector
    prior_score: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class CandidateRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a_id: str = Field(min_length=1)
    b_id: str = Field(min_length=1)
    relation_type: RelationType
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    hard_conflict: bool = False


class CandidateGraphDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)
    hard_conflict_count: int = Field(ge=0)
    junction_count: int = Field(ge=0)
    continuation_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)


class WallCandidateGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    candidates: list[WallCandidate] = Field(default_factory=list)
    relations: list[CandidateRelation] = Field(default_factory=list)
    diagnostics: CandidateGraphDiagnostics
