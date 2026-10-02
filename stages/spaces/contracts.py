from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SpaceState = Literal["VALID", "REVIEW", "UNRESOLVED"]
SemanticAssignmentState = Literal["MATCHED", "UNMATCHED"]
FaceSemanticState = Literal[
    "UNASSIGNED_FACE",
    "SEMANTICALLY_SEPARATED",
    "UNDER_SEGMENTED_MULTIPLE_SPACES",
]


class SpaceDerivedWall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_wall_ids: list[str] = Field(default_factory=list)
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    thickness_px: float = Field(gt=0.0)
    role: str
    confidence: float = Field(ge=0.0, le=1.0)
    orientation: Literal["H", "V", "O"]
    axis_px: float | None = None


class SpaceLogicalClosure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    length_px: float = Field(gt=0.0)
    length_m: float = Field(gt=0.0)
    wall_ids: list[str] = Field(default_factory=list)
    source_gap_id: str
    physical_wall_present: Literal[False] = False
    logical_continuity_only: Literal[True] = True


class SpaceGeometryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    level_view_id: str
    vertices_px: list[tuple[float, float]] = Field(default_factory=list)
    vertices_m: list[tuple[float, float]] = Field(default_factory=list)
    area_m2: float = Field(gt=0.0)
    perimeter_m: float = Field(gt=0.0)
    minimum_width_m: float = Field(gt=0.0)
    wall_ids: list[str] = Field(default_factory=list)
    logical_closure_ids: list[str] = Field(default_factory=list)
    state: Literal["REVIEW"] = "REVIEW"


class SpaceClosureDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_wall_count: int = Field(ge=0)
    derived_wall_count: int = Field(ge=0)
    axis_cluster_count: int = Field(ge=0)
    residual_parallel_merge_count: int = Field(ge=0)
    logical_closure_count: int = Field(ge=0)
    raw_polygon_count: int = Field(ge=0)
    meaningful_space_count: int = Field(ge=0)
    rejected_face_count: int = Field(ge=0)
    dangle_length_px: float = Field(ge=0.0)
    cut_length_px: float = Field(ge=0.0)
    invalid_ring_length_px: float = Field(ge=0.0)


class SpaceClosureResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "SPACE_CLOSURE_ENGINE_V1"
    level_view_id: str
    px_per_m: float = Field(gt=0.0)
    derived_walls: list[SpaceDerivedWall] = Field(default_factory=list)
    logical_closures: list[SpaceLogicalClosure] = Field(default_factory=list)
    spaces: list[SpaceGeometryCandidate] = Field(default_factory=list)
    wall_owners: dict[str, list[str]] = Field(default_factory=dict)
    wall_remap: dict[str, str] = Field(default_factory=dict)
    diagnostics: SpaceClosureDiagnostics


class ReferenceFootprintCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str | None = None
    vertices_px: list[tuple[float, float]] = Field(default_factory=list)
    area_m2: float = Field(gt=0.0)
    perimeter_m: float = Field(gt=0.0)
    classification: Literal["REFERENCE_FOOTPRINT_CANDIDATE_NOT_CONFIRMED_TERRAIN_AREA"] = (
        "REFERENCE_FOOTPRINT_CANDIDATE_NOT_CONFIRMED_TERRAIN_AREA"
    )


class SemanticSpaceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    semantic_id: str | None = None
    name: str | None = None
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_px: tuple[float, float, float, float]
    has_explicit_dimensions: bool = False


class SemanticSpaceAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    semantic_id: str | None = None
    name: str | None = None
    assigned_face_index: int | None = None
    center_match_count: int = Field(ge=0)
    bbox_overlap_ratio: float = Field(ge=0.0, le=1.0)
    status: SemanticAssignmentState


class SpaceFaceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    face_index: int = Field(ge=0)
    area_m2: float = Field(gt=0.0)
    semantic_space_ids: list[str] = Field(default_factory=list)
    state: FaceSemanticState


class Call2RemovalAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_id: str
    role_before_call2: str
    confidence_before_call2: float = Field(ge=0.0, le=1.0)
    f03_seed_protected: bool
    length_m: float = Field(gt=0.0)
    call2_reason: str
    still_in_final_graph: bool
    requires_area_topology_review: bool


class SpaceConstraintMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area_conservation_ratio: float = Field(ge=0.0, le=1.0)
    semantic_separation_ratio: float = Field(ge=0.0, le=1.0)
    semantic_assignment_ratio: float = Field(ge=0.0, le=1.0)
    dangle_quality: float = Field(ge=0.0, le=1.0)
    closure_score: float = Field(ge=0.0, le=1.0)


class SpaceConstraintResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "SPACE_CONSTRAINT_VALIDATOR_V1"
    level_view_id: str
    state: SpaceState
    footprint: ReferenceFootprintCandidate | None = None
    semantic_spaces: list[SemanticSpaceObservation] = Field(default_factory=list)
    assignments: list[SemanticSpaceAssignment] = Field(default_factory=list)
    faces: list[SpaceFaceAudit] = Field(default_factory=list)
    call2_removals: list[Call2RemovalAudit] = Field(default_factory=list)
    metrics: SpaceConstraintMetrics | None = None
    warnings: list[str] = Field(default_factory=list)
