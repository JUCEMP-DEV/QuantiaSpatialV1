from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ModuleName = Literal[
    "F03_STRUCTURAL_SEED",
    "CONTEXT_NEGATIVE_MASK",
    "WALL_MASK_RECOVERY",
    "WALL_CANONICALIZATION",
    "STRICT_CONNECTIVITY",
    "ROOM_TOPOLOGY",
    "MULTIMODAL_CALL2",
]

RouteMode = Literal["SEED_FIRST", "BALANCED", "DENSE_RECOVERY", "SEED_RESCUE"]
WallRole = Literal["PERIMETER", "DIVIDER", "REVIEW"]


class ModuleDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: ModuleName
    enabled: bool
    reason: str


class AdaptiveRoutePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: RouteMode
    modules: list[ModuleDecision]
    recovery_strength: float = Field(ge=0.0, le=1.0)
    require_multimodal_review: bool = False
    reasons: list[str] = Field(default_factory=list)

    def enabled(self, module: ModuleName) -> bool:
        return any(item.module == module and item.enabled for item in self.modules)


class ReconstructionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str
    f03_seed_count: int = Field(ge=0)
    discovered_candidate_count: int = Field(ge=0)
    quarantined_candidate_count: int = Field(ge=0)
    hybrid_candidate_count: int = Field(ge=0)
    selected_wall_count: int = Field(ge=0)
    component_count: int = Field(ge=0)
    junction_count: int = Field(ge=0)
    virtual_bridge_count: int = Field(ge=0)
    interior_space_count: int = Field(ge=0)
    review_wall_count: int = Field(ge=0)
    perimeter_wall_count: int = Field(ge=0)
    divider_wall_count: int = Field(ge=0)
    seed_coverage_ratio: float = Field(ge=0.0, le=1.0)
    quarantine_ratio: float = Field(ge=0.0, le=1.0)
    review_ratio: float = Field(ge=0.0, le=1.0)


class SingleLineWall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    thickness_px: float = Field(gt=0.0)
    role: WallRole
    confidence: float = Field(ge=0.0, le=1.0)
    source_candidate_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    f03_seed_protected: bool = False
    context_state: str = "ACTIVE"
    topology_support: dict[str, Any] = Field(default_factory=dict)


class LogicalGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    gap_px: float = Field(gt=0.0)
    wall_ids: list[str] = Field(default_factory=list)
    protected_by_seed: bool = False
    physical_wall_present: bool = False
    logical_continuity_only: bool = True


class SingleLineWallGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str
    level_name: str
    image_size_px: tuple[int, int]
    px_per_m: float = Field(gt=0.0)
    walls: list[SingleLineWall] = Field(default_factory=list)
    logical_gaps: list[LogicalGap] = Field(default_factory=list)
    interior_space_count: int = Field(ge=0)
    route_plan: AdaptiveRoutePlan
    diagnostics: ReconstructionDiagnostics


DeltaType = Literal[
    "ADD_WALL",
    "REMOVE_WALL",
    "EXTEND_WALL",
    "TRIM_WALL",
    "REPOSITION_WALL",
    "MERGE_WALLS",
    "SPLIT_WALL",
]


class WallDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DeltaType
    wall_id: str | None = None
    wall_ids: list[str] = Field(default_factory=list)
    start_px: tuple[float, float] | None = None
    end_px: tuple[float, float] | None = None
    endpoint: Literal["START", "END"] | None = None
    new_point_px: tuple[float, float] | None = None
    role_hint: WallRole | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class GapDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gap_id: str
    classification: Literal["WALL_CONTINUITY", "PROBABLE_OPENING", "UNCERTAIN", "NOT_A_GAP"]
    host_wall_continuity: bool
    solid_wall_present: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class MultimodalWallReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str
    graph_state: Literal["VALID", "PARTIAL", "REVIEW"]
    deltas: list[WallDelta] = Field(default_factory=list)
    gap_decisions: list[GapDecision] = Field(default_factory=list)
    non_wall_architectural_regions: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_regions: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""




class Call2DeltaValidationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta: WallDelta
    accepted: bool
    reason: str


class Call2GapValidationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gap_decision: GapDecision
    accepted: bool
    reason: str


class Call2ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str
    accepted_review: MultimodalWallReview
    delta_items: list[Call2DeltaValidationItem] = Field(default_factory=list)
    gap_items: list[Call2GapValidationItem] = Field(default_factory=list)
    accepted_architectural_regions: list[dict[str, Any]] = Field(default_factory=list)
    rejected_architectural_regions: list[dict[str, Any]] = Field(default_factory=list)
    accepted_unresolved_regions: list[dict[str, Any]] = Field(default_factory=list)
    rejected_unresolved_regions: list[dict[str, Any]] = Field(default_factory=list)
    missing_gap_ids: list[str] = Field(default_factory=list)
    duplicate_gap_ids: list[str] = Field(default_factory=list)
    gap_coverage_ratio: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def accepted_delta_count(self) -> int:
        return sum(item.accepted for item in self.delta_items)

    @property
    def rejected_delta_count(self) -> int:
        return sum(not item.accepted for item in self.delta_items)


@dataclass(frozen=True)
class PhysicalWallTrack:
    wall_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    length_px: float
    thickness_px: float
    confidence: float
    source_candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_names: tuple[str, ...]
    region_support: float
    thickness_support: float
    vector_support: float
    raster_line_support: float
    source_consensus: float
    axis_support: float
    dashed_penalty: float
    context_state: str = "ACTIVE"


@dataclass(frozen=True)
class VirtualBridge:
    bridge_id: str
    kind: str
    start: tuple[float, float]
    end: tuple[float, float]
    gap_px: float
    wall_ids: tuple[str, ...]
    protected_by_seed: bool


@dataclass(frozen=True)
class SpaceTopology:
    labels: Any
    exterior_labels: frozenset[int]
    interior_labels: frozenset[int]
    areas_px2: dict[int, int]


@dataclass(frozen=True)
class ClassifiedWall:
    wall_id: str
    role: WallRole
    confidence: float
    perimeter_votes: int
    divider_votes: int
    review_votes: int
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]


@dataclass
class AdaptiveReconstructionRuntime:
    graph: SingleLineWallGraph
    wall_tracks: list[PhysicalWallTrack]
    bridges: list[VirtualBridge]
    topology: SpaceTopology
    context_quarantined_count: int
    artifacts: dict[str, Any] = field(default_factory=dict)
