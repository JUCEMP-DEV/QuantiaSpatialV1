from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.adaptive_reconstruction.contracts import SingleLineWall, SingleLineWallGraph


FilterClass = Literal[
    "WALL",
    "ARCHITECTURAL_ELEMENT",
    "EXCLUDED_GRAPHIC",
    "UNRESOLVED",
]

FilterModule = Literal[
    "CONTEXT_CLASSIFIER",
    "TOPOLOGY_GUARD",
    "GRAPHIC_EVIDENCE_FILTER",
    "ARCHITECTURAL_ELEMENT_FILTER",
    "REPETITIVE_PATTERN_FILTER",
    "ANGULAR_HUB_FILTER",
    "ORPHAN_GRAPHIC_FILTER",
    "DUPLICATE_WALL_CANONICALIZER",
    "WALL_FACE_CANONICALIZER",
    "NEAR_COINCIDENT_CANONICALIZER",
    "RAILING_GUARDRAIL_FILTER",
    "STAIR_HANDRAIL_FILTER",
    "SLAB_EDGE_LEVEL_CHANGE_FILTER",
    "OPENING_CLOSURE_FILTER",
    "ALUMINUM_GLAZING_FILTER",
]

PatternKind = Literal[
    "REPETITIVE_PARALLEL",
    "ORTHOGONAL_GRID",
    "ANGULAR_HUB",
]


class FilterModuleDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: FilterModule
    enabled: bool
    reason: str


class PostFilterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modules: list[FilterModuleDecision] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    def enabled(self, module: FilterModule) -> bool:
        return any(item.module == module and item.enabled for item in self.modules)


class SourceContextSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    states: list[str] = Field(default_factory=list)
    region_types: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    quarantined_source_count: int = Field(default=0, ge=0)
    review_source_count: int = Field(default=0, ge=0)
    protected_source_count: int = Field(default=0, ge=0)
    max_region_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_context_risk: float = Field(default=0.0, ge=0.0, le=1.0)


class PatternEvidence(BaseModel):
    """Patrón geométrico detectado después de la reconstrucción adaptativa.

    Es evidencia de clasificación, no una mutación del WallGraph original.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: PatternKind
    wall_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    output_class_hint: FilterClass
    architectural_family_hint: str | None = None
    reason: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class FilterDecision(BaseModel):
    """Clasifica sin destruir evidencia.

    `included_in_wallgraph` controla únicamente la publicación en el WallGraph
    filtrado. La geometría original siempre permanece referenciada por
    `source_wall_id` y `source_candidate_ids`.
    """

    model_config = ConfigDict(extra="forbid")

    source_wall_id: str
    output_class: FilterClass
    included_in_wallgraph: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    source_candidate_ids: list[str] = Field(default_factory=list)
    source_context: SourceContextSummary = Field(default_factory=SourceContextSummary)
    architectural_family_hint: str | None = None
    pattern_ids: list[str] = Field(default_factory=list)
    applied_modules: list[str] = Field(default_factory=list)
    preserved: bool = True


class EvidenceBucketItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_wall: SingleLineWall
    decision: FilterDecision


class CanonicalWallMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_wall_id: str
    source_wall_ids: list[str] = Field(default_factory=list)
    source_candidate_ids: list[str] = Field(default_factory=list)
    reason: str


class PostFilterDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_wall_count: int = Field(ge=0)
    wall_class_count: int = Field(ge=0)
    architectural_element_count: int = Field(ge=0)
    excluded_graphic_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    canonical_wall_count: int = Field(ge=0)
    collapsed_hypothesis_count: int = Field(ge=0)
    pattern_group_count: int = Field(default=0, ge=0)
    repetitive_pattern_count: int = Field(default=0, ge=0)
    angular_hub_count: int = Field(default=0, ge=0)
    enabled_module_count: int = Field(default=0, ge=0)
    near_coincident_pair_count: int = Field(default=0, ge=0)


class PostReconstructionFilterResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    version: str = "POST_RECONSTRUCTION_FILTER_V4"
    level_view_id: str
    plan: PostFilterPlan
    decisions: list[FilterDecision] = Field(default_factory=list)
    pattern_evidence: list[PatternEvidence] = Field(default_factory=list)
    filtered_wall_graph: SingleLineWallGraph
    architectural_elements: list[EvidenceBucketItem] = Field(default_factory=list)
    excluded_graphics: list[EvidenceBucketItem] = Field(default_factory=list)
    unresolved: list[EvidenceBucketItem] = Field(default_factory=list)
    canonical_mapping: list[CanonicalWallMapping] = Field(default_factory=list)
    diagnostics: PostFilterDiagnostics
    audit: dict[str, Any] = Field(default_factory=dict)
