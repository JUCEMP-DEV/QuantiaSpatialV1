from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.reconstruction_core.drawing_model import DrawingBBox, DrawingPoint


ElementClass = Literal["DOOR", "WINDOW", "NOT_OPENING"]
ElementDecision = Literal["ACCEPTED", "REVIEW", "REJECTED"]


class ArchitecturalRAGExample(BaseModel):
    """Ejemplo propio de Quantia para recuperación arquitectónica.

    No contiene pesos ni salidas de modelos externos. Describe patrones
    geométricos/semánticos interpretables que pueden crecer con evidencia
    validada del proyecto.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: ElementClass
    description: str = Field(min_length=1)
    features: dict[str, float | bool | str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    source: str = "QUANTIA_RULE_SEED_V1"


class ArchitecturalRAGHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str = Field(min_length=1)
    label: ElementClass
    similarity: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=1)
    source: str = Field(min_length=1)


class ElementFeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_region_type: str = Field(min_length=1)
    region_confidence: float = Field(ge=0.0, le=1.0)
    opening_verified: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_support: float = Field(default=0.0, ge=0.0, le=1.0)
    architecture_support: float = Field(default=0.0, ge=0.0, le=1.0)
    host_local_support: float = Field(default=0.0, ge=0.0, le=1.0)
    host_wall_match: float = Field(default=0.0, ge=0.0, le=1.0)
    door_arc_support: float = Field(default=0.0, ge=0.0, le=1.0)
    door_leaf_support: float = Field(default=0.0, ge=0.0, le=1.0)
    window_frame_support: float = Field(default=0.0, ge=0.0, le=1.0)
    wall_through_support: float = Field(default=0.0, ge=0.0, le=1.0)
    wall_gap_support: float = Field(default=0.0, ge=0.0, le=1.0)
    member_count_support: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_class: ElementClass
    tags: list[str] = Field(default_factory=list)

    def as_retrieval_dict(self) -> dict[str, float | bool | str]:
        return {
            "region_confidence": self.region_confidence,
            "opening_verified": self.opening_verified,
            "semantic_support": self.semantic_support,
            "architecture_support": self.architecture_support,
            "host_local_support": self.host_local_support,
            "host_wall_match": self.host_wall_match,
            "door_arc_support": self.door_arc_support,
            "door_leaf_support": self.door_leaf_support,
            "window_frame_support": self.window_frame_support,
            "wall_through_support": self.wall_through_support,
            "wall_gap_support": self.wall_gap_support,
            "member_count_support": self.member_count_support,
            "expected_class": self.expected_class,
        }


class HostWallMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_id: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    distance_px: float | None = Field(default=None, ge=0.0)
    orientation_support: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_support: float = Field(default=0.0, ge=0.0, le=1.0)
    projected_offset_px: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ElementSemanticResolution(BaseModel):
    """Salida opcional de un resolver multimodal/LLM externo.

    V1 no obliga una llamada nueva: la evidencia Gemini ya persistida entra por
    DrawingModel. Este contrato deja preparado el mini-motor para resolvers
    adicionales sin acoplarlos a la geometría.
    """

    model_config = ConfigDict(extra="forbid")

    label: ElementClass
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_code: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpeningElementProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_view_id: str = Field(min_length=1)
    source_region_id: str = Field(min_length=1)
    source_bbox: DrawingBBox
    predicted_class: ElementClass
    decision: ElementDecision
    confidence: float = Field(ge=0.0, le=1.0)
    center_px: DrawingPoint
    orientation_deg: float = Field(ge=0.0, lt=180.0)
    estimated_width_px: float = Field(gt=0.0)
    host_wall: HostWallMatch
    features: ElementFeatureVector
    rag_hits: list[ArchitecturalRAGHit] = Field(default_factory=list)
    semantic_resolution: ElementSemanticResolution | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArchitecturalElementsDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_region_count: int = Field(ge=0)
    opening_region_count: int = Field(ge=0)
    proposal_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    published_opening_count: int = Field(ge=0)
    door_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
    independent_candidate_count: int = Field(default=0, ge=0)
    semantic_candidate_count: int = Field(default=0, ge=0)
    semantic_resolution_count: int = Field(default=0, ge=0)
    source_counts: dict[str, int] = Field(default_factory=dict)


class ArchitecturalElementsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    proposals: list[OpeningElementProposal] = Field(default_factory=list)
    diagnostics: ArchitecturalElementsDiagnostics
    warnings: list[str] = Field(default_factory=list)
