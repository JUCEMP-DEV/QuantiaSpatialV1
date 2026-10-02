from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .candidate_models import WallCandidate
from .drawing_model import DrawingBBox


ContextRegionType = Literal[
    # Clasificación semántica/funcional V6.
    "STAIR_FLIGHT_REGION",
    "FLOOR_FINISH_GRID_REGION",
    "HATCH_FILL_REGION",
    "AXIS_GRID_REGION",
    "DIMENSION_REGION",
    "FURNITURE_MODULE_REGION",
    "SYMBOL_FIXTURE_REGION",
    "DOOR_REGION",
    "WINDOW_REGION",
    "OPENING_REGION",
    "COLUMN_REGION",
    "TEXT_SYMBOL_REGION",
    "WALL_PROTECTED_REGION",
    "UNKNOWN_REPETITIVE_REGION",
    # Tipos V5 conservados para compatibilidad/replay y para diagnóstico interno.
    "REPETITIVE_PARALLEL_REGION",
    "REPETITIVE_GRID_REGION",
]

CandidateContextState = Literal["ACTIVE", "PROTECTED", "QUARANTINE", "REVIEW"]


class RepetitiveAxisProfile(BaseModel):
    """Firma geométrica de una familia repetitiva observada en DrawingModel.

    No representa un muro ni una exclusión. Solo describe una modulación de
    primitivas gráficas antes de construir el CandidateGraph.
    """

    model_config = ConfigDict(extra="forbid")

    angle_deg: float = Field(ge=0.0, lt=180.0)
    spacing_px: float = Field(gt=0.0)
    spacing_cv: float = Field(ge=0.0)
    member_count: int = Field(ge=3)
    median_length_px: float = Field(gt=0.0)
    common_span_ratio: float = Field(ge=0.0, le=1.0)
    length_similarity: float = Field(ge=0.0, le=1.0)
    track_coordinates: list[float] = Field(min_length=3)
    member_line_ids: list[str] = Field(min_length=3)


class ContextRegion(BaseModel):
    """Región contextual V6 usada para clasificar evidencia antes del solver.

    `profiles` puede estar vacío para clases que no nacen de periodicidad lineal,
    por ejemplo una región protegida por el perímetro F02 o una región de celdas
    rectangulares detectada directamente.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    region_type: ContextRegionType
    bbox: DrawingBBox
    confidence: float = Field(ge=0.0, le=1.0)
    quarantine_enabled: bool = False
    profiles: list[RepetitiveAxisProfile] = Field(default_factory=list)
    member_line_ids: list[str] = Field(default_factory=list)
    semantic_support: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class CandidateContextDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    state: CandidateContextState
    reason: str = Field(min_length=1)
    region_id: str | None = None
    region_type: ContextRegionType | None = None
    region_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    spatial_containment: float = Field(default=0.0, ge=0.0, le=1.0)
    face_pattern_support: float = Field(default=0.0, ge=0.0, le=1.0)
    orientation_match: float = Field(default=0.0, ge=0.0, le=1.0)
    spacing_match: float = Field(default=0.0, ge=0.0, le=1.0)
    context_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class CandidateContextGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_candidates: list[WallCandidate] = Field(default_factory=list)
    quarantined_candidates: list[WallCandidate] = Field(default_factory=list)
    review_candidates: list[WallCandidate] = Field(default_factory=list)
    decisions: list[CandidateContextDecision] = Field(default_factory=list)
    regions: list[ContextRegion] = Field(default_factory=list)

    @property
    def active_candidate_ids(self) -> set[str]:
        return {item.id for item in self.active_candidates}

    @property
    def quarantined_candidate_ids(self) -> set[str]:
        return {item.id for item in self.quarantined_candidates}

    @property
    def review_candidate_ids(self) -> set[str]:
        return {item.id for item in self.review_candidates}

    @property
    def solver_candidates(self) -> list[WallCandidate]:
        """Entrega ACTIVE/PROTECTED + REVIEW con contexto adjunto.

        QUARANTINE continúa completamente encapsulado fuera del CandidateGraph.
        PROTECTED no recibe riesgo negativo; REVIEW conserva `context_risk` para
        el scoring estandarizado de Solver Evidence V1.
        """
        decisions = {item.candidate_id: item for item in self.decisions}
        output: dict[str, WallCandidate] = {}
        for candidate in [*self.active_candidates, *self.review_candidates]:
            decision = decisions.get(candidate.id)
            metadata = dict(candidate.metadata)
            if decision is not None:
                metadata.update(
                    {
                        "context_state": decision.state,
                        "context_reason": decision.reason,
                        "context_region_id": decision.region_id,
                        "context_region_type": decision.region_type,
                        "context_region_confidence": decision.region_confidence,
                        "context_spatial_containment": decision.spatial_containment,
                        "context_face_pattern_support": decision.face_pattern_support,
                        "context_orientation_match": decision.orientation_match,
                        "context_spacing_match": decision.spacing_match,
                        "context_risk": decision.context_risk,
                        "context_decision_metadata": dict(decision.metadata),
                        "structural_lineage": dict(decision.metadata.get("structural_lineage") or {}),
                    }
                )
            output[candidate.id] = candidate.model_copy(update={"metadata": metadata})
        return list(output.values())

    @property
    def solver_candidate_ids(self) -> set[str]:
        return {item.id for item in self.solver_candidates}
