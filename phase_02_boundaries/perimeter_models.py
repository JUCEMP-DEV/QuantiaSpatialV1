from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PerimeterGeometrySource = Literal["VECTOR", "RASTER", "HYBRID"]
PerimeterResolutionState = Literal["RESOLVED", "REVIEW", "UNRESOLVED"]
PerimeterValidationState = Literal["VALID", "REVIEW", "INVALID"]
PerimeterDrawingOrientation = Literal["HORIZONTAL", "VERTICAL", "DIAGONAL"]
PerimeterDrawingSide = Literal["TOP", "BOTTOM", "LEFT", "RIGHT", "AMBIGUOUS"]
PerimeterCardinalSide = Literal["NORTH", "SOUTH", "EAST", "WEST"]
PerimeterMetricStatus = Literal["UNRESOLVED", "PARTIAL", "GROUNDED", "CONFLICT"]
ScaleConsistencyState = Literal[
    "UNRESOLVED",
    "SINGLE_REFERENCE",
    "CONSISTENT",
    "CONFLICT",
]
DimensionAssociationState = Literal[
    "UNRESOLVED",
    "GROUNDED",
    "CONFLICT",
]
WallRepresentation = Literal["EXTERIOR_BOUNDARY_REFERENCE"]
MetricScaleSource = Literal["DIMENSION_RECONCILIATION"]


class PerimeterPointPx(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0)
    y: float = Field(ge=0.0)


class PerimeterBBoxPx(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_extent(self) -> "PerimeterBBoxPx":
        if self.x_max <= self.x_min:
            raise ValueError("x_max debe ser mayor que x_min.")
        if self.y_max <= self.y_min:
            raise ValueError("y_max debe ser mayor que y_min.")
        return self


class PerimeterPolygonPx(BaseModel):
    """Contorno exterior de control en coordenadas locales del LevelView."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    points: list[PerimeterPointPx] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_polygon(self) -> "PerimeterPolygonPx":
        coordinates = [(point.x, point.y) for point in self.points]
        if len(set(coordinates)) < 3:
            raise ValueError("El perímetro requiere al menos 3 puntos distintos.")
        if coordinates[0] == coordinates[-1]:
            raise ValueError(
                "PerimeterPolygonPx utiliza cierre implícito; no repita "
                "el primer punto al final."
            )
        if math.isclose(self.area, 0.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("El perímetro no puede tener área nula.")
        return self

    @property
    def signed_area(self) -> float:
        total = 0.0
        for index, current in enumerate(self.points):
            following = self.points[(index + 1) % len(self.points)]
            total += current.x * following.y
            total -= following.x * current.y
        return total / 2.0

    @property
    def area(self) -> float:
        return abs(self.signed_area)

    @property
    def bbox(self) -> PerimeterBBoxPx:
        xs = [point.x for point in self.points]
        ys = [point.y for point in self.points]
        return PerimeterBBoxPx(
            x_min=min(xs),
            y_min=min(ys),
            x_max=max(xs),
            y_max=max(ys),
        )


class PerimeterEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    kind: str | None = None
    semantic_category: str | None = None
    text: str | None = None
    description: str | None = None
    confidence: float | None = None


class PerimeterCandidate(BaseModel):
    """Candidato geométrico neutral. Todavía no es muro final."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_view_id: str = Field(min_length=1)
    geometry: PerimeterPolygonPx
    geometry_source: PerimeterGeometrySource
    evidence: list[PerimeterEvidenceRef] = Field(default_factory=list)
    semantic_evidence_ids: list[str] = Field(default_factory=list)
    semantic_localized_evidence_ids: list[str] = Field(default_factory=list)
    interior_ring_count: int = Field(default=0, ge=0)
    confirmed: Literal[False] = False


class PerimeterResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    selected: PerimeterCandidate | None = None
    candidates: list[PerimeterCandidate] = Field(default_factory=list)
    state: PerimeterResolutionState
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution(self) -> "PerimeterResolutionResult":
        candidate_ids: set[str] = set()
        for candidate in self.candidates:
            if candidate.level_view_id != self.level_view_id:
                raise ValueError("Existe un candidato asociado a otro LevelView.")
            if candidate.id in candidate_ids:
                raise ValueError(f"PerimeterCandidate.id duplicado: {candidate.id}.")
            candidate_ids.add(candidate.id)

        if self.selected is not None:
            if self.selected.level_view_id != self.level_view_id:
                raise ValueError("El perímetro seleccionado pertenece a otro LevelView.")
            if self.selected.id not in candidate_ids:
                raise ValueError("El perímetro seleccionado no existe en candidates.")

        if self.state == "RESOLVED" and self.selected is None:
            raise ValueError("RESOLVED requiere un perímetro seleccionado.")
        if self.state != "RESOLVED" and self.selected is not None:
            raise ValueError("Solo RESOLVED puede publicar un perímetro seleccionado.")
        return self


class PerimeterWallRun(BaseModel):
    """
    Tramo único del muro perimetral representado sobre el boundary exterior.

    No afirma centerline ni espesor. Esa información solo podrá publicarse
    cuando exista evidencia suficiente para reconstruirla.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_view_id: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    start_px: PerimeterPointPx
    end_px: PerimeterPointPx
    drawing_orientation: PerimeterDrawingOrientation
    drawing_side: PerimeterDrawingSide
    cardinal_side: PerimeterCardinalSide | None = None
    representation: WallRepresentation = "EXTERIOR_BOUNDARY_REFERENCE"
    length_px: float = Field(gt=0.0)
    length_m: float | None = Field(default=None, gt=0.0)
    metric_status: PerimeterMetricStatus = "UNRESOLVED"
    geometry_source: PerimeterGeometrySource
    evidence_ids: list[str] = Field(default_factory=list)
    dimensional_evidence_ids: list[str] = Field(default_factory=list)
    confirmed: Literal[False] = False


class PerimeterDrawingSideSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drawing_side: PerimeterDrawingSide
    wall_run_count: int = Field(ge=0)
    length_px: float = Field(ge=0.0)
    length_m: float | None = Field(default=None, ge=0.0)


class PerimeterDimensionReference(BaseModel):
    """Relación verificable entre una cota observada y una geometría px."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    visible_text: str | None = None
    value_m: float = Field(gt=0.0)
    orientation: Literal["HORIZONTAL", "VERTICAL"]
    span_type: str | None = None
    reference_start: str | None = None
    reference_end: str | None = None
    geometry_target: Literal["BOUNDING_SPAN_X", "BOUNDING_SPAN_Y"]
    geometry_length_px: float = Field(gt=0.0)
    meters_per_px: float = Field(gt=0.0)
    state: DimensionAssociationState
    grounded_wall_run_ids: list[str] = Field(default_factory=list)
    corroborating_evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PerimeterMetricScale(BaseModel):
    """Escala px->m validada por varias asociaciones dimensionales."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    meters_per_px: float = Field(gt=0.0)
    source: MetricScaleSource = "DIMENSION_RECONCILIATION"
    evidence_ids: list[str] = Field(min_length=2)
    dimension_reference_ids: list[str] = Field(min_length=2)
    description: str | None = None


class PerimeterDimensionGroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    status: PerimeterMetricStatus
    scale_consistency: ScaleConsistencyState
    dimension_references: list[PerimeterDimensionReference] = Field(default_factory=list)
    metric_scale: PerimeterMetricScale | None = None
    declared_scale_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_dimension_evidence_ids: list[str] = Field(default_factory=list)
    conflict_evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PerimeterWallLayer(BaseModel):
    """Producto arquitectónico principal de Fase 02 por LevelView."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_view_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    geometry_source: PerimeterGeometrySource
    polygon: PerimeterPolygonPx
    wall_runs: list[PerimeterWallRun] = Field(min_length=3)
    drawing_side_summaries: list[PerimeterDrawingSideSummary] = Field(default_factory=list)
    total_length_px: float = Field(gt=0.0)
    total_length_m: float | None = Field(default=None, gt=0.0)
    metric_status: PerimeterMetricStatus = "UNRESOLVED"
    metric_scale: PerimeterMetricScale | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confirmed: Literal[False] = False


PerimeterEvidenceSupportState = Literal["SUPPORTED", "PARTIAL", "UNRESOLVED"]
PerimeterTextRole = Literal[
    "AXIS_TOKEN_CANDIDATE",
    "NUMERIC_TOKEN_CANDIDATE",
    "DIMENSION_TEXT_CANDIDATE",
    "OTHER_TEXT",
]


class PerimeterSourceEvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    evidence_count: int = Field(ge=0)
    perimeter_support_count: int = Field(ge=0)
    parameterized_support_count: int = Field(ge=0)
    raw_fallback_support_count: int = Field(ge=0)
    kind_counts: dict[str, int] = Field(default_factory=dict)


class PerimeterTextEvidenceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    role: PerimeterTextRole
    parsed_values: list[float] = Field(default_factory=list)
    explicit_unit: str | None = None
    bbox_px: PerimeterBBoxPx | None = None
    parameterized: bool = False


class PerimeterMotorSegmentSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence_index: int = Field(ge=0)
    start_px: PerimeterPointPx
    end_px: PerimeterPointPx
    orientation: PerimeterDrawingOrientation
    length_px: float = Field(gt=0.0)
    evidence_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    parameterized_evidence_ids: list[str] = Field(default_factory=list)
    raw_fallback_evidence_ids: list[str] = Field(default_factory=list)


class PerimeterCandidateEvidenceReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    state: PerimeterEvidenceSupportState
    independent_geometry_sources: list[str] = Field(default_factory=list)
    segment_support: list[PerimeterMotorSegmentSupport] = Field(default_factory=list)
    source_summaries: list[PerimeterSourceEvidenceSummary] = Field(default_factory=list)
    localized_text_evidence: list[PerimeterTextEvidenceObservation] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    parameterized_evidence_ids: list[str] = Field(default_factory=list)
    raw_fallback_evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PerimeterEvidenceReconciliationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    non_gemini_evidence_count: int = Field(ge=0)
    gemini_evidence_count: int = Field(ge=0)
    candidates: list[PerimeterCandidateEvidenceReconciliation] = Field(default_factory=list)
    unassigned_text_evidence: list[PerimeterTextEvidenceObservation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


PerimeterComparisonState = Literal[
    "MATCH",
    "PARTIAL",
    "CONFLICT",
    "UNRESOLVED",
    "NOT_AVAILABLE",
]


class PerimeterSemanticDimension(BaseModel):
    """Cota general semántica usada únicamente como baseline de comparación."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    orientation: Literal["HORIZONTAL", "VERTICAL"]
    value_m: float = Field(gt=0.0)
    visible_text: str | None = None
    span_type: str | None = None
    reference_start: str | None = None
    reference_end: str | None = None


class PerimeterCandidateSemanticComparison(BaseModel):
    """
    Comparación de un candidato generado por el motor contra las cotas
    GENERALES observadas por Gemini legacy.

    No selecciona el candidato ni corrige su geometría.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    motor_span_x_px: float = Field(gt=0.0)
    motor_span_y_px: float = Field(gt=0.0)
    horizontal_value_m: float | None = Field(default=None, gt=0.0)
    vertical_value_m: float | None = Field(default=None, gt=0.0)
    horizontal_meters_per_px: float | None = Field(default=None, gt=0.0)
    vertical_meters_per_px: float | None = Field(default=None, gt=0.0)
    relative_scale_difference: float | None = Field(default=None, ge=0.0)
    motor_horizontal_segment_lengths_px: list[float] = Field(default_factory=list)
    motor_vertical_segment_lengths_px: list[float] = Field(default_factory=list)
    gemini_horizontal_tramos_m: list[float] = Field(default_factory=list)
    gemini_vertical_tramos_m: list[float] = Field(default_factory=list)
    state: PerimeterComparisonState
    notes: list[str] = Field(default_factory=list)


class PerimeterSemanticComparisonResult(BaseModel):
    """
    Auditoría F02 ↔ Gemini.

    Gemini es baseline semántico para fine tuning, no autoridad geométrica.
    No existe umbral arquitectónico implícito: si dos asociaciones no son
    numéricamente iguales, se reporta la diferencia y no se fuerza aceptación.
    """

    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    state: PerimeterComparisonState
    gemini_available: bool
    semantic_dimensions: list[PerimeterSemanticDimension] = Field(default_factory=list)
    semantic_chain_dimensions: list[PerimeterSemanticDimension] = Field(default_factory=list)
    candidate_comparisons: list[PerimeterCandidateSemanticComparison] = Field(
        default_factory=list
    )
    selected_candidate_id: str | None = None
    selected_candidate_state: PerimeterComparisonState | None = None
    notes: list[str] = Field(default_factory=list)
