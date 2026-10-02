from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .perimeter_models import (
    PerimeterDimensionGroundingResult,
    PerimeterMetricStatus,
    PerimeterPointPx,
    PerimeterSemanticComparisonResult,
    PerimeterWallLayer,
)


EditablePerimeterStatus = Literal[
    "GEOMETRY_ONLY",
    "PARTIAL_METRIC",
    "METRIC_READY",
    "METRIC_CONFLICT",
]


class EditablePerimeterVertex(BaseModel):
    """Vértice editable de la poligonal publicada por F02."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    point_px: PerimeterPointPx
    incoming_wall_id: str | None = None
    outgoing_wall_id: str | None = None
    confirmed: Literal[False] = False


class EditablePerimeterWall(BaseModel):
    """Muro perimetral editable y medible entregado por F02."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    start_vertex_id: str = Field(min_length=1)
    end_vertex_id: str = Field(min_length=1)
    drawing_orientation: str = Field(min_length=1)
    drawing_side: str = Field(min_length=1)
    cardinal_side: str | None = None
    length_px: float = Field(gt=0.0)
    length_m: float | None = Field(default=None, gt=0.0)
    metric_status: PerimeterMetricStatus
    evidence_ids: list[str] = Field(default_factory=list)
    dimensional_evidence_ids: list[str] = Field(default_factory=list)
    confirmed: Literal[False] = False


class PerimeterMetricSummary(BaseModel):
    """
    Cierre aritmético de F02 para cuantificación.

    Los valores métricos solo existen si quedaron fundamentados por evidencia
    no-Gemini. `null` nunca se sustituye por 0.
    """

    model_config = ConfigDict(extra="forbid")

    status: PerimeterMetricStatus
    total_perimeter_ml: float | None = Field(default=None, gt=0.0)
    horizontal_total_m: float | None = Field(default=None, ge=0.0)
    vertical_total_m: float | None = Field(default=None, ge=0.0)
    diagonal_total_m: float | None = Field(default=None, ge=0.0)
    grounded_wall_count: int = Field(ge=0)
    unresolved_wall_ids: list[str] = Field(default_factory=list)
    conflict_wall_ids: list[str] = Field(default_factory=list)
    dimension_reference_ids: list[str] = Field(default_factory=list)
    dimension_evidence_ids: list[str] = Field(default_factory=list)
    metric_scale_m_per_px: float | None = Field(default=None, gt=0.0)


class PerimeterComparisonBaseline(BaseModel):
    """Resumen de Gemini para auditoría; nunca alimenta cálculos del motor."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    state: str = Field(min_length=1)
    horizontal_general_m: float | None = Field(default=None, gt=0.0)
    vertical_general_m: float | None = Field(default=None, gt=0.0)
    selected_candidate_state: str | None = None
    notes: list[str] = Field(default_factory=list)


class EditablePerimeterModel(BaseModel):
    """
    Contrato F02 consumible por F03 y por el editor de 04.

    - `vertices` + `walls` = poligonal editable.
    - `metric_summary` = datos aritméticos para cantidades.
    - `comparison_baseline` = Gemini separado y solo comparativo.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_view_id: str = Field(min_length=1)
    source_wall_layer_id: str = Field(min_length=1)
    source_candidate_id: str = Field(min_length=1)
    geometry_revision: int = Field(default=0, ge=0)
    edited_by_user: bool = False
    status: EditablePerimeterStatus
    vertices: list[EditablePerimeterVertex] = Field(min_length=3)
    walls: list[EditablePerimeterWall] = Field(min_length=3)
    metric_summary: PerimeterMetricSummary
    comparison_baseline: PerimeterComparisonBaseline
    evidence_ids: list[str] = Field(default_factory=list)
    confirmed: Literal[False] = False

    @model_validator(mode="after")
    def validate_closed_chain(self) -> "EditablePerimeterModel":
        vertex_ids = [item.id for item in self.vertices]
        if len(vertex_ids) != len(set(vertex_ids)):
            raise ValueError("EditablePerimeterModel contiene vertex ids duplicados.")

        wall_ids = [item.id for item in self.walls]
        if len(wall_ids) != len(set(wall_ids)):
            raise ValueError("EditablePerimeterModel contiene wall ids duplicados.")

        valid_vertices = set(vertex_ids)
        for wall in self.walls:
            if wall.start_vertex_id not in valid_vertices:
                raise ValueError(f"start_vertex_id inexistente: {wall.start_vertex_id}")
            if wall.end_vertex_id not in valid_vertices:
                raise ValueError(f"end_vertex_id inexistente: {wall.end_vertex_id}")

        return self


class PerimeterDeliveryBuilder:
    """Empaqueta la verdad ya cerrada de F02 sin reinterpretar geometría."""

    def build(
        self,
        *,
        wall_layer: PerimeterWallLayer,
        grounding: PerimeterDimensionGroundingResult,
        semantic_comparison: PerimeterSemanticComparisonResult,
    ) -> EditablePerimeterModel:
        if grounding.level_view_id != wall_layer.level_view_id:
            raise ValueError("Grounding pertenece a otro LevelView.")
        if semantic_comparison.level_view_id != wall_layer.level_view_id:
            raise ValueError("SemanticComparison pertenece a otro LevelView.")

        runs = sorted(wall_layer.wall_runs, key=lambda item: item.sequence_index)
        vertices: list[EditablePerimeterVertex] = []
        walls: list[EditablePerimeterWall] = []

        for index, run in enumerate(runs):
            vertex_id = f"{wall_layer.id}__V{index + 1:03d}"
            incoming = runs[index - 1].id if runs else None
            outgoing = run.id
            vertices.append(
                EditablePerimeterVertex(
                    id=vertex_id,
                    sequence_index=index,
                    point_px=run.start_px.model_copy(deep=True),
                    incoming_wall_id=incoming,
                    outgoing_wall_id=outgoing,
                )
            )

        vertex_ids = [item.id for item in vertices]
        for index, run in enumerate(runs):
            walls.append(
                EditablePerimeterWall(
                    id=run.id,
                    sequence_index=run.sequence_index,
                    start_vertex_id=vertex_ids[index],
                    end_vertex_id=vertex_ids[(index + 1) % len(vertex_ids)],
                    drawing_orientation=run.drawing_orientation,
                    drawing_side=run.drawing_side,
                    cardinal_side=run.cardinal_side,
                    length_px=run.length_px,
                    length_m=run.length_m,
                    metric_status=run.metric_status,
                    evidence_ids=list(run.evidence_ids),
                    dimensional_evidence_ids=list(run.dimensional_evidence_ids),
                )
            )

        metric_summary = self._metric_summary(
            wall_layer=wall_layer,
            grounding=grounding,
        )
        comparison_baseline = self._comparison_baseline(semantic_comparison)

        if wall_layer.metric_status == "GROUNDED" and metric_summary.total_perimeter_ml is not None:
            status: EditablePerimeterStatus = "METRIC_READY"
        elif wall_layer.metric_status == "PARTIAL":
            status = "PARTIAL_METRIC"
        elif wall_layer.metric_status == "CONFLICT":
            status = "METRIC_CONFLICT"
        else:
            status = "GEOMETRY_ONLY"

        return EditablePerimeterModel(
            id=f"{wall_layer.id}__EDITABLE",
            level_view_id=wall_layer.level_view_id,
            source_wall_layer_id=wall_layer.id,
            source_candidate_id=wall_layer.candidate_id,
            status=status,
            vertices=vertices,
            walls=walls,
            metric_summary=metric_summary,
            comparison_baseline=comparison_baseline,
            evidence_ids=list(wall_layer.evidence_ids),
        )

    @staticmethod
    def _metric_summary(
        *,
        wall_layer: PerimeterWallLayer,
        grounding: PerimeterDimensionGroundingResult,
    ) -> PerimeterMetricSummary:
        horizontal_values: list[float] = []
        vertical_values: list[float] = []
        diagonal_values: list[float] = []
        unresolved: list[str] = []
        conflicts: list[str] = []
        grounded_count = 0

        for run in wall_layer.wall_runs:
            if run.length_m is None:
                if run.metric_status == "CONFLICT":
                    conflicts.append(run.id)
                else:
                    unresolved.append(run.id)
                continue

            grounded_count += 1
            if run.drawing_orientation == "HORIZONTAL":
                horizontal_values.append(run.length_m)
            elif run.drawing_orientation == "VERTICAL":
                vertical_values.append(run.length_m)
            else:
                diagonal_values.append(run.length_m)

        def sum_or_none(values: list[float]) -> float | None:
            return sum(values) if values else None

        return PerimeterMetricSummary(
            status=wall_layer.metric_status,
            total_perimeter_ml=wall_layer.total_length_m,
            horizontal_total_m=sum_or_none(horizontal_values),
            vertical_total_m=sum_or_none(vertical_values),
            diagonal_total_m=sum_or_none(diagonal_values),
            grounded_wall_count=grounded_count,
            unresolved_wall_ids=unresolved,
            conflict_wall_ids=conflicts,
            dimension_reference_ids=[item.id for item in grounding.dimension_references],
            dimension_evidence_ids=sorted(
                {
                    item.evidence_id
                    for item in grounding.dimension_references
                }
            ),
            metric_scale_m_per_px=(
                grounding.metric_scale.meters_per_px
                if grounding.metric_scale is not None
                else None
            ),
        )

    @staticmethod
    def _comparison_baseline(
        comparison: PerimeterSemanticComparisonResult,
    ) -> PerimeterComparisonBaseline:
        horizontal = None
        vertical = None
        for item in comparison.semantic_dimensions:
            if item.orientation == "HORIZONTAL" and horizontal is None:
                horizontal = item.value_m
            elif item.orientation == "VERTICAL" and vertical is None:
                vertical = item.value_m

        return PerimeterComparisonBaseline(
            available=comparison.gemini_available,
            state=comparison.state,
            horizontal_general_m=horizontal,
            vertical_general_m=vertical,
            selected_candidate_state=comparison.selected_candidate_state,
            notes=list(comparison.notes),
        )
