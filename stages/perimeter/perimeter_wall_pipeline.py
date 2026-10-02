from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView

from .boundary_geometry_detector import BoundaryGeometryDetector
from .perimeter_dimension_grounder import PerimeterDimensionGrounder
from .perimeter_delivery import EditablePerimeterModel, PerimeterDeliveryBuilder
from .perimeter_evidence_reconciler import PerimeterEvidenceReconciler
from .perimeter_semantic_comparator import PerimeterSemanticComparator
from .perimeter_models import (
    PerimeterCandidate,
    PerimeterDimensionGroundingResult,
    PerimeterEvidenceReconciliationResult,
    PerimeterResolutionResult,
    PerimeterSemanticComparisonResult,
    PerimeterValidationState,
    PerimeterWallLayer,
)
from .perimeter_wall_builder import PerimeterWallBuilder
from .perimeter_wall_graph import PerimeterWallGraphBuilder, PerimeterWallGraphResult
from .perimeter_wall_resolver import PerimeterWallResolver
from .perimeter_wall_validator import (
    PerimeterWallValidationResult,
    PerimeterWallValidator,
)


class PerimeterWallPipelineDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_evidence_count: int = 0
    vector_line_count: int = 0
    vector_primitive_count: int = 0
    raster_line_count: int = 0
    raster_contour_count: int = 0
    gemini_observation_count: int = 0
    semantic_boundary_wall_count: int = 0
    semantic_dimension_count: int = 0
    parameter_count: int = 0
    reconciled_supported_candidate_count: int = 0
    reconciled_partial_candidate_count: int = 0
    reconciled_unresolved_candidate_count: int = 0
    candidate_count: int = 0
    wall_run_count: int = 0
    grounded_wall_run_count: int = 0
    horizontal_wall_run_count: int = 0
    vertical_wall_run_count: int = 0
    diagonal_wall_run_count: int = 0
    dimension_reference_count: int = 0
    metric_scale_resolved: bool = False
    wall_graph_state: str = "UNRESOLVED"
    wall_graph_edge_count: int = 0
    wall_graph_supported_edge_count: int = 0
    input_kind_counts: dict[str, int] = Field(default_factory=dict)


class PerimeterWallPipelineResult(BaseModel):
    """Salida canónica de Fase 02 por LevelView."""

    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    candidates: list[PerimeterCandidate] = Field(default_factory=list)
    wall_graph: PerimeterWallGraphResult
    evidence_reconciliation: PerimeterEvidenceReconciliationResult
    resolution: PerimeterResolutionResult
    motor_wall_layer: PerimeterWallLayer | None = None
    motor_grounding: PerimeterDimensionGroundingResult | None = None
    wall_layer: PerimeterWallLayer | None = None
    grounding: PerimeterDimensionGroundingResult | None = None
    editable_perimeter: EditablePerimeterModel | None = None
    semantic_comparison: PerimeterSemanticComparisonResult
    validation: PerimeterWallValidationResult
    state: PerimeterValidationState
    diagnostics: PerimeterWallPipelineDiagnostics
    warnings: list[str] = Field(default_factory=list)


class PerimeterWallPipeline:
    """
    Fase 02 — MUROS PERIMETRALES MEDIBLES POR NIVEL.

    Flujo:
        LevelView + RawEvidence 01.5
            -> grafo/ciclo exterior desde contorno estructural + líneas H/V
            -> fallback de candidatos geométricos legacy solo si el grafo no cierra
            -> reconciliación PyMuPDF/OpenCV/OCR (parámetros primero, crudo fallback)
            -> resolución geométrica propia del motor
            -> wall_runs únicos sobre el boundary exterior
            -> grounding cota real <-> geometría px SOLO con evidencia del motor
            -> comparación independiente contra Gemini
            -> validación geométrica + dimensional
            -> PerimeterWallLayer

    No recibe metric_scale desde afuera. F02 debe resolver su propio grounding
    o declarar explícitamente UNRESOLVED/PARTIAL/CONFLICT.
    """

    def __init__(
        self,
        *,
        geometry_detector: BoundaryGeometryDetector | None = None,
        wall_graph_builder: PerimeterWallGraphBuilder | None = None,
        resolver: PerimeterWallResolver | None = None,
        evidence_reconciler: PerimeterEvidenceReconciler | None = None,
        wall_builder: PerimeterWallBuilder | None = None,
        dimension_grounder: PerimeterDimensionGrounder | None = None,
        semantic_comparator: PerimeterSemanticComparator | None = None,
        delivery_builder: PerimeterDeliveryBuilder | None = None,
        validator: PerimeterWallValidator | None = None,
    ) -> None:
        self.geometry_detector = geometry_detector or BoundaryGeometryDetector()
        self.wall_graph_builder = wall_graph_builder or PerimeterWallGraphBuilder()
        self.evidence_reconciler = evidence_reconciler or PerimeterEvidenceReconciler()
        self.resolver = resolver or PerimeterWallResolver()
        self.wall_builder = wall_builder or PerimeterWallBuilder()
        self.dimension_grounder = dimension_grounder or PerimeterDimensionGrounder()
        self.semantic_comparator = semantic_comparator or PerimeterSemanticComparator()
        self.delivery_builder = delivery_builder or PerimeterDeliveryBuilder()
        self.validator = validator or PerimeterWallValidator()

    def run(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> PerimeterWallPipelineResult:
        raw = list(evidence)
        self._validate_input(level_view=level_view, evidence=raw)

        wall_graph = self.wall_graph_builder.build(
            level_view=level_view,
            evidence=raw,
        )

        if wall_graph.state == "RESOLVED" and wall_graph.candidate is not None:
            # El grafo ya cerró la verdad geométrica de F02. No se vuelve a
            # multiplicar ruido ejecutando el detector de polígonos legacy.
            candidates = [wall_graph.candidate.model_copy(deep=True)]
        else:
            candidates = self.geometry_detector.detect(
                level_view=level_view,
                evidence=raw,
            )
            if wall_graph.candidate is not None and all(
                item.id != wall_graph.candidate.id for item in candidates
            ):
                candidates = [wall_graph.candidate.model_copy(deep=True), *candidates]

        evidence_reconciliation = self.evidence_reconciler.reconcile(
            level_view=level_view,
            candidates=candidates,
            evidence=raw,
        )

        resolution = self.resolver.resolve(
            level_view=level_view,
            candidates=candidates,
            evidence=raw,
            reconciliation=evidence_reconciliation,
            wall_graph=wall_graph,
        )

        semantic_comparison = self.semantic_comparator.compare(
            level_view=level_view,
            candidates=resolution.candidates,
            evidence=raw,
            selected_candidate_id=(
                resolution.selected.id if resolution.selected is not None else None
            ),
            reconciliation=evidence_reconciliation,
        )

        motor_wall_layer: PerimeterWallLayer | None = None
        motor_grounding: PerimeterDimensionGroundingResult | None = None
        wall_layer: PerimeterWallLayer | None = None
        grounding: PerimeterDimensionGroundingResult | None = None
        editable_perimeter: EditablePerimeterModel | None = None

        if resolution.state == "RESOLVED" and resolution.selected is not None:
            wall_layer = self.wall_builder.build(
                level_view=level_view,
                candidate=resolution.selected,
                evidence=raw,
            )
            wall_layer, grounding = self.dimension_grounder.ground(
                level_view=level_view,
                wall_layer=wall_layer,
                evidence=raw,
            )
            motor_wall_layer = wall_layer.model_copy(deep=True)
            motor_grounding = grounding.model_copy(deep=True)
            if semantic_comparison.selected_candidate_state == "CONFLICT":
                self._invalidate_metric_on_semantic_conflict(
                    wall_layer=wall_layer,
                    grounding=grounding,
                    comparison=semantic_comparison,
                )

            editable_perimeter = self.delivery_builder.build(
                wall_layer=wall_layer,
                grounding=grounding,
                semantic_comparison=semantic_comparison,
            )

        validation = self.validator.validate(
            level_view=level_view,
            resolution=resolution,
            wall_layer=wall_layer,
            grounding=grounding,
            wall_graph=wall_graph,
        )

        diagnostics = self._build_diagnostics(
            evidence=raw,
            candidates=candidates,
            reconciliation=evidence_reconciliation,
            wall_layer=wall_layer,
            grounding=grounding,
            wall_graph=wall_graph,
        )

        warnings = list(
            dict.fromkeys(
                [
                    *evidence_reconciliation.notes,
                    *resolution.notes,
                    *(grounding.notes if grounding is not None else []),
                    *semantic_comparison.notes,
                    *validation.warnings,
                    *validation.errors,
                ]
            )
        )

        result = PerimeterWallPipelineResult(
            level_view_id=level_view.id,
            candidates=[candidate.model_copy(deep=True) for candidate in candidates],
            wall_graph=wall_graph,
            evidence_reconciliation=evidence_reconciliation,
            resolution=resolution,
            motor_wall_layer=motor_wall_layer,
            motor_grounding=motor_grounding,
            wall_layer=wall_layer,
            grounding=grounding,
            editable_perimeter=editable_perimeter,
            semantic_comparison=semantic_comparison,
            validation=validation,
            state=validation.state,
            diagnostics=diagnostics,
            warnings=warnings,
        )

        self._validate_output(level_view=level_view, result=result)
        return result

    @staticmethod
    def _invalidate_metric_on_semantic_conflict(
        *,
        wall_layer: PerimeterWallLayer,
        grounding: PerimeterDimensionGroundingResult,
        comparison: PerimeterSemanticComparisonResult,
    ) -> None:
        """
        Invalida únicamente la publicación métrica final cuando la auditoría
        F02↔Gemini reporta CONFLICT.

        Se conserva:
        - la geometría reconstruida por el motor en px dentro de wall_layer;
        - el PerimeterSemanticComparisonResult completo con baseline Gemini,
          spans px del motor, relaciones m/px y diferencia relativa.

        No se usa Gemini para corregir ni seleccionar geometría.
        """
        conflict_ids = {
            item.evidence_id for item in comparison.semantic_dimensions
        }
        for run in wall_layer.wall_runs:
            run.length_m = None
            run.metric_status = "CONFLICT"
            run.dimensional_evidence_ids = sorted(
                set(run.dimensional_evidence_ids) | conflict_ids
            )
        for summary in wall_layer.drawing_side_summaries:
            summary.length_m = None
        wall_layer.total_length_m = None
        wall_layer.metric_status = "CONFLICT"
        wall_layer.metric_scale = None

        grounding.status = "CONFLICT"
        grounding.scale_consistency = "CONFLICT"
        grounding.metric_scale = None
        grounding.conflict_evidence_ids = sorted(
            set(grounding.conflict_evidence_ids) | conflict_ids
        )
        note = (
            "Comparación F02↔Gemini en conflicto; las longitudes métricas finales se "
            "mantienen null. Se conserva la geometría del motor en px y la comparación "
            "Gemini completa para fine tuning."
        )
        if note not in grounding.notes:
            grounding.notes.append(note)

    @staticmethod
    def _validate_input(
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> None:
        if not level_view.id:
            raise ValueError("LevelView.id vacío.")
        if level_view.raster_width_px <= 0 or level_view.raster_height_px <= 0:
            raise ValueError("LevelView tiene dimensiones raster inválidas.")

        ids: set[str] = set()
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(f"RawEvidence {item.id} pertenece a otro LevelView.")
            if item.id in ids:
                raise ValueError(f"RawEvidence.id duplicado: {item.id}.")
            ids.add(item.id)
            if item.confirmed:
                raise ValueError(f"RawEvidence {item.id} llegó confirmed=True.")

    @staticmethod
    def _build_diagnostics(
        *,
        evidence: Sequence[RawEvidence],
        candidates: Sequence[PerimeterCandidate],
        reconciliation: PerimeterEvidenceReconciliationResult,
        wall_layer: PerimeterWallLayer | None,
        grounding: PerimeterDimensionGroundingResult | None,
        wall_graph: PerimeterWallGraphResult,
    ) -> PerimeterWallPipelineDiagnostics:
        kind_counter = Counter(str(item.kind) for item in evidence)
        runs = wall_layer.wall_runs if wall_layer is not None else []

        semantic_boundary_wall_count = sum(
            1
            for item in evidence
            if item.kind == "GEMINI_OBSERVATION"
            and str(item.metadata.get("semantic_category", "")).upper()
            in {"BOUNDARY_OBSERVATION", "WALL_OBSERVATION"}
        )
        semantic_dimension_count = sum(
            1
            for item in evidence
            if item.kind == "GEMINI_OBSERVATION"
            and str(item.metadata.get("semantic_category", "")).upper() == "DIMENSION"
        )

        return PerimeterWallPipelineDiagnostics(
            input_evidence_count=len(evidence),
            vector_line_count=kind_counter.get("VECTOR_LINE", 0),
            vector_primitive_count=kind_counter.get("VECTOR_PRIMITIVE", 0),
            raster_line_count=kind_counter.get("RASTER_LINE", 0),
            raster_contour_count=kind_counter.get("RASTER_CONTOUR", 0),
            gemini_observation_count=kind_counter.get("GEMINI_OBSERVATION", 0),
            semantic_boundary_wall_count=semantic_boundary_wall_count,
            semantic_dimension_count=semantic_dimension_count,
            parameter_count=sum(len(getattr(item, "parameters", [])) for item in evidence),
            reconciled_supported_candidate_count=sum(
                1 for item in reconciliation.candidates if item.state == "SUPPORTED"
            ),
            reconciled_partial_candidate_count=sum(
                1 for item in reconciliation.candidates if item.state == "PARTIAL"
            ),
            reconciled_unresolved_candidate_count=sum(
                1 for item in reconciliation.candidates if item.state == "UNRESOLVED"
            ),
            candidate_count=len(candidates),
            wall_run_count=len(runs),
            grounded_wall_run_count=sum(1 for run in runs if run.length_m is not None),
            horizontal_wall_run_count=sum(
                1 for run in runs if run.drawing_orientation == "HORIZONTAL"
            ),
            vertical_wall_run_count=sum(
                1 for run in runs if run.drawing_orientation == "VERTICAL"
            ),
            diagonal_wall_run_count=sum(
                1 for run in runs if run.drawing_orientation == "DIAGONAL"
            ),
            dimension_reference_count=(
                len(grounding.dimension_references) if grounding is not None else 0
            ),
            metric_scale_resolved=(
                grounding is not None and grounding.metric_scale is not None
            ),
            input_kind_counts=dict(sorted(kind_counter.items())),
            wall_graph_state=wall_graph.state,
            wall_graph_edge_count=len(wall_graph.edges),
            wall_graph_supported_edge_count=sum(1 for edge in wall_graph.edges if edge.evidence_ids),
        )

    @staticmethod
    def _validate_output(
        *,
        level_view: LevelView,
        result: PerimeterWallPipelineResult,
    ) -> None:
        if result.level_view_id != level_view.id:
            raise RuntimeError("PerimeterWallPipelineResult pertenece a otro LevelView.")
        if result.wall_graph.level_view_id != level_view.id:
            raise RuntimeError("PerimeterWallGraphResult pertenece a otro LevelView.")
        if result.evidence_reconciliation.level_view_id != level_view.id:
            raise RuntimeError("PerimeterEvidenceReconciliationResult pertenece a otro LevelView.")
        if result.resolution.level_view_id != level_view.id:
            raise RuntimeError("PerimeterResolutionResult pertenece a otro LevelView.")
        if result.validation.level_view_id != level_view.id:
            raise RuntimeError("PerimeterWallValidationResult pertenece a otro LevelView.")
        if result.state != result.validation.state:
            raise RuntimeError("El estado final no coincide con la validación.")

        if result.state == "VALID":
            if result.wall_layer is None:
                raise RuntimeError("F02 quedó VALID sin wall_layer.")
            if result.resolution.state != "RESOLVED":
                raise RuntimeError("F02 quedó VALID sin resolución RESOLVED.")

        if result.wall_layer is not None:
            if result.wall_layer.confirmed:
                raise RuntimeError("PerimeterWallLayer salió confirmed=True.")
            if any(run.confirmed for run in result.wall_layer.wall_runs):
                raise RuntimeError("PerimeterWallRun salió confirmed=True.")
