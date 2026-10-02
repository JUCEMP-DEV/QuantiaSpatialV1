from __future__ import annotations

import math

from app.quantia_spatialV1.core.models.level_view import LevelView

from .perimeter_delivery import EditablePerimeterModel
from .perimeter_models import (
    PerimeterBBoxPx,
    PerimeterCandidate,
    PerimeterCandidateEvidenceReconciliation,
    PerimeterCandidateSemanticComparison,
    PerimeterDimensionGroundingResult,
    PerimeterDimensionReference,
    PerimeterEvidenceReconciliationResult,
    PerimeterMetricScale,
    PerimeterMotorSegmentSupport,
    PerimeterPointPx,
    PerimeterPolygonPx,
    PerimeterSemanticComparisonResult,
    PerimeterTextEvidenceObservation,
    PerimeterWallLayer,
    PerimeterWallRun,
)
from .perimeter_wall_graph import PerimeterGraphEdge, PerimeterGraphNode, PerimeterWallGraphResult
from .perimeter_wall_pipeline import PerimeterWallPipelineResult


class PerimeterRasterReprojector:
    """Reproyecta la verdad F02 validada a un nuevo raster del mismo LevelView.

    No redetecta ni reinterpreta el perímetro. Únicamente cambia su sistema de
    coordenadas px cuando F01 publica un raster canónico con otra densidad.
    Las longitudes métricas ya fundamentadas permanecen invariantes.
    """

    VERSION = "PERIMETER_RASTER_REPROJECTOR_V1"

    def reproject(
        self,
        *,
        source_result: PerimeterWallPipelineResult,
        source_level_view: LevelView,
        target_level_view: LevelView,
    ) -> PerimeterWallPipelineResult:
        if source_level_view.id != target_level_view.id:
            raise ValueError("La reproyección F02 requiere el mismo LevelView.id.")
        if source_result.level_view_id != source_level_view.id:
            raise ValueError("PerimeterWallPipelineResult pertenece a otro LevelView.")

        sx = float(target_level_view.raster_width_px) / float(source_level_view.raster_width_px)
        sy = float(target_level_view.raster_height_px) / float(source_level_view.raster_height_px)
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("Factores de reproyección F02 inválidos.")

        candidates = [self._candidate(item, sx=sx, sy=sy) for item in source_result.candidates]
        candidate_by_id = {item.id: item for item in candidates}

        resolution_candidates = [
            self._candidate(item, sx=sx, sy=sy) for item in source_result.resolution.candidates
        ]
        resolution_by_id = {item.id: item for item in resolution_candidates}
        selected = None
        if source_result.resolution.selected is not None:
            selected = resolution_by_id.get(source_result.resolution.selected.id)
            if selected is None:
                selected = self._candidate(source_result.resolution.selected, sx=sx, sy=sy)
        resolution = source_result.resolution.model_copy(
            deep=True,
            update={"candidates": resolution_candidates, "selected": selected},
        )

        graph_nodes = [
            item.model_copy(deep=True, update={"point": self._point(item.point, sx=sx, sy=sy)})
            for item in source_result.wall_graph.nodes
        ]
        graph_edges = [self._graph_edge(item, sx=sx, sy=sy) for item in source_result.wall_graph.edges]
        graph_candidate = None
        if source_result.wall_graph.candidate is not None:
            graph_candidate = candidate_by_id.get(source_result.wall_graph.candidate.id)
            if graph_candidate is None:
                graph_candidate = self._candidate(source_result.wall_graph.candidate, sx=sx, sy=sy)
        wall_graph = source_result.wall_graph.model_copy(
            deep=True,
            update={"nodes": graph_nodes, "edges": graph_edges, "candidate": graph_candidate},
        )

        reconciliation = self._reconciliation(
            source_result.evidence_reconciliation,
            sx=sx,
            sy=sy,
        )

        motor_wall_layer = self._wall_layer(source_result.motor_wall_layer, sx=sx, sy=sy)
        wall_layer = self._wall_layer(source_result.wall_layer, sx=sx, sy=sy)
        motor_grounding = self._grounding(source_result.motor_grounding, sx=sx, sy=sy)
        grounding = self._grounding(source_result.grounding, sx=sx, sy=sy)
        editable = self._editable(source_result.editable_perimeter, sx=sx, sy=sy)
        semantic = self._semantic(source_result.semantic_comparison, sx=sx, sy=sy)

        warnings = list(source_result.warnings)
        warnings.append(
            f"{self.VERSION}: F02 reproyectado {sx:.6f}x/{sy:.6f}y sin redetección."
        )

        return source_result.model_copy(
            deep=True,
            update={
                "candidates": candidates,
                "wall_graph": wall_graph,
                "evidence_reconciliation": reconciliation,
                "resolution": resolution,
                "motor_wall_layer": motor_wall_layer,
                "motor_grounding": motor_grounding,
                "wall_layer": wall_layer,
                "grounding": grounding,
                "editable_perimeter": editable,
                "semantic_comparison": semantic,
                "warnings": list(dict.fromkeys(warnings)),
            },
        )

    @staticmethod
    def _point(point: PerimeterPointPx, *, sx: float, sy: float) -> PerimeterPointPx:
        return PerimeterPointPx(x=float(point.x) * sx, y=float(point.y) * sy)

    @staticmethod
    def _bbox(bbox: PerimeterBBoxPx | None, *, sx: float, sy: float) -> PerimeterBBoxPx | None:
        if bbox is None:
            return None
        return PerimeterBBoxPx(
            x_min=float(bbox.x_min) * sx,
            y_min=float(bbox.y_min) * sy,
            x_max=float(bbox.x_max) * sx,
            y_max=float(bbox.y_max) * sy,
        )

    @classmethod
    def _polygon(cls, polygon: PerimeterPolygonPx, *, sx: float, sy: float) -> PerimeterPolygonPx:
        return PerimeterPolygonPx(points=[cls._point(item, sx=sx, sy=sy) for item in polygon.points])

    @classmethod
    def _candidate(cls, candidate: PerimeterCandidate, *, sx: float, sy: float) -> PerimeterCandidate:
        return candidate.model_copy(
            deep=True,
            update={"geometry": cls._polygon(candidate.geometry, sx=sx, sy=sy)},
        )

    @staticmethod
    def _orientation_scale(orientation: str, *, sx: float, sy: float) -> float:
        value = str(orientation or "").upper()
        if value == "HORIZONTAL":
            return sx
        if value == "VERTICAL":
            return sy
        return math.sqrt(sx * sy)

    @classmethod
    def _graph_edge(cls, edge: PerimeterGraphEdge, *, sx: float, sy: float) -> PerimeterGraphEdge:
        factor = cls._orientation_scale(edge.orientation, sx=sx, sy=sy)
        return edge.model_copy(deep=True, update={"length_px": float(edge.length_px) * factor})

    @classmethod
    def _segment(
        cls,
        item: PerimeterMotorSegmentSupport,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterMotorSegmentSupport:
        start = cls._point(item.start_px, sx=sx, sy=sy)
        end = cls._point(item.end_px, sx=sx, sy=sy)
        length = math.hypot(end.x - start.x, end.y - start.y)
        return item.model_copy(
            deep=True,
            update={"start_px": start, "end_px": end, "length_px": length},
        )

    @classmethod
    def _text_observation(
        cls,
        item: PerimeterTextEvidenceObservation,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterTextEvidenceObservation:
        return item.model_copy(
            deep=True,
            update={"bbox_px": cls._bbox(item.bbox_px, sx=sx, sy=sy)},
        )

    @classmethod
    def _candidate_reconciliation(
        cls,
        item: PerimeterCandidateEvidenceReconciliation,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterCandidateEvidenceReconciliation:
        return item.model_copy(
            deep=True,
            update={
                "segment_support": [cls._segment(value, sx=sx, sy=sy) for value in item.segment_support],
                "localized_text_evidence": [
                    cls._text_observation(value, sx=sx, sy=sy)
                    for value in item.localized_text_evidence
                ],
            },
        )

    @classmethod
    def _reconciliation(
        cls,
        result: PerimeterEvidenceReconciliationResult,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterEvidenceReconciliationResult:
        return result.model_copy(
            deep=True,
            update={
                "candidates": [
                    cls._candidate_reconciliation(item, sx=sx, sy=sy)
                    for item in result.candidates
                ],
                "unassigned_text_evidence": [
                    cls._text_observation(item, sx=sx, sy=sy)
                    for item in result.unassigned_text_evidence
                ],
            },
        )

    @classmethod
    def _metric_scale(
        cls,
        scale: PerimeterMetricScale | None,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterMetricScale | None:
        if scale is None:
            return None
        factor = math.sqrt(sx * sy)
        return scale.model_copy(
            deep=True,
            update={"meters_per_px": float(scale.meters_per_px) / factor},
        )

    @classmethod
    def _dimension_reference(
        cls,
        item: PerimeterDimensionReference,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterDimensionReference:
        factor = cls._orientation_scale(item.orientation, sx=sx, sy=sy)
        return item.model_copy(
            deep=True,
            update={
                "geometry_length_px": float(item.geometry_length_px) * factor,
                "meters_per_px": float(item.meters_per_px) / factor,
            },
        )

    @classmethod
    def _grounding(
        cls,
        result: PerimeterDimensionGroundingResult | None,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterDimensionGroundingResult | None:
        if result is None:
            return None
        return result.model_copy(
            deep=True,
            update={
                "dimension_references": [
                    cls._dimension_reference(item, sx=sx, sy=sy)
                    for item in result.dimension_references
                ],
                "metric_scale": cls._metric_scale(result.metric_scale, sx=sx, sy=sy),
            },
        )

    @classmethod
    def _wall_run(cls, run: PerimeterWallRun, *, sx: float, sy: float) -> PerimeterWallRun:
        start = cls._point(run.start_px, sx=sx, sy=sy)
        end = cls._point(run.end_px, sx=sx, sy=sy)
        length = math.hypot(end.x - start.x, end.y - start.y)
        return run.model_copy(
            deep=True,
            update={"start_px": start, "end_px": end, "length_px": length},
        )

    @classmethod
    def _wall_layer(
        cls,
        layer: PerimeterWallLayer | None,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterWallLayer | None:
        if layer is None:
            return None
        runs = [cls._wall_run(item, sx=sx, sy=sy) for item in layer.wall_runs]
        summaries = []
        for item in layer.drawing_side_summaries:
            side = str(item.drawing_side).upper()
            factor = sx if side in {"TOP", "BOTTOM"} else sy if side in {"LEFT", "RIGHT"} else math.sqrt(sx * sy)
            summaries.append(
                item.model_copy(deep=True, update={"length_px": float(item.length_px) * factor})
            )
        return layer.model_copy(
            deep=True,
            update={
                "polygon": cls._polygon(layer.polygon, sx=sx, sy=sy),
                "wall_runs": runs,
                "drawing_side_summaries": summaries,
                "total_length_px": sum(item.length_px for item in runs),
                "metric_scale": cls._metric_scale(layer.metric_scale, sx=sx, sy=sy),
            },
        )

    @classmethod
    def _editable(
        cls,
        perimeter: EditablePerimeterModel | None,
        *,
        sx: float,
        sy: float,
    ) -> EditablePerimeterModel | None:
        if perimeter is None:
            return None
        vertices = [
            item.model_copy(
                deep=True,
                update={"point_px": cls._point(item.point_px, sx=sx, sy=sy)},
            )
            for item in perimeter.vertices
        ]
        vertex_by_id = {item.id: item for item in vertices}
        walls = []
        for wall in perimeter.walls:
            start = vertex_by_id[wall.start_vertex_id].point_px
            end = vertex_by_id[wall.end_vertex_id].point_px
            walls.append(
                wall.model_copy(
                    deep=True,
                    update={"length_px": math.hypot(end.x - start.x, end.y - start.y)},
                )
            )
        metric_summary = perimeter.metric_summary.model_copy(deep=True)
        if metric_summary.metric_scale_m_per_px is not None:
            metric_summary = metric_summary.model_copy(
                deep=True,
                update={
                    "metric_scale_m_per_px": float(metric_summary.metric_scale_m_per_px)
                    / math.sqrt(sx * sy)
                },
            )
        return perimeter.model_copy(
            deep=True,
            update={"vertices": vertices, "walls": walls, "metric_summary": metric_summary},
        )

    @classmethod
    def _candidate_semantic(
        cls,
        item: PerimeterCandidateSemanticComparison,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterCandidateSemanticComparison:
        h_scale = float(item.horizontal_meters_per_px) / sx if item.horizontal_meters_per_px else None
        v_scale = float(item.vertical_meters_per_px) / sy if item.vertical_meters_per_px else None
        relative = item.relative_scale_difference
        if h_scale is not None and v_scale is not None:
            relative = abs(h_scale - v_scale) / max(h_scale, v_scale)
        return item.model_copy(
            deep=True,
            update={
                "motor_span_x_px": float(item.motor_span_x_px) * sx,
                "motor_span_y_px": float(item.motor_span_y_px) * sy,
                "horizontal_meters_per_px": h_scale,
                "vertical_meters_per_px": v_scale,
                "relative_scale_difference": relative,
                "motor_horizontal_segment_lengths_px": [float(value) * sx for value in item.motor_horizontal_segment_lengths_px],
                "motor_vertical_segment_lengths_px": [float(value) * sy for value in item.motor_vertical_segment_lengths_px],
            },
        )

    @classmethod
    def _semantic(
        cls,
        result: PerimeterSemanticComparisonResult,
        *,
        sx: float,
        sy: float,
    ) -> PerimeterSemanticComparisonResult:
        return result.model_copy(
            deep=True,
            update={
                "candidate_comparisons": [
                    cls._candidate_semantic(item, sx=sx, sy=sy)
                    for item in result.candidate_comparisons
                ]
            },
        )
