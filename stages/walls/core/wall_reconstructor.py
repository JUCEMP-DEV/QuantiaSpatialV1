from __future__ import annotations

import hashlib
import math

from shapely.geometry import LineString

from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel
from app.quantia_spatialV1.core.models.parametric import (
    ParametricLevel,
    ParametricPoint,
    ParametricWall,
    QuantiaParametricModel,
)

from .candidate_models import WallCandidateGraph
from .global_topology_solver import GlobalTopologySolution
from .perimeter_adapter import line_is_inside_perimeter, perimeter_polygon


class ParametricWallReconstructor:
    def build(
        self,
        *,
        level_view: LevelView,
        perimeter: EditablePerimeterModel,
        graph: WallCandidateGraph,
        solution: GlobalTopologySolution,
    ) -> QuantiaParametricModel:
        candidate_by_id = {item.id: item for item in graph.candidates}
        vertices = {item.id: item for item in perimeter.vertices}
        walls: list[ParametricWall] = []
        metric_scale = perimeter.metric_summary.metric_scale_m_per_px

        # F02 se copia, no se reinterpreta.
        for wall in sorted(perimeter.walls, key=lambda item: item.sequence_index):
            start = vertices[wall.start_vertex_id].point_px
            end = vertices[wall.end_vertex_id].point_px
            walls.append(
                ParametricWall(
                    id=wall.id,
                    level_id=level_view.id,
                    role="PERIMETER",
                    reference_path=[
                        ParametricPoint(x_px=float(start.x), y_px=float(start.y)),
                        ParametricPoint(x_px=float(end.x), y_px=float(end.y)),
                    ],
                    anchor_mode="F02_FIXED",
                    length_px=float(wall.length_px),
                    length_m=wall.length_m,
                    source_f02_wall_id=wall.id,
                    evidence_ids=list(wall.evidence_ids),
                    confidence=1.0,
                    editable=True,
                )
            )

        polygon = perimeter_polygon(perimeter)
        for index, candidate_id in enumerate(solution.selected_candidate_ids, 1):
            candidate = candidate_by_id[candidate_id]
            candidate_line = LineString([
                (candidate.start.x, candidate.start.y),
                (candidate.end.x, candidate.end.y),
            ])
            if not line_is_inside_perimeter(
                line=candidate_line,
                polygon=polygon,
                tolerance_px=2.0,
            ):
                raise ValueError(
                    f"Solver intentó publicar un DIVIDER fuera de F02: {candidate.id}"
                )
            length_m = candidate.length_px * metric_scale if metric_scale else None
            thickness_m = candidate.thickness_px * metric_scale if metric_scale else None
            ident = hashlib.sha1(f"{level_view.id}|{candidate.id}".encode()).hexdigest()[:16]
            walls.append(
                ParametricWall(
                    id=f"{level_view.id}__QPM_WALL_INT__{index:03d}_{ident}",
                    level_id=level_view.id,
                    role="DIVIDER",
                    reference_path=[
                        ParametricPoint(x_px=candidate.start.x, y_px=candidate.start.y),
                        ParametricPoint(x_px=candidate.end.x, y_px=candidate.end.y),
                    ],
                    anchor_mode="CENTERLINE",
                    length_px=candidate.length_px,
                    length_m=length_m,
                    thickness_px=candidate.thickness_px,
                    thickness_m=thickness_m,
                    source_candidate_id=candidate.id,
                    evidence_ids=list(candidate.evidence_ids),
                    confidence=candidate.prior_score,
                    editable=True,
                )
            )

        level = ParametricLevel(
            id=level_view.id,
            name=level_view.level_name,
            source_level_view_id=level_view.id,
            source_f02_model_id=perimeter.id,
            walls=walls,
        )
        model_id = hashlib.sha1(f"QPM|{level_view.id}|{perimeter.id}".encode()).hexdigest()[:16]
        return QuantiaParametricModel(
            id=f"QUANTIA_PARAMETRIC__{model_id}",
            levels=[level],
            project_axes=[],
            model_revision=0,
        )
