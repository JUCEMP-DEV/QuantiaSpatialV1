from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from shapely.geometry import LineString

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView

from .perimeter_models import (
    PerimeterCandidate,
    PerimeterDrawingOrientation,
    PerimeterDrawingSide,
    PerimeterDrawingSideSummary,
    PerimeterPointPx,
    PerimeterWallLayer,
    PerimeterWallRun,
)


class PerimeterWallBuilder:
    """
    Convierte el boundary resuelto en tramos únicos de muro perimetral.

    La representación geométrica se ancla al boundary exterior. No afirma
    centerline ni espesor si esos datos no están fundamentados.

    No asigna cardinalidad NORTH/SOUTH/EAST/WEST por posición de pantalla.
    Solo publica orientación geométrica HORIZONTAL/VERTICAL/DIAGONAL y lado
    visual TOP/BOTTOM/LEFT/RIGHT cuando la relación es exacta.
    """

    def build(
        self,
        *,
        level_view: LevelView,
        candidate: PerimeterCandidate,
        evidence: Sequence[RawEvidence],
    ) -> PerimeterWallLayer:
        if candidate.level_view_id != level_view.id:
            raise ValueError("El candidato perimetral pertenece a otro LevelView.")
        if candidate.confirmed:
            raise ValueError("El candidato perimetral llegó confirmed=True.")

        raw = list(evidence)
        raw_by_id = {item.id: item for item in raw}
        if len(raw_by_id) != len(raw):
            raise ValueError("RawEvidence.id duplicado recibido por PerimeterWallBuilder.")

        points = [(point.x, point.y) for point in candidate.geometry.points]
        merged_edges = self._merge_collinear_ring_edges(points)
        if len(merged_edges) < 3:
            raise ValueError("El contorno no produce al menos 3 tramos perimetrales.")

        bbox = candidate.geometry.bbox
        wall_runs: list[PerimeterWallRun] = []

        for index, (start, end) in enumerate(merged_edges):
            length_px = math.hypot(end[0] - start[0], end[1] - start[1])
            if length_px <= 0.0:
                continue

            orientation = self._drawing_orientation(start=start, end=end)
            drawing_side = self._drawing_side(
                start=start,
                end=end,
                orientation=orientation,
                bbox=(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max),
            )
            evidence_ids = self._evidence_for_segment(
                start=start,
                end=end,
                candidate=candidate,
                raw_by_id=raw_by_id,
            )

            wall_runs.append(
                PerimeterWallRun(
                    id=self._wall_run_id(
                        level_view_id=level_view.id,
                        candidate_id=candidate.id,
                        index=index,
                        start=start,
                        end=end,
                    ),
                    level_view_id=level_view.id,
                    sequence_index=index,
                    start_px=PerimeterPointPx(x=start[0], y=start[1]),
                    end_px=PerimeterPointPx(x=end[0], y=end[1]),
                    drawing_orientation=orientation,
                    drawing_side=drawing_side,
                    cardinal_side=None,
                    representation="EXTERIOR_BOUNDARY_REFERENCE",
                    length_px=length_px,
                    length_m=None,
                    metric_status="UNRESOLVED",
                    geometry_source=candidate.geometry_source,
                    evidence_ids=evidence_ids,
                    dimensional_evidence_ids=[],
                    confirmed=False,
                )
            )

        if len(wall_runs) < 3:
            raise ValueError("No fue posible construir una capa perimetral cerrada.")

        evidence_ids = sorted(
            {evidence_id for run in wall_runs for evidence_id in run.evidence_ids}
            | {ref.evidence_id for ref in candidate.evidence}
        )

        return PerimeterWallLayer(
            id=f"{level_view.id}__PERIMETER_WALL_LAYER",
            level_view_id=level_view.id,
            candidate_id=candidate.id,
            geometry_source=candidate.geometry_source,
            polygon=candidate.geometry.model_copy(deep=True),
            wall_runs=wall_runs,
            drawing_side_summaries=self._build_side_summaries(wall_runs),
            total_length_px=sum(run.length_px for run in wall_runs),
            total_length_m=None,
            metric_status="UNRESOLVED",
            metric_scale=None,
            evidence_ids=evidence_ids,
            confirmed=False,
        )

    @staticmethod
    def _merge_collinear_ring_edges(
        points: Sequence[tuple[float, float]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        if len(points) < 3:
            return []

        edges = [
            (points[index], points[(index + 1) % len(points)])
            for index in range(len(points))
            if points[index] != points[(index + 1) % len(points)]
        ]
        if len(edges) < 3:
            return []

        changed = True
        while changed and len(edges) >= 3:
            changed = False
            result: list[tuple[tuple[float, float], tuple[float, float]]] = []
            index = 0
            while index < len(edges):
                if index + 1 < len(edges) and PerimeterWallBuilder._can_merge_collinear(
                    edges[index], edges[index + 1]
                ):
                    result.append((edges[index][0], edges[index + 1][1]))
                    index += 2
                    changed = True
                else:
                    result.append(edges[index])
                    index += 1
            edges = result

        while len(edges) >= 3 and PerimeterWallBuilder._can_merge_collinear(
            edges[-1], edges[0]
        ):
            merged = (edges[-1][0], edges[0][1])
            edges = [merged, *edges[1:-1]]

        return edges

    @staticmethod
    def _can_merge_collinear(
        first: tuple[tuple[float, float], tuple[float, float]],
        second: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        if first[1] != second[0]:
            return False

        dx1 = first[1][0] - first[0][0]
        dy1 = first[1][1] - first[0][1]
        dx2 = second[1][0] - second[0][0]
        dy2 = second[1][1] - second[0][1]

        cross = dx1 * dy2 - dy1 * dx2
        dot = dx1 * dx2 + dy1 * dy2
        return cross == 0.0 and dot > 0.0

    @staticmethod
    def _drawing_orientation(
        *,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> PerimeterDrawingOrientation:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if dy == 0.0:
            return "HORIZONTAL"
        if dx == 0.0:
            return "VERTICAL"
        return "DIAGONAL"

    @staticmethod
    def _drawing_side(
        *,
        start: tuple[float, float],
        end: tuple[float, float],
        orientation: PerimeterDrawingOrientation,
        bbox: tuple[float, float, float, float],
    ) -> PerimeterDrawingSide:
        x_min, y_min, x_max, y_max = bbox
        if orientation == "HORIZONTAL":
            if start[1] == y_min and end[1] == y_min:
                return "TOP"
            if start[1] == y_max and end[1] == y_max:
                return "BOTTOM"
        if orientation == "VERTICAL":
            if start[0] == x_min and end[0] == x_min:
                return "LEFT"
            if start[0] == x_max and end[0] == x_max:
                return "RIGHT"
        return "AMBIGUOUS"

    @staticmethod
    def _evidence_for_segment(
        *,
        start: tuple[float, float],
        end: tuple[float, float],
        candidate: PerimeterCandidate,
        raw_by_id: dict[str, RawEvidence],
    ) -> list[str]:
        segment = LineString([start, end])
        candidate_ids = {ref.evidence_id for ref in candidate.evidence}
        supported: set[str] = set()

        for evidence_id in candidate_ids:
            evidence = raw_by_id.get(evidence_id)
            if evidence is None:
                continue
            geometry = evidence.geometry

            try:
                if geometry.geometry_type == "SEGMENT" and len(geometry.points) == 2:
                    line = LineString(
                        [(float(point.x), float(point.y)) for point in geometry.points]
                    )
                    if segment.intersection(line).length > 0.0:
                        supported.add(evidence_id)

                elif geometry.geometry_type == "POLYLINE" and len(geometry.points) >= 2:
                    line = LineString(
                        [(float(point.x), float(point.y)) for point in geometry.points]
                    )
                    if segment.intersection(line).length > 0.0:
                        supported.add(evidence_id)
            except Exception:
                continue

        return sorted(supported)

    @staticmethod
    def _build_side_summaries(
        wall_runs: Sequence[PerimeterWallRun],
    ) -> list[PerimeterDrawingSideSummary]:
        order: tuple[PerimeterDrawingSide, ...] = (
            "TOP",
            "BOTTOM",
            "LEFT",
            "RIGHT",
            "AMBIGUOUS",
        )
        result: list[PerimeterDrawingSideSummary] = []
        for drawing_side in order:
            selected = [run for run in wall_runs if run.drawing_side == drawing_side]
            result.append(
                PerimeterDrawingSideSummary(
                    drawing_side=drawing_side,
                    wall_run_count=len(selected),
                    length_px=sum(run.length_px for run in selected),
                    length_m=None,
                )
            )
        return result

    @staticmethod
    def _wall_run_id(
        *,
        level_view_id: str,
        candidate_id: str,
        index: int,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> str:
        payload = f"{candidate_id}|{index}|{start}|{end}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"{level_view_id}__PERIMETER_WALL_RUN_{index + 1}__{digest}"
