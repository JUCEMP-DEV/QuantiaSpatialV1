from __future__ import annotations

import math
from statistics import median
from typing import Iterable

from shapely.geometry import LineString, Point, box

from app.quantia_spatialV1.core.models.parametric import ParametricWall
from app.quantia_spatialV1.stages.walls.core.context_models import ContextRegion
from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingLine, DrawingModel

from .contracts import HostWallMatch


class HostWallMatcher:
    """Asocia una propuesta semántica con muros ya publicados por Spatial.

    El muro es autoridad geométrica. El mini-motor no crea, recorta ni elimina
    muros; solo propone una relación host_wall y un offset proyectado.
    """

    def match(
        self,
        *,
        region: ContextRegion,
        drawing: DrawingModel,
        walls: Iterable[ParametricWall],
    ) -> HostWallMatch:
        walls = list(walls)
        if not walls:
            return HostWallMatch(metadata={"reason": "NO_WALLS"})

        center = Point(
            (region.bbox.x_min + region.bbox.x_max) * 0.5,
            (region.bbox.y_min + region.bbox.y_max) * 0.5,
        )
        region_angle = self._region_orientation(region=region, drawing=drawing)
        host_tokens = set(region.metadata.get("host_line_ids") or [])
        hinted_wall_id = str(region.metadata.get("host_wall_hint_id") or "").strip() or None
        try:
            hinted_wall_score = max(0.0, min(1.0, float(region.metadata.get("host_wall_hint_score") or 0.0)))
        except (TypeError, ValueError):
            hinted_wall_score = 0.0
        major = max(
            region.bbox.x_max - region.bbox.x_min,
            region.bbox.y_max - region.bbox.y_min,
        )
        search_px = max(10.0, major * 0.45)

        best: HostWallMatch | None = None
        for wall in walls:
            if len(wall.reference_path) < 2:
                continue
            start = wall.reference_path[0]
            end = wall.reference_path[-1]
            line = LineString([(start.x_px, start.y_px), (end.x_px, end.y_px)])
            distance = float(line.distance(center))
            distance_support = max(0.0, 1.0 - distance / max(search_px, 1e-6))
            wall_angle = self._angle(start.x_px, start.y_px, end.x_px, end.y_px)
            orientation_support = self._orientation_support(region_angle, wall_angle)

            wall_tokens = set(wall.evidence_ids)
            if wall.source_candidate_id:
                wall_tokens.add(wall.source_candidate_id)
            if wall.source_f02_wall_id:
                wall_tokens.add(wall.source_f02_wall_id)
            provenance_support = 1.0 if host_tokens and (host_tokens & wall_tokens) else 0.0

            hint_support = hinted_wall_score if hinted_wall_id == wall.id else 0.0
            score = (
                0.56 * distance_support
                + 0.18 * orientation_support
                + 0.10 * provenance_support
                + 0.16 * hint_support
            )
            if provenance_support >= 1.0:
                score = max(score, 0.80)
            if hint_support >= 0.50:
                score = max(score, 0.74 * hint_support + 0.18 * distance_support + 0.08 * orientation_support)
            if distance > search_px * 1.45 and provenance_support <= 0.0:
                score *= 0.25

            offset = float(line.project(center))
            candidate = HostWallMatch(
                wall_id=wall.id if score >= 0.25 else None,
                score=max(0.0, min(1.0, score)),
                distance_px=max(0.0, distance),
                orientation_support=orientation_support,
                provenance_support=provenance_support,
                projected_offset_px=max(0.0, offset),
                metadata={
                    "region_angle_deg": region_angle,
                    "wall_angle_deg": wall_angle,
                    "search_distance_px": search_px,
                    "hinted_wall_id": hinted_wall_id,
                    "hint_support": hint_support,
                },
            )
            if best is None or candidate.score > best.score:
                best = candidate

        return best or HostWallMatch(metadata={"reason": "NO_VALID_WALL_GEOMETRY"})

    def _region_orientation(self, *, region: ContextRegion, drawing: DrawingModel) -> float:
        tokens = set(region.metadata.get("host_line_ids") or [])
        if not tokens:
            tokens = set(region.member_line_ids)
        angles: list[float] = []
        for line in drawing.lines:
            lineage = {line.id, *line.evidence_ids}
            if lineage & tokens:
                angles.append(float(line.angle_deg))
        if angles:
            return float(median(angles)) % 180.0

        width = region.bbox.x_max - region.bbox.x_min
        height = region.bbox.y_max - region.bbox.y_min
        return 0.0 if width >= height else 90.0

    @staticmethod
    def _angle(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0

    @staticmethod
    def _orientation_support(first: float, second: float) -> float:
        diff = abs(first - second) % 180.0
        diff = min(diff, 180.0 - diff)
        return max(0.0, 1.0 - diff / 35.0)
