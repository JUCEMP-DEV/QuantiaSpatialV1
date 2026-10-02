from __future__ import annotations

import math
from typing import Sequence

from shapely.geometry import LineString, Point

from .candidate_models import (
    CandidateGraphDiagnostics,
    CandidateRelation,
    WallCandidate,
    WallCandidateGraph,
)


class WallCandidateGraphBuilder:
    """Construye relaciones estructurales y conflictos antes de resolver.

    Selection Contracts V1.2 impide que líneas del mismo patrón repetitivo se
    premien entre sí mediante JUNCTION/CONTINUATION.
    """

    MIN_JUNCTION_ANGLE_DEG = 25.0
    MAX_JUNCTION_ANGLE_DEG = 155.0
    NOISE_CONTEXT_TYPES = {
        "STAIR_FLIGHT_REGION",
        "FLOOR_FINISH_GRID_REGION",
        "FURNITURE_MODULE_REGION",
        "HATCH_FILL_REGION",
        "UNKNOWN_REPETITIVE_REGION",
        "REPETITIVE_GRID_REGION",
        "REPETITIVE_PARALLEL_REGION",
    }

    def build(
        self,
        *,
        level_view_id: str,
        candidates: Sequence[WallCandidate],
    ) -> WallCandidateGraph:
        relations: list[CandidateRelation] = []
        median_thickness = self._median([item.thickness_px for item in candidates]) or 4.0
        px_per_m = self._metric_px_per_m(candidates)
        junction_tol = max(2.0, 0.90 * median_thickness)
        continuation_gap = max(3.0, 1.7 * median_thickness)
        if px_per_m is not None:
            # La mediana de candidatos puede quedar contaminada por módulos de
            # baldosa/escalera. La escala V2.3 limita la tolerancia relacional a
            # distancias físicamente razonables sin asumir un espesor de muro.
            junction_tol = min(junction_tol, max(3.0, 0.18 * px_per_m))
            continuation_gap = min(continuation_gap, max(4.0, 0.25 * px_per_m))

        for index, first in enumerate(candidates):
            line_a = self._line(first)
            for second in candidates[index + 1 :]:
                line_b = self._line(second)
                angle = self._angle_diff(first.angle_deg, second.angle_deg)
                if angle <= 3.0:
                    overlap = self._parallel_overlap(first, second)
                    distance = line_a.distance(line_b)
                    # Alternativas casi coincidentes: no se deben publicar las dos.
                    if overlap >= 0.58 and distance <= max(2.0, 0.55 * max(first.thickness_px, second.thickness_px)):
                        relations.append(
                            CandidateRelation(
                                a_id=first.id,
                                b_id=second.id,
                                relation_type="CONFLICT_ALTERNATIVE",
                                strength=max(0.6, overlap),
                                hard_conflict=True,
                            )
                        )
                        continue
                    gap = self._endpoint_gap(first, second)
                    thickness_compatibility = self._thickness_compatibility(first, second)
                    if (
                        gap <= continuation_gap
                        and distance <= max(2.0, 0.45 * median_thickness)
                        and thickness_compatibility >= 0.45
                        and not self._blocks_positive_relation(first, second)
                    ):
                        base_strength = max(0.0, min(1.0, 1.0 - gap / max(continuation_gap, 1.0)))
                        relations.append(
                            CandidateRelation(
                                a_id=first.id,
                                b_id=second.id,
                                relation_type="CONTINUATION",
                                strength=max(0.0, min(1.0, base_strength * thickness_compatibility)),
                            )
                        )
                else:
                    distance = line_a.distance(line_b)
                    if distance > junction_tol:
                        continue
                    if not (self.MIN_JUNCTION_ANGLE_DEG <= angle <= self.MAX_JUNCTION_ANGLE_DEG):
                        continue
                    point = self._junction_point(line_a, line_b)
                    endpoint_a = self._endpoint_distance(first, point)
                    endpoint_b = self._endpoint_distance(second, point)
                    if min(endpoint_a, endpoint_b) <= junction_tol:
                        if self._blocks_positive_relation(first, second):
                            continue
                        strength = max(0.0, min(1.0, 1.0 - min(endpoint_a, endpoint_b) / max(junction_tol, 1e-6)))
                        relations.append(
                            CandidateRelation(
                                a_id=first.id,
                                b_id=second.id,
                                relation_type="JUNCTION",
                                strength=max(0.35, strength),
                            )
                        )
                    else:
                        relations.append(
                            CandidateRelation(
                                a_id=first.id,
                                b_id=second.id,
                                relation_type="CROSSING",
                                strength=0.25,
                            )
                        )

        relation_keys: set[tuple] = set()
        unique: list[CandidateRelation] = []
        for relation in relations:
            key = tuple(sorted((relation.a_id, relation.b_id))) + (relation.relation_type,)
            if key in relation_keys:
                continue
            relation_keys.add(key)
            unique.append(relation)

        return WallCandidateGraph(
            level_view_id=level_view_id,
            candidates=list(candidates),
            relations=unique,
            diagnostics=CandidateGraphDiagnostics(
                candidate_count=len(candidates),
                relation_count=len(unique),
                hard_conflict_count=sum(1 for item in unique if item.hard_conflict),
                junction_count=sum(1 for item in unique if item.relation_type == "JUNCTION"),
                continuation_count=sum(1 for item in unique if item.relation_type == "CONTINUATION"),
                duplicate_count=sum(1 for item in unique if item.relation_type == "DUPLICATE"),
            ),
        )


    @classmethod
    def _blocks_positive_relation(cls, first: WallCandidate, second: WallCandidate) -> bool:
        """Evita que el ruido contextual se autoconfirme como topología."""
        first_region = first.metadata.get("context_region_id")
        second_region = second.metadata.get("context_region_id")
        first_type = first.metadata.get("context_region_type")
        second_type = second.metadata.get("context_region_type")
        if (
            first_region
            and second_region
            and first_region == second_region
            and (first_type in cls.NOISE_CONTEXT_TYPES or second_type in cls.NOISE_CONTEXT_TYPES)
        ):
            return True

        first_risk = cls._metadata_float(first.metadata.get("context_risk"))
        second_risk = cls._metadata_float(second.metadata.get("context_risk"))
        if (
            first.metadata.get("context_state") == "REVIEW"
            and second.metadata.get("context_state") == "REVIEW"
            and first_risk >= 0.45
            and second_risk >= 0.45
        ):
            return True
        return False

    @staticmethod
    def _thickness_compatibility(first: WallCandidate, second: WallCandidate) -> float:
        a = max(float(first.thickness_px), 1e-6)
        b = max(float(second.thickness_px), 1e-6)
        return max(0.0, min(1.0, min(a, b) / max(a, b)))

    @classmethod
    def _metric_px_per_m(cls, candidates: Sequence[WallCandidate]) -> float | None:
        values: list[float] = []
        for candidate in candidates:
            raw = candidate.metadata.get("metric_thickness_m")
            try:
                thickness_m = float(raw)
            except (TypeError, ValueError):
                continue
            if thickness_m <= 1e-9:
                continue
            values.append(float(candidate.thickness_px) / thickness_m)
        return cls._median(values)

    @staticmethod
    def _metadata_float(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
    @staticmethod
    def _line(candidate: WallCandidate) -> LineString:
        return LineString([(candidate.start.x, candidate.start.y), (candidate.end.x, candidate.end.y)])

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _median(values: Sequence[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return float((ordered[middle - 1] + ordered[middle]) / 2.0)

    @staticmethod
    def _endpoint_gap(first: WallCandidate, second: WallCandidate) -> float:
        points_a = [Point(first.start.x, first.start.y), Point(first.end.x, first.end.y)]
        points_b = [Point(second.start.x, second.start.y), Point(second.end.x, second.end.y)]
        return min(a.distance(b) for a in points_a for b in points_b)

    @staticmethod
    def _endpoint_distance(candidate: WallCandidate, point: Point) -> float:
        return min(
            point.distance(Point(candidate.start.x, candidate.start.y)),
            point.distance(Point(candidate.end.x, candidate.end.y)),
        )

    @staticmethod
    def _junction_point(first: LineString, second: LineString) -> Point:
        intersection = first.intersection(second)
        if not intersection.is_empty:
            if intersection.geom_type == "Point":
                return intersection
            return intersection.centroid
        p1, p2 = first.interpolate(first.project(second.centroid)), second.interpolate(second.project(first.centroid))
        return Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)

    @staticmethod
    def _parallel_overlap(first: WallCandidate, second: WallCandidate) -> float:
        angle = math.radians(first.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        a = [first.start.x * ux + first.start.y * uy, first.end.x * ux + first.end.y * uy]
        b = [second.start.x * ux + second.start.y * uy, second.end.x * ux + second.end.y * uy]
        a0, a1 = min(a), max(a)
        b0, b1 = min(b), max(b)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        return max(0.0, min(1.0, overlap / max(1e-6, min(a1 - a0, b1 - b0))))
