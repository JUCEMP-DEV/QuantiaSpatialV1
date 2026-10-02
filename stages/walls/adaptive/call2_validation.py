from __future__ import annotations

import math
from typing import Any

from .contracts import (
    Call2DeltaValidationItem,
    Call2GapValidationItem,
    Call2ValidationResult,
    GapDecision,
    MultimodalWallReview,
    SingleLineWall,
    SingleLineWallGraph,
    WallDelta,
)


class Call2DeltaValidator:
    """Valida deltas multimodales antes de modificar el WallGraph.

    La respuesta cruda de Call 2 siempre se conserva. Este validador únicamente
    decide qué correcciones pueden pasar al aplicador determinista.
    """

    DEFAULT_MIN_CONFIDENCE = {
        "REMOVE_WALL": 0.82,
        "ADD_WALL": 0.78,
        "EXTEND_WALL": 0.78,
        "TRIM_WALL": 0.78,
        "REPOSITION_WALL": 0.80,
        "MERGE_WALLS": 0.80,
        "SPLIT_WALL": 0.80,
    }

    def validate(
        self,
        *,
        graph: SingleLineWallGraph,
        review: MultimodalWallReview,
    ) -> Call2ValidationResult:
        if review.level_view_id != graph.level_view_id:
            raise ValueError("Review pertenece a otro LevelView.")

        walls = {wall.id: wall for wall in graph.walls}
        accepted_deltas: list[WallDelta] = []
        delta_items: list[Call2DeltaValidationItem] = []

        for delta in review.deltas:
            accepted, reason = self._validate_delta(graph=graph, walls=walls, delta=delta)
            delta_items.append(Call2DeltaValidationItem(delta=delta, accepted=accepted, reason=reason))
            if accepted:
                accepted_deltas.append(delta)

        known_gaps = {gap.id for gap in graph.logical_gaps}
        accepted_gaps: list[GapDecision] = []
        gap_items: list[Call2GapValidationItem] = []
        for decision in review.gap_decisions:
            accepted, reason = self._validate_gap(decision=decision, known_gaps=known_gaps)
            gap_items.append(Call2GapValidationItem(gap_decision=decision, accepted=accepted, reason=reason))
            if accepted:
                accepted_gaps.append(decision)

        accepted_arch, rejected_arch = self._validate_regions(
            review.non_wall_architectural_regions,
            graph=graph,
            min_confidence=0.60,
        )
        accepted_unresolved, rejected_unresolved = self._validate_regions(
            review.unresolved_regions,
            graph=graph,
            min_confidence=0.0,
        )

        accepted_review = review.model_copy(update={
            "deltas": accepted_deltas,
            "gap_decisions": accepted_gaps,
            "non_wall_architectural_regions": accepted_arch,
            "unresolved_regions": accepted_unresolved,
        }, deep=True)

        return Call2ValidationResult(
            level_view_id=graph.level_view_id,
            accepted_review=accepted_review,
            delta_items=delta_items,
            gap_items=gap_items,
            accepted_architectural_regions=accepted_arch,
            rejected_architectural_regions=rejected_arch,
            accepted_unresolved_regions=accepted_unresolved,
            rejected_unresolved_regions=rejected_unresolved,
        )

    def _validate_delta(
        self,
        *,
        graph: SingleLineWallGraph,
        walls: dict[str, SingleLineWall],
        delta: WallDelta,
    ) -> tuple[bool, str]:
        threshold = self.DEFAULT_MIN_CONFIDENCE[delta.action]
        if delta.confidence < threshold:
            return False, f"confidence<{threshold:.2f}"

        if delta.action == "REMOVE_WALL":
            if not delta.wall_id or delta.wall_id not in walls:
                return False, "wall_id_missing_or_unknown"
            if walls[delta.wall_id].f03_seed_protected and delta.confidence < 0.90:
                return False, "seed_protected_remove_requires_0.90"
            return True, "valid_remove"

        if delta.action == "ADD_WALL":
            if delta.start_px is None or delta.end_px is None:
                return False, "add_requires_start_end"
            if not self._point_in_bounds(delta.start_px, graph) or not self._point_in_bounds(delta.end_px, graph):
                return False, "add_out_of_bounds"
            if self._distance(delta.start_px, delta.end_px) / graph.px_per_m < 0.15:
                return False, "add_too_short"
            return True, "valid_add"

        if delta.action in {"EXTEND_WALL", "TRIM_WALL"}:
            if not delta.wall_id or delta.wall_id not in walls:
                return False, "wall_id_missing_or_unknown"
            if delta.endpoint not in {"START", "END"} or delta.new_point_px is None:
                return False, "endpoint_and_new_point_required"
            if not self._point_in_bounds(delta.new_point_px, graph):
                return False, "new_point_out_of_bounds"
            wall = walls[delta.wall_id]
            other = wall.end_px if delta.endpoint == "START" else wall.start_px
            old_point = wall.start_px if delta.endpoint == "START" else wall.end_px
            old_length = self._distance(old_point, other)
            new_length = self._distance(delta.new_point_px, other)
            if new_length / graph.px_per_m < 0.15:
                return False, "resulting_wall_too_short"
            tolerance = 0.03 * graph.px_per_m
            if delta.action == "EXTEND_WALL" and new_length + tolerance < old_length:
                return False, "extend_would_shorten"
            if delta.action == "TRIM_WALL" and new_length - tolerance > old_length:
                return False, "trim_would_extend"
            return True, "valid_endpoint_edit"

        if delta.action == "REPOSITION_WALL":
            if not delta.wall_id or delta.wall_id not in walls:
                return False, "wall_id_missing_or_unknown"
            wall = walls[delta.wall_id]
            start = delta.start_px or wall.start_px
            end = delta.end_px or wall.end_px
            if not self._point_in_bounds(start, graph) or not self._point_in_bounds(end, graph):
                return False, "reposition_out_of_bounds"
            if self._distance(start, end) / graph.px_per_m < 0.15:
                return False, "reposition_too_short"
            return True, "valid_reposition"

        if delta.action == "MERGE_WALLS":
            ids = [wall_id for wall_id in delta.wall_ids if wall_id in walls]
            if len(ids) < 2 or len(ids) != len(delta.wall_ids):
                return False, "merge_requires_known_wall_ids"
            source = [walls[wall_id] for wall_id in ids]
            base_angle = self._angle(source[0])
            if any(self._angle_diff(base_angle, self._angle(item)) > 6.0 for item in source[1:]):
                return False, "merge_non_collinear"
            if self._max_endpoint_gap(source) / graph.px_per_m > 0.40:
                return False, "merge_gap_too_large"
            return True, "valid_merge"

        if delta.action == "SPLIT_WALL":
            if not delta.wall_id or delta.wall_id not in walls or delta.new_point_px is None:
                return False, "split_requires_wall_and_point"
            wall = walls[delta.wall_id]
            if not self._point_in_bounds(delta.new_point_px, graph):
                return False, "split_point_out_of_bounds"
            distance_to_segment = self._point_segment_distance(delta.new_point_px, wall.start_px, wall.end_px)
            if distance_to_segment / graph.px_per_m > 0.12:
                return False, "split_point_not_on_wall"
            if min(
                self._distance(delta.new_point_px, wall.start_px),
                self._distance(delta.new_point_px, wall.end_px),
            ) / graph.px_per_m < 0.10:
                return False, "split_point_too_close_to_endpoint"
            return True, "valid_split"

        return False, "unsupported_action"

    @staticmethod
    def _validate_gap(*, decision: GapDecision, known_gaps: set[str]) -> tuple[bool, str]:
        if decision.gap_id not in known_gaps:
            return False, "unknown_gap_id"
        if decision.classification == "PROBABLE_OPENING":
            if not decision.host_wall_continuity or decision.solid_wall_present:
                return False, "probable_opening_semantics_invalid"
        if decision.classification == "WALL_CONTINUITY":
            if not decision.host_wall_continuity or not decision.solid_wall_present:
                return False, "wall_continuity_semantics_invalid"
        return True, "valid_gap_decision"

    def _validate_regions(
        self,
        regions: list[dict[str, Any]],
        *,
        graph: SingleLineWallGraph,
        min_confidence: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for region in regions:
            confidence = float(region.get("confidence", 0.0) or 0.0)
            bbox = region.get("bbox_px")
            if confidence < min_confidence or not self._valid_bbox(bbox, graph):
                rejected.append(region)
            else:
                accepted.append(region)
        return accepted, rejected

    @staticmethod
    def _valid_bbox(bbox: Any, graph: SingleLineWallGraph) -> bool:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except Exception:
            return False
        width, height = graph.image_size_px
        return 0.0 <= x1 <= x2 <= width and 0.0 <= y1 <= y2 <= height

    @staticmethod
    def _point_in_bounds(point: tuple[float, float], graph: SingleLineWallGraph) -> bool:
        width, height = graph.image_size_px
        return 0.0 <= point[0] <= width and 0.0 <= point[1] <= height

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @staticmethod
    def _angle(wall: SingleLineWall) -> float:
        angle = math.degrees(math.atan2(wall.end_px[1] - wall.start_px[1], wall.end_px[0] - wall.start_px[0]))
        return angle % 180.0

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs(a - b) % 180.0
        return min(diff, 180.0 - diff)

    @classmethod
    def _max_endpoint_gap(cls, walls: list[SingleLineWall]) -> float:
        # Medida conservadora para evitar fusionar muros claramente separados.
        gaps: list[float] = []
        for i, first in enumerate(walls):
            for second in walls[i + 1:]:
                gaps.append(min(
                    cls._distance(first.start_px, second.start_px),
                    cls._distance(first.start_px, second.end_px),
                    cls._distance(first.end_px, second.start_px),
                    cls._distance(first.end_px, second.end_px),
                ))
        return max(gaps, default=0.0)

    @staticmethod
    def _point_segment_distance(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        px, py = point
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
        cx, cy = x1 + t * dx, y1 + t * dy
        return math.hypot(px - cx, py - cy)
