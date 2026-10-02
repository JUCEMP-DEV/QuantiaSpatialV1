from __future__ import annotations

from app.quantia_spatialV1.stages.walls.core.context_models import ContextRegion

from .contracts import ElementFeatureVector, ElementClass, HostWallMatch


class ElementFeatureBuilder:
    def build(
        self,
        *,
        region: ContextRegion,
        host_wall: HostWallMatch,
    ) -> ElementFeatureVector:
        meta = region.metadata
        expected = self._expected_class(region.region_type)
        source = str(meta.get("source") or "").upper()

        door_arc = self._f(meta.get("door_arc_score"))
        door_leaf = self._f(meta.get("door_leaf_support"))
        door_host = self._f(meta.get("door_host_support"))
        window_frame = max(
            self._f(meta.get("parallel_overlap_support")),
            min(1.0, self._f(meta.get("track_count")) / 4.0),
        )
        window_host = max(
            self._f(meta.get("host_wall_support")),
            self._f(meta.get("host_embedding_support")),
        )
        semantic = max(
            self._f(meta.get("semantic_search_support")),
            self._f(region.semantic_support),
        )
        verified = 1.0 if bool(meta.get("opening_verified")) else 0.0
        architecture = self._f(meta.get("architecture_support"))
        through = self._f(meta.get("through_wall_support"))
        gap = self._f(meta.get("wall_gap_support"))
        member_support = min(1.0, len(region.member_line_ids) / 4.0)

        tags: list[str] = []
        if "BEZIER" in source or door_arc > 0.0:
            tags.append("CURVE_SWING")
        if "WINDOW_FRAME" in source or window_frame > 0.0:
            tags.append("PARALLEL_FRAME")
        if verified >= 1.0:
            tags.append("OPENING_VERIFIED")
        if through >= 0.55:
            tags.append("CONTINUOUS_WALL")
        if semantic >= 0.45:
            tags.append("SEMANTIC_SUPPORT")
        if host_wall.score >= 0.55:
            tags.append("HOST_WALL_MATCH")
        if gap >= 0.55:
            tags.append("WALL_GAP")

        return ElementFeatureVector(
            source_region_type=region.region_type,
            region_confidence=region.confidence,
            opening_verified=verified,
            semantic_support=semantic,
            architecture_support=architecture,
            host_local_support=max(door_host, window_host),
            host_wall_match=host_wall.score,
            door_arc_support=door_arc,
            door_leaf_support=door_leaf,
            window_frame_support=window_frame,
            wall_through_support=through,
            wall_gap_support=gap,
            member_count_support=member_support,
            expected_class=expected,
            tags=tags,
        )

    @staticmethod
    def _expected_class(region_type: str) -> ElementClass:
        if region_type == "DOOR_REGION":
            return "DOOR"
        if region_type == "WINDOW_REGION":
            return "WINDOW"
        return "NOT_OPENING"

    @staticmethod
    def _f(value) -> float:
        try:
            return max(0.0, min(1.0, float(value or 0.0)))
        except (TypeError, ValueError):
            return 0.0
