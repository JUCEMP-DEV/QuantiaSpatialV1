from __future__ import annotations

import math
from copy import deepcopy

from .contracts import MultimodalWallReview, SingleLineWall, SingleLineWallGraph


class WallGraphCorrectionApplier:
    """Aplica deltas de Call 2 de forma determinista y trazable."""

    def apply(
        self,
        *,
        graph: SingleLineWallGraph,
        review: MultimodalWallReview,
        min_confidence: float = 0.60,
    ) -> SingleLineWallGraph:
        if review.level_view_id != graph.level_view_id:
            raise ValueError("Review pertenece a otro LevelView.")

        walls = {item.id: item.model_copy(deep=True) for item in graph.walls}
        add_index = 1

        for delta in review.deltas:
            if delta.confidence < min_confidence:
                continue

            if delta.action == "REMOVE_WALL":
                if delta.wall_id:
                    walls.pop(delta.wall_id, None)
                continue

            if delta.action == "ADD_WALL":
                if delta.start_px is None or delta.end_px is None:
                    continue
                wall_id = delta.wall_id or f"CALL2_ADD_{add_index:03d}"
                add_index += 1
                walls[wall_id] = SingleLineWall(
                    id=wall_id,
                    start_px=delta.start_px,
                    end_px=delta.end_px,
                    thickness_px=max(2.0, 0.12 * graph.px_per_m),
                    role=delta.role_hint or "REVIEW",
                    confidence=delta.confidence,
                    source_candidate_ids=[],
                    evidence_ids=[],
                    source_names=["GEMINI_CALL2"],
                    f03_seed_protected=False,
                    context_state="CALL2_ADDED",
                    topology_support={"reason": delta.reason},
                )
                continue

            if delta.action in {"EXTEND_WALL", "TRIM_WALL"}:
                if not delta.wall_id or delta.wall_id not in walls or delta.new_point_px is None or delta.endpoint is None:
                    continue
                wall = walls[delta.wall_id]
                update = {
                    "start_px": delta.new_point_px if delta.endpoint == "START" else wall.start_px,
                    "end_px": delta.new_point_px if delta.endpoint == "END" else wall.end_px,
                    "confidence": max(wall.confidence, delta.confidence),
                    "context_state": f"CALL2_{delta.action}",
                }
                walls[delta.wall_id] = wall.model_copy(update=update)
                continue

            if delta.action == "REPOSITION_WALL":
                if not delta.wall_id or delta.wall_id not in walls:
                    continue
                wall = walls[delta.wall_id]
                walls[delta.wall_id] = wall.model_copy(update={
                    "start_px": delta.start_px or wall.start_px,
                    "end_px": delta.end_px or wall.end_px,
                    "confidence": max(wall.confidence, delta.confidence),
                    "context_state": "CALL2_REPOSITION",
                })
                continue

            if delta.action == "MERGE_WALLS":
                ids = [wall_id for wall_id in delta.wall_ids if wall_id in walls]
                if len(ids) < 2:
                    continue
                source = [walls[wall_id] for wall_id in ids]
                points = [point for wall in source for point in (wall.start_px, wall.end_px)]
                # Usa el par de extremos más distante para conservar la extensión total.
                best = None
                best_distance = -1.0
                for i, first in enumerate(points):
                    for second in points[i + 1:]:
                        distance = math.hypot(second[0] - first[0], second[1] - first[1])
                        if distance > best_distance:
                            best_distance = distance
                            best = (first, second)
                if best is None:
                    continue
                target_id = delta.wall_id or ids[0]
                merged = source[0].model_copy(update={
                    "id": target_id,
                    "start_px": best[0],
                    "end_px": best[1],
                    "thickness_px": sum(wall.thickness_px for wall in source) / len(source),
                    "confidence": max(delta.confidence, max(wall.confidence for wall in source)),
                    "source_candidate_ids": sorted({item for wall in source for item in wall.source_candidate_ids}),
                    "evidence_ids": sorted({item for wall in source for item in wall.evidence_ids}),
                    "source_names": sorted({item for wall in source for item in wall.source_names} | {"GEMINI_CALL2"}),
                    "context_state": "CALL2_MERGED",
                })
                for wall_id in ids:
                    walls.pop(wall_id, None)
                walls[target_id] = merged
                continue

            if delta.action == "SPLIT_WALL":
                if not delta.wall_id or delta.wall_id not in walls or delta.new_point_px is None:
                    continue
                wall = walls.pop(delta.wall_id)
                first_id = f"{delta.wall_id}__A"
                second_id = f"{delta.wall_id}__B"
                walls[first_id] = wall.model_copy(update={
                    "id": first_id,
                    "end_px": delta.new_point_px,
                    "confidence": max(wall.confidence, delta.confidence),
                    "context_state": "CALL2_SPLIT",
                })
                walls[second_id] = wall.model_copy(update={
                    "id": second_id,
                    "start_px": delta.new_point_px,
                    "confidence": max(wall.confidence, delta.confidence),
                    "context_state": "CALL2_SPLIT",
                })

        return graph.model_copy(update={
            "walls": sorted(walls.values(), key=lambda item: item.id),
        }, deep=True)
