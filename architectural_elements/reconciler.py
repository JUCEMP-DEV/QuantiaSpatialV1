from __future__ import annotations

import hashlib

from app.quantia_spatialV1.parametric_model import ParametricOpening, QuantiaParametricModel

from .contracts import OpeningElementProposal


class OpeningReconciler:
    """Publica openings sobre el modelo Spatial sin modificar sus muros."""

    def merge(
        self,
        *,
        model: QuantiaParametricModel,
        level_view_id: str,
        proposals: list[OpeningElementProposal],
    ) -> tuple[QuantiaParametricModel, int]:
        levels = list(model.levels)
        level_index = next(
            (idx for idx, level in enumerate(levels) if level.source_level_view_id == level_view_id or level.id == level_view_id),
            None,
        )
        if level_index is None:
            raise ValueError(f"No existe nivel {level_view_id!r} en QuantiaParametricModel.")

        level = levels[level_index]
        wall_by_id = {wall.id: wall for wall in level.walls}
        accepted = [
            item for item in proposals
            if item.decision == "ACCEPTED"
            and item.predicted_class in {"DOOR", "WINDOW"}
            and item.host_wall.wall_id in wall_by_id
            and item.host_wall.projected_offset_px is not None
        ]
        accepted.sort(key=lambda item: (-item.confidence, item.id))

        published: list[ParametricOpening] = list(level.openings)
        new_count = 0
        for proposal in accepted:
            wall = wall_by_id[proposal.host_wall.wall_id]
            wall_len = float(wall.length_px)
            width_px = min(float(proposal.estimated_width_px), wall_len * 0.92)
            if width_px <= 1.0:
                continue
            offset = min(max(float(proposal.host_wall.projected_offset_px), 0.0), wall_len)
            if self._is_duplicate(
                openings=published,
                host_wall_id=wall.id,
                opening_type=proposal.predicted_class,
                offset_px=offset,
                width_px=width_px,
            ):
                continue

            width_m = None
            if wall.length_m is not None and wall.length_px > 0.0:
                width_m = width_px * (float(wall.length_m) / float(wall.length_px))
            digest = hashlib.sha1(
                f"{level.id}|{proposal.id}|{wall.id}|{offset:.4f}|{width_px:.4f}".encode()
            ).hexdigest()[:14]
            published.append(
                ParametricOpening(
                    id=f"{level.id}__OPENING__{digest}",
                    level_id=level.id,
                    host_wall_id=wall.id,
                    opening_type=proposal.predicted_class,
                    offset_px=offset,
                    width_px=width_px,
                    width_m=width_m,
                    evidence_ids=list(proposal.evidence_ids),
                )
            )
            new_count += 1

        if new_count <= 0:
            return model.model_copy(deep=True), 0

        levels[level_index] = level.model_copy(update={"openings": published})
        updated = model.model_copy(
            update={
                "levels": levels,
                "model_revision": model.model_revision + 1,
            },
            deep=True,
        )
        return updated, new_count

    @staticmethod
    def _is_duplicate(
        *,
        openings: list[ParametricOpening],
        host_wall_id: str,
        opening_type: str,
        offset_px: float,
        width_px: float,
    ) -> bool:
        for item in openings:
            if item.host_wall_id != host_wall_id or item.opening_type != opening_type:
                continue
            tolerance = max(4.0, 0.35 * max(width_px, float(item.width_px)))
            if abs(float(item.offset_px) - offset_px) <= tolerance:
                return True
        return False
