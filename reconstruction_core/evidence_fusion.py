from __future__ import annotations

from .candidate_models import WallEvidenceVector


class WallEvidenceFusion:
    """Convierte evidencias heterogéneas en un prior, no en aceptación final."""

    def prior(
        self,
        evidence: WallEvidenceVector,
        *,
        generator: str,
        independent_source_count: int,
        length_support: float,
    ) -> float:
        if generator == "DOUBLE_FACE":
            score = (
                0.20 * evidence.pair_overlap
                + 0.16 * evidence.thickness_support
                + 0.12 * evidence.vector_support
                + 0.10 * evidence.raster_line_support
                + 0.16 * evidence.region_support
                + 0.12 * evidence.source_consensus
                + 0.12 * evidence.perimeter_containment
                + 0.02 * evidence.semantic_support
                + 0.04 * evidence.axis_support
                - 0.20 * evidence.dashed_penalty
            )
            if independent_source_count <= 1:
                score *= 0.60
            elif independent_source_count == 2:
                score *= 0.90
            score *= 0.68 + 0.32 * max(0.0, min(1.0, length_support))
        else:
            score = (
                0.30 * evidence.region_support
                + 0.15 * evidence.thickness_support
                + 0.12 * evidence.vector_support
                + 0.12 * evidence.raster_line_support
                + 0.16 * evidence.source_consensus
                + 0.10 * evidence.perimeter_containment
                + 0.02 * evidence.axis_support
            )
            if independent_source_count <= 1:
                score *= 0.58
            elif independent_source_count == 2:
                score *= 0.88
            score *= 0.66 + 0.34 * max(0.0, min(1.0, length_support))
        return max(0.0, min(1.0, float(score)))
