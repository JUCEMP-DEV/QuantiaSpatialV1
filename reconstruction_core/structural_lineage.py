from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from shapely.geometry import LineString, Point

from .candidate_models import WallCandidate


@dataclass(frozen=True)
class StructuralLineage:
    candidate_id: str
    axis_support: float
    collinear_anchor_support: float
    junction_anchor_support: float
    long_segment_support: float
    score: float
    strong: bool

    def as_metadata(self) -> dict:
        return {
            "axis_support": self.axis_support,
            "collinear_anchor_support": self.collinear_anchor_support,
            "junction_anchor_support": self.junction_anchor_support,
            "long_segment_support": self.long_segment_support,
            "score": self.score,
            "strong": self.strong,
        }


class StructuralLineageAnalyzer:
    """Evalúa linaje estructural sin permitir autoconfirmación contextual.

    Selection Contracts V1 separa dos preguntas:
    1) ¿el candidato cayó dentro de un patrón repetitivo?, y
    2) ¿existe una ancla estructural *independiente* que justifique rescatarlo?

    Un candidato REVIEW/QUARANTINE nunca puede servir como ancla de otro. Dos
    miembros del mismo contexto repetitivo tampoco se rescatan mutuamente, salvo
    una continuación claramente más larga que atraviese y salga de la región.
    El eje por sí solo no crea linaje fuerte.
    """

    VERSION = "STRUCTURAL_LINEAGE_SELECTION_CONTRACTS_V1"
    ANGLE_TOLERANCE_DEG = 6.0

    def analyze(
        self,
        candidates: Sequence[WallCandidate],
        *,
        decisions: Mapping[str, object] | None = None,
    ) -> dict[str, StructuralLineage]:
        candidates = list(candidates)
        if not candidates:
            return {}
        median_length = float(median([item.length_px for item in candidates]))
        median_thickness = float(median([item.thickness_px for item in candidates]))
        output: dict[str, StructuralLineage] = {}

        for candidate in candidates:
            collinear = 0.0
            junction = 0.0
            for other in candidates:
                if other.id == candidate.id:
                    continue
                if not self._anchor_allowed(
                    candidate=candidate,
                    other=other,
                    decisions=decisions,
                ):
                    continue

                if self._angle_diff(candidate.angle_deg, other.angle_deg) <= self.ANGLE_TOLERANCE_DEG:
                    collinear = max(
                        collinear,
                        self._collinear_anchor_support(
                            candidate=candidate,
                            other=other,
                            median_thickness=median_thickness,
                        ),
                    )
                else:
                    junction = max(
                        junction,
                        self._junction_anchor_support(
                            candidate=candidate,
                            other=other,
                            median_thickness=median_thickness,
                        ),
                    )

            axis = max(0.0, min(1.0, float(candidate.evidence.axis_support)))
            long_segment = max(0.0, min(1.0, candidate.length_px / max(2.4 * median_length, 1e-6)))
            score = max(
                0.78 * collinear + 0.12 * axis + 0.10 * long_segment,
                0.72 * junction + 0.18 * axis + 0.10 * long_segment,
                0.45 * axis + 0.15 * long_segment,
            )
            score = max(0.0, min(1.0, score))
            anchor_support = max(collinear, junction)
            output[candidate.id] = StructuralLineage(
                candidate_id=candidate.id,
                axis_support=axis,
                collinear_anchor_support=collinear,
                junction_anchor_support=junction,
                long_segment_support=long_segment,
                score=score,
                strong=score >= 0.68 and anchor_support >= 0.55,
            )
        return output

    def _anchor_allowed(
        self,
        *,
        candidate: WallCandidate,
        other: WallCandidate,
        decisions: Mapping[str, object] | None,
    ) -> bool:
        # Sin contexto (uso unitario directo), la independencia se deriva de una
        # diferencia real de jerarquía geométrica; líneas similares no se anclan.
        if decisions is None:
            return other.length_px >= 1.45 * candidate.length_px

        candidate_decision = decisions.get(candidate.id)
        other_decision = decisions.get(other.id)
        if other_decision is None:
            return False

        other_state = str(getattr(other_decision, "state", ""))
        if other_state in {"QUARANTINE", "REVIEW"}:
            return False
        if other_state == "PROTECTED":
            return True

        candidate_region = getattr(candidate_decision, "region_id", None) if candidate_decision is not None else None
        other_region = getattr(other_decision, "region_id", None)
        if candidate_region and other_region and candidate_region == other_region:
            # Única excepción: una ancla mucho más larga puede cruzar la misma bbox
            # contextual siempre que la mayor parte de su identidad NO derive del
            # patrón. Esto conserva muros que atraviesan acabados sin permitir que
            # los propios módulos se rescaten entre sí.
            reason = str(getattr(other_decision, "reason", ""))
            containment = float(getattr(other_decision, "spatial_containment", 1.0) or 1.0)
            face_support = float(getattr(other_decision, "face_pattern_support", 0.0) or 0.0)
            clear_external_anchor = (
                other_state == "ACTIVE"
                and containment <= 0.65
                and face_support <= 0.20
                and other.length_px >= 1.80 * candidate.length_px
            )
            explicit_continuation = (
                reason == "CANDIDATE_CONTINUES_OUTSIDE_CONTEXT_REGION"
                and containment <= 0.78
                and other.length_px >= 1.80 * candidate.length_px
            )
            if not (clear_external_anchor or explicit_continuation):
                return False

        # Fuera del mismo patrón, el otro candidato aún debe tener jerarquía
        # geométrica suficiente. axis_support solo mejora una ancla ya existente.
        return other.length_px >= 1.45 * candidate.length_px

    def _collinear_anchor_support(
        self,
        *,
        candidate: WallCandidate,
        other: WallCandidate,
        median_thickness: float,
    ) -> float:
        length_ratio = other.length_px / max(candidate.length_px, 1e-6)
        if length_ratio < 1.45:
            return 0.0
        length_strength = max(0.0, min(1.0, (length_ratio - 1.25) / 1.75))
        anchor_strength = min(
            1.0,
            0.82 * length_strength + 0.18 * float(other.evidence.axis_support),
        )
        if anchor_strength < 0.40:
            return 0.0

        a = self._line(candidate)
        b = self._line(other)
        line_distance = a.distance(b)
        tolerance = max(2.0, 0.35 * median_thickness)
        if line_distance > tolerance:
            return 0.0

        gap = self._projection_gap(candidate, other)
        allowed_gap = max(3.0 * median_thickness, 0.22 * min(candidate.length_px, other.length_px))
        if gap > allowed_gap:
            return 0.0
        gap_score = max(0.0, 1.0 - gap / max(allowed_gap, 1e-6))
        distance_score = max(0.0, 1.0 - line_distance / max(tolerance, 1e-6))
        return max(0.0, min(1.0, anchor_strength * (0.55 * gap_score + 0.45 * distance_score)))

    def _junction_anchor_support(
        self,
        *,
        candidate: WallCandidate,
        other: WallCandidate,
        median_thickness: float,
    ) -> float:
        diff = self._angle_diff(candidate.angle_deg, other.angle_deg)
        if diff < 45.0 or diff > 135.0:
            return 0.0
        # Una intersección de patrón (escalón/baldosa) es demasiado común para
        # rescatar sin apoyo adicional del propio candidato.
        if float(candidate.evidence.axis_support) < 0.50:
            return 0.0

        length_ratio = other.length_px / max(candidate.length_px, 1e-6)
        if length_ratio < 1.50:
            return 0.0
        length_strength = max(0.0, min(1.0, (length_ratio - 1.30) / 1.70))
        anchor_strength = min(
            1.0,
            0.82 * length_strength + 0.18 * float(other.evidence.axis_support),
        )
        if anchor_strength < 0.42:
            return 0.0

        other_line = self._line(other)
        endpoints = [
            Point(candidate.start.x, candidate.start.y),
            Point(candidate.end.x, candidate.end.y),
        ]
        distance = min(other_line.distance(point) for point in endpoints)
        tolerance = max(3.0, 1.20 * max(candidate.thickness_px, other.thickness_px, median_thickness))
        if distance > tolerance:
            return 0.0
        distance_score = max(0.0, 1.0 - distance / max(tolerance, 1e-6))
        return max(0.0, min(1.0, anchor_strength * distance_score))

    @staticmethod
    def _projection_gap(first: WallCandidate, second: WallCandidate) -> float:
        angle = math.radians(first.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        first_values = [first.start.x * ux + first.start.y * uy, first.end.x * ux + first.end.y * uy]
        second_values = [second.start.x * ux + second.start.y * uy, second.end.x * ux + second.end.y * uy]
        a0, a1 = sorted(first_values)
        b0, b1 = sorted(second_values)
        if a1 >= b0 and b1 >= a0:
            return 0.0
        return max(b0 - a1, a0 - b1, 0.0)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _line(candidate: WallCandidate) -> LineString:
        return LineString([
            (candidate.start.x, candidate.start.y),
            (candidate.end.x, candidate.end.y),
        ])
