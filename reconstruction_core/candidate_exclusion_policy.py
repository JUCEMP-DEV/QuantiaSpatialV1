from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Sequence

from .candidate_models import WallCandidate


@dataclass(frozen=True)
class CandidateExclusionDecision:
    candidate_id: str
    reason: str
    pattern_id: str | None
    pattern_class: str | None
    wall_likeness_score: float
    pattern_confidence: float


@dataclass
class CandidateExclusionResult:
    kept_candidates: list[WallCandidate] = field(default_factory=list)
    excluded_candidates: list[WallCandidate] = field(default_factory=list)
    decisions: list[CandidateExclusionDecision] = field(default_factory=list)
    protected_pattern_candidate_ids: list[str] = field(default_factory=list)
    pattern_candidate_ids: list[str] = field(default_factory=list)


@dataclass
class _Track:
    orientation: str
    candidate_ids: list[str]
    cross: float
    lo: float
    hi: float
    median_length: float
    median_thickness: float


@dataclass
class _PatternFamily:
    pattern_id: str
    orientation: str
    tracks: list[_Track]
    regularity: float
    length_consistency: float
    compactness: float
    gap_ratio: float
    spacing_to_thickness_ratio: float
    confidence: float
    pattern_class: str = "REPETITIVE_PARALLEL_PATTERN"

    @property
    def candidate_ids(self) -> set[str]:
        return {
            candidate_id
            for track in self.tracks
            for candidate_id in track.candidate_ids
        }


class CandidateExclusionPolicy:
    """Detecta patrones repetitivos sin destruir candidatos ambiguos.

    V3 corrige el defecto conceptual de V2: un candidato no puede demostrar que
    es muro usando las mismas evidencias locales que hicieron que naciera como
    candidato. La repetición se convierte en evidencia NEGATIVA para el solver,
    no en una exclusión dura previa al CandidateGraph.

    Flujo:
        candidato normalizado
        -> detectar familia repetitiva
        -> anotar pattern prior/log-odds penalty
        -> conservar candidato
        -> CandidateGraph
        -> GlobalTopologySolver decide globalmente

    La única verdad publicada por esta clase es geométrica: "pertenece a una
    familia repetitiva". No decide semánticamente si es escalera, hatch o muro.
    """

    RULE_VERSION = "CANDIDATE_PATTERN_CONTEXT_V3_SOFT_PRIOR"

    def apply(
        self,
        *,
        candidates: Sequence[WallCandidate],
        width_px: int,
        height_px: int,
    ) -> CandidateExclusionResult:
        source = list(candidates)
        if len(source) < 3:
            return CandidateExclusionResult(kept_candidates=source)

        local_scores = {
            item.id: self.wall_likeness_score(item)
            for item in source
        }

        tracks = self._build_distinct_tracks(
            candidates=source,
            wall_scores=local_scores,
        )
        families = self._detect_pattern_families(
            tracks=tracks,
            candidates=source,
            width_px=width_px,
            height_px=height_px,
        )
        self._mark_rectangular_families(families)

        pattern_by_candidate: dict[str, _PatternFamily] = {}
        for family in families:
            for candidate_id in family.candidate_ids:
                current = pattern_by_candidate.get(candidate_id)
                if current is None or family.confidence > current.confidence:
                    pattern_by_candidate[candidate_id] = family

        kept: list[WallCandidate] = []
        decisions: list[CandidateExclusionDecision] = []
        patterned: list[str] = []

        for candidate in source:
            family = pattern_by_candidate.get(candidate.id)
            if family is None:
                kept.append(candidate)
                continue

            penalty = self._pattern_logodds_penalty(family)
            metadata = dict(candidate.metadata)
            metadata.update({
                "repetitive_pattern_id": family.pattern_id,
                "repetitive_pattern_class": family.pattern_class,
                "repetitive_pattern_confidence": family.confidence,
                "repetitive_pattern_family_size": len(family.tracks),
                "repetitive_pattern_regularity": family.regularity,
                "repetitive_pattern_length_consistency": family.length_consistency,
                "repetitive_pattern_compactness": family.compactness,
                "repetitive_pattern_gap_ratio": family.gap_ratio,
                "repetitive_pattern_spacing_to_thickness_ratio": family.spacing_to_thickness_ratio,
                "repetitive_pattern_logodds_penalty": penalty,
                "local_observation_score": local_scores[candidate.id],
                "candidate_exclusion_policy": self.RULE_VERSION,
                "candidate_exclusion_state": "PATTERN_PRIOR_ONLY",
            })
            kept.append(candidate.model_copy(update={"metadata": metadata}, deep=True))
            patterned.append(candidate.id)
            decisions.append(
                CandidateExclusionDecision(
                    candidate_id=candidate.id,
                    reason="REPETITIVE_PATTERN_SOFT_PRIOR",
                    pattern_id=family.pattern_id,
                    pattern_class=family.pattern_class,
                    wall_likeness_score=local_scores[candidate.id],
                    pattern_confidence=family.confidence,
                )
            )

        return CandidateExclusionResult(
            kept_candidates=kept,
            excluded_candidates=[],
            decisions=decisions,
            protected_pattern_candidate_ids=[],
            pattern_candidate_ids=sorted(patterned),
        )

    @staticmethod
    def wall_likeness_score(candidate: WallCandidate) -> float:
        """Compatibilidad diagnóstica: calidad de observación local, NO semántica.

        Se conserva el nombre por compatibilidad con reportes anteriores. Este
        valor ya no puede proteger ni excluir candidatos. pair/region/thickness
        describen observabilidad geométrica, no prueban que el objeto sea muro.
        """
        e = candidate.evidence
        if candidate.generator == "DOUBLE_FACE":
            score = (
                0.30 * e.pair_overlap
                + 0.22 * e.thickness_support
                + 0.16 * e.region_support
                + 0.10 * e.source_consensus
                + 0.08 * e.raster_line_support
                + 0.05 * e.vector_support
                + 0.05 * min(
                    1.0,
                    candidate.length_px / max(4.0 * candidate.thickness_px, 1.0),
                )
                - 0.18 * e.dashed_penalty
            )
        else:
            score = (
                0.34 * e.region_support
                + 0.24 * e.thickness_support
                + 0.14 * e.source_consensus
                + 0.10 * e.raster_line_support
                + 0.06 * e.vector_support
                + 0.06 * min(
                    1.0,
                    candidate.length_px / max(5.0 * candidate.thickness_px, 1.0),
                )
                - 0.12 * e.dashed_penalty
            )
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _pattern_logodds_penalty(family: _PatternFamily) -> float:
        """Convierte confianza de patrón a la misma escala log-odds del solver."""
        p = max(0.5001, min(0.995, float(family.confidence)))
        reference = 0.66
        value = math.log(p / (1.0 - p)) - math.log(reference / (1.0 - reference))
        value = max(0.0, value)
        if family.pattern_class == "REPETITIVE_RECTANGULAR_PATTERN":
            value *= 1.18
        return float(min(3.0, value))

    def _build_distinct_tracks(
        self,
        *,
        candidates: Sequence[WallCandidate],
        wall_scores: dict[str, float],
    ) -> list[_Track]:
        """Colapsa hipótesis prácticamente coincidentes antes de medir repetición."""
        oriented = [item for item in candidates if self._orientation(item) in {"H", "V"}]
        if not oriented:
            return []

        thicknesses = [item.thickness_px for item in oriented if item.thickness_px > 0]
        median_thickness = float(median(thicknesses)) if thicknesses else 4.0
        duplicate_offset_tol = max(2.0, 0.45 * median_thickness)

        remaining = set(item.id for item in oriented)
        by_id = {item.id: item for item in oriented}
        tracks: list[_Track] = []

        while remaining:
            seed_id = max(
                remaining,
                key=lambda item_id: (
                    wall_scores.get(item_id, 0.0),
                    by_id[item_id].length_px,
                ),
            )
            seed = by_id[seed_id]
            orientation = self._orientation(seed)
            component = [seed]
            remaining.remove(seed_id)

            changed = True
            while changed:
                changed = False
                for candidate_id in list(remaining):
                    item = by_id[candidate_id]
                    if self._orientation(item) != orientation:
                        continue
                    cross, lo, hi = self._axis_data(item)
                    if min(
                        abs(cross - self._axis_data(member)[0])
                        for member in component
                    ) > duplicate_offset_tol:
                        continue
                    if max(
                        self._interval_overlap_ratio(
                            lo,
                            hi,
                            *self._axis_data(member)[1:],
                        )
                        for member in component
                    ) < 0.65:
                        continue
                    component.append(item)
                    remaining.remove(candidate_id)
                    changed = True

            crosses = [self._axis_data(item)[0] for item in component]
            los = [self._axis_data(item)[1] for item in component]
            his = [self._axis_data(item)[2] for item in component]
            lengths = [item.length_px for item in component]
            local_thickness = [item.thickness_px for item in component]
            tracks.append(
                _Track(
                    orientation=orientation,
                    candidate_ids=sorted(item.id for item in component),
                    cross=float(median(crosses)),
                    lo=float(min(los)),
                    hi=float(max(his)),
                    median_length=float(median(lengths)),
                    median_thickness=float(median(local_thickness)),
                )
            )

        return tracks

    def _detect_pattern_families(
        self,
        *,
        tracks: Sequence[_Track],
        candidates: Sequence[WallCandidate],
        width_px: int,
        height_px: int,
    ) -> list[_PatternFamily]:
        """Detecta modulación repetitiva; 3-4 tracks exigen firma más fuerte."""
        min_dim = max(1.0, float(min(width_px, height_px)))
        candidate_thicknesses = [item.thickness_px for item in candidates if item.thickness_px > 0]
        global_t = float(median(candidate_thicknesses)) if candidate_thicknesses else 4.0
        max_neighbor_offset = max(8.0 * global_t, 0.08 * min_dim)

        families: list[_PatternFamily] = []
        pattern_index = 0
        for orientation in ("H", "V"):
            local = [track for track in tracks if track.orientation == orientation]
            if len(local) < 3:
                continue

            adjacency: dict[int, set[int]] = {index: set() for index in range(len(local))}
            for i, first in enumerate(local):
                for j in range(i + 1, len(local)):
                    second = local[j]
                    offset = abs(first.cross - second.cross)
                    if offset <= max(
                        2.5,
                        0.55 * min(first.median_thickness, second.median_thickness),
                    ):
                        continue
                    if offset > max_neighbor_offset:
                        continue
                    overlap = self._interval_overlap_ratio(
                        first.lo,
                        first.hi,
                        second.lo,
                        second.hi,
                    )
                    length_ratio = min(first.median_length, second.median_length) / max(
                        first.median_length,
                        second.median_length,
                        1e-6,
                    )
                    if overlap < 0.52 or length_ratio < 0.45:
                        continue
                    adjacency[i].add(j)
                    adjacency[j].add(i)

            remaining = set(adjacency)
            while remaining:
                root = remaining.pop()
                stack = [root]
                component = {root}
                while stack:
                    current = stack.pop()
                    for neighbor in adjacency[current]:
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            stack.append(neighbor)

                family_tracks = [local[index] for index in component]
                if len(family_tracks) < 3:
                    continue

                family_tracks.sort(key=lambda item: item.cross)
                coords = [item.cross for item in family_tracks]
                gaps = [coords[index + 1] - coords[index] for index in range(len(coords) - 1)]
                positive_gaps = [gap for gap in gaps if gap > 1e-6]
                if len(positive_gaps) < 2:
                    continue

                med_gap = float(median(positive_gaps))
                if med_gap <= 1e-6:
                    continue
                gap_mad = float(median([abs(gap - med_gap) for gap in positive_gaps]))
                regularity = max(0.0, min(1.0, 1.0 - gap_mad / med_gap))

                lengths = [item.median_length for item in family_tracks]
                med_length = float(median(lengths))
                if med_length <= 1e-6:
                    continue
                length_mad = float(median([abs(value - med_length) for value in lengths]))
                length_consistency = max(0.0, min(1.0, 1.0 - length_mad / med_length))

                cross_span = coords[-1] - coords[0]
                compactness = cross_span / med_length
                gap_ratio = med_gap / med_length
                family_size = len(family_tracks)
                median_track_thickness = float(
                    median([track.median_thickness for track in family_tracks])
                )
                spacing_to_thickness_ratio = med_gap / max(
                    median_track_thickness,
                    1.0,
                )

                # Repetición no equivale a varias hipótesis casi coincidentes del
                # mismo muro. La modulación debe estar separada varias veces el
                # espesor observado; esto distingue treads/retículas de alternativas
                # de centerline o caras de una misma banda.
                if family_size >= 5:
                    qualifies = (
                        regularity >= 0.68
                        and length_consistency >= 0.50
                        and compactness <= 1.10
                        and gap_ratio <= 0.35
                        and spacing_to_thickness_ratio >= 1.50
                    )
                else:
                    qualifies = (
                        regularity >= 0.82
                        and length_consistency >= 0.65
                        and compactness <= 0.70
                        and gap_ratio <= 0.25
                        and spacing_to_thickness_ratio >= 1.75
                    )
                if not qualifies:
                    continue

                size_support = max(0.0, min(1.0, (family_size - 2) / 5.0))
                compactness_support = max(0.0, min(1.0, 1.0 - compactness / 1.10))
                gap_support = max(0.0, min(1.0, 1.0 - gap_ratio / 0.35))
                confidence = (
                    0.38 * regularity
                    + 0.24 * length_consistency
                    + 0.14 * size_support
                    + 0.12 * compactness_support
                    + 0.12 * gap_support
                )
                if confidence < 0.66:
                    continue

                pattern_index += 1
                families.append(
                    _PatternFamily(
                        pattern_id=f"RP{pattern_index:03d}",
                        orientation=orientation,
                        tracks=family_tracks,
                        regularity=regularity,
                        length_consistency=length_consistency,
                        compactness=compactness,
                        gap_ratio=gap_ratio,
                        spacing_to_thickness_ratio=spacing_to_thickness_ratio,
                        confidence=max(0.0, min(1.0, confidence)),
                    )
                )

        return families

    def _mark_rectangular_families(self, families: list[_PatternFamily]) -> None:
        horizontal = [item for item in families if item.orientation == "H"]
        vertical = [item for item in families if item.orientation == "V"]
        for h_family in horizontal:
            h_bbox = self._family_bbox(h_family)
            for v_family in vertical:
                v_bbox = self._family_bbox(v_family)
                if not self._bbox_overlaps(h_bbox, v_bbox):
                    continue
                intersections = 0
                for h_track in h_family.tracks:
                    for v_track in v_family.tracks:
                        if (
                            v_track.cross >= h_track.lo
                            and v_track.cross <= h_track.hi
                            and h_track.cross >= v_track.lo
                            and h_track.cross <= v_track.hi
                        ):
                            intersections += 1
                if intersections >= 4:
                    h_family.pattern_class = "REPETITIVE_RECTANGULAR_PATTERN"
                    v_family.pattern_class = "REPETITIVE_RECTANGULAR_PATTERN"

    @staticmethod
    def _family_bbox(family: _PatternFamily) -> tuple[float, float, float, float]:
        if family.orientation == "H":
            return (
                min(track.lo for track in family.tracks),
                min(track.cross for track in family.tracks),
                max(track.hi for track in family.tracks),
                max(track.cross for track in family.tracks),
            )
        return (
            min(track.cross for track in family.tracks),
            min(track.lo for track in family.tracks),
            max(track.cross for track in family.tracks),
            max(track.hi for track in family.tracks),
        )

    @staticmethod
    def _bbox_overlaps(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> bool:
        ax0, ay0, ax1, ay1 = first
        bx0, by0, bx1, by1 = second
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    @staticmethod
    def _orientation(candidate: WallCandidate) -> str:
        dx = abs(candidate.end.x - candidate.start.x)
        dy = abs(candidate.end.y - candidate.start.y)
        if dx >= 4.0 * max(dy, 1e-6):
            return "H"
        if dy >= 4.0 * max(dx, 1e-6):
            return "V"
        return "D"

    @staticmethod
    def _axis_data(candidate: WallCandidate) -> tuple[float, float, float]:
        orientation = CandidateExclusionPolicy._orientation(candidate)
        if orientation == "H":
            cross = (candidate.start.y + candidate.end.y) / 2.0
            lo = min(candidate.start.x, candidate.end.x)
            hi = max(candidate.start.x, candidate.end.x)
            return float(cross), float(lo), float(hi)
        if orientation == "V":
            cross = (candidate.start.x + candidate.end.x) / 2.0
            lo = min(candidate.start.y, candidate.end.y)
            hi = max(candidate.start.y, candidate.end.y)
            return float(cross), float(lo), float(hi)
        return 0.0, 0.0, float(candidate.length_px)

    @staticmethod
    def _interval_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1e-6, min(a1 - a0, b1 - b0))
        return max(0.0, min(1.0, overlap / denom))
