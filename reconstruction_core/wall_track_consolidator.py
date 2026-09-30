from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Sequence

from shapely.geometry import LineString

from .candidate_models import WallCandidate, WallEvidenceVector
from .drawing_model import DrawingPoint


class WallTrackConsolidator:
    """Une fragmentos colineales en hipótesis de muro continuas.

    No decide aceptación. La salida sigue siendo WallCandidate y será evaluada
    globalmente. La unión solo ocurre cuando los fragmentos son compatibles y
    no se solapan ampliamente (evita colapsar alternativas paralelas). La cantidad
    de fragmentos no aumenta la confianza del track.
    """

    def consolidate(
        self,
        *,
        candidates: Sequence[WallCandidate],
        width_px: int,
        height_px: int,
    ) -> list[WallCandidate]:
        if len(candidates) < 2:
            return list(candidates)

        median_t = self._median([item.thickness_px for item in candidates]) or 4.0
        min_dim = min(width_px, height_px)
        coordinate_tol = max(2.0, 0.85 * median_t)
        gap_tol = max(4.0, 1.8 * median_t, min_dim * 0.010)

        adjacency: dict[int, set[int]] = {index: set() for index in range(len(candidates))}
        for i, first in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                second = candidates[j]
                if self._angle_diff(first.angle_deg, second.angle_deg) > 3.0:
                    continue
                if self._line_distance(first, second) > coordinate_tol:
                    continue
                overlap = self._overlap_ratio(first, second)
                # Solape amplio significa hipótesis alternativas, no fragmentos.
                if overlap > 0.30:
                    continue
                gap = self._projection_gap(first, second)
                if gap > gap_tol:
                    continue
                ratio = max(first.thickness_px, second.thickness_px) / max(
                    min(first.thickness_px, second.thickness_px), 1e-6
                )
                if ratio > 3.0:
                    continue
                adjacency[i].add(j)
                adjacency[j].add(i)

        remaining = set(range(len(candidates)))
        output: list[WallCandidate] = []
        while remaining:
            seed = next(iter(remaining))
            queue = deque([seed])
            component: list[int] = []
            while queue:
                index = queue.popleft()
                if index not in remaining:
                    continue
                remaining.remove(index)
                component.append(index)
                queue.extend(adjacency[index])
            items = [candidates[index] for index in component]
            if len(items) == 1:
                output.append(items[0])
            else:
                output.append(self._merge(items))

        return sorted(output, key=lambda item: (-item.prior_score, -item.length_px))

    def _merge(self, items: Sequence[WallCandidate]) -> WallCandidate:
        reference = max(items, key=lambda item: item.length_px)
        angle = math.radians(reference.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux

        t_values: list[float] = []
        offsets: list[float] = []
        total_weight = 0.0
        thickness_sum = 0.0
        for item in items:
            for point in (item.start, item.end):
                t_values.append(point.x * ux + point.y * uy)
                offsets.append(point.x * nx + point.y * ny)
            total_weight += item.length_px
            thickness_sum += item.thickness_px * item.length_px

        lo, hi = min(t_values), max(t_values)
        offset = sum(offsets) / len(offsets)
        start = DrawingPoint(x=lo * ux + offset * nx, y=lo * uy + offset * ny)
        end = DrawingPoint(x=hi * ux + offset * nx, y=hi * uy + offset * ny)
        length = math.hypot(end.x - start.x, end.y - start.y)
        thickness = thickness_sum / max(total_weight, 1e-6)

        def weighted(name: str) -> float:
            value = sum(getattr(item.evidence, name) * item.length_px for item in items) / max(total_weight, 1e-6)
            return max(0.0, min(1.0, float(value)))

        evidence = WallEvidenceVector(
            pair_overlap=weighted("pair_overlap"),
            thickness_support=weighted("thickness_support"),
            vector_support=max(item.evidence.vector_support for item in items),
            raster_line_support=max(item.evidence.raster_line_support for item in items),
            region_support=weighted("region_support"),
            source_consensus=weighted("source_consensus"),
            perimeter_containment=weighted("perimeter_containment"),
            semantic_support=weighted("semantic_support"),
            axis_support=weighted("axis_support"),
            dashed_penalty=weighted("dashed_penalty"),
        )
        # La cantidad de fragmentos no es evidencia independiente. Una escalera o
        # retícula puede producir muchos trozos colineales; por eso consolidar no
        # añade bonus. El prior del track es el promedio ponderado por longitud.
        prior = sum(item.prior_score * item.length_px for item in items) / max(total_weight, 1e-6)
        prior = max(0.0, min(1.0, float(prior)))
        ident = hashlib.sha1("|".join(sorted(item.id for item in items)).encode()).hexdigest()[:16]
        return WallCandidate(
            id=f"{reference.id.split('__WC_')[0]}__WC_TRACK__{ident}",
            generator="DOUBLE_FACE" if any(item.generator == "DOUBLE_FACE" for item in items) else "REGION_CENTERLINE",
            start=start,
            end=end,
            angle_deg=reference.angle_deg,
            length_px=length,
            thickness_px=max(1.0, thickness),
            face_ids=sorted({face for item in items for face in item.face_ids}),
            evidence_ids=sorted({eid for item in items for eid in item.evidence_ids}),
            source_names=sorted({source for item in items for source in item.source_names}),
            evidence=evidence,
            prior_score=prior,
            metadata={
                "consolidated_from": [item.id for item in items],
                "fragment_count": len(items),
                # Element Context Isolation V4: no perder provenance al unir
                # fragmentos. Cada grupo conserva la identidad de la cara física
                # que originó el candidato antes de la consolidación.
                "face_lineage_groups": self._merge_face_lineage_groups(items),
                "lineage_line_ids": sorted({
                    lineage_id
                    for item in items
                    for lineage_id in self._candidate_lineage_ids(item)
                }),
            },
        )

    @staticmethod
    def _candidate_lineage_ids(candidate: WallCandidate) -> set[str]:
        ids: set[str] = {str(item) for item in candidate.face_ids if item}
        raw = candidate.metadata.get("lineage_line_ids")
        if isinstance(raw, (list, tuple, set)):
            ids.update(str(item) for item in raw if item)
        return ids

    @classmethod
    def _merge_face_lineage_groups(cls, items: Sequence[WallCandidate]) -> list[list[str]]:
        groups: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for item in items:
            raw_groups = item.metadata.get("face_lineage_groups")
            if isinstance(raw_groups, list):
                for raw in raw_groups:
                    if not isinstance(raw, (list, tuple, set)):
                        continue
                    group = tuple(sorted({str(value) for value in raw if value}))
                    if group and group not in seen:
                        seen.add(group)
                        groups.append(list(group))
            elif item.face_ids:
                for face_id in item.face_ids:
                    group = (str(face_id),)
                    if group not in seen:
                        seen.add(group)
                        groups.append([str(face_id)])
        return groups

    @staticmethod
    def _line(candidate: WallCandidate) -> LineString:
        return LineString([(candidate.start.x, candidate.start.y), (candidate.end.x, candidate.end.y)])

    def _line_distance(self, first: WallCandidate, second: WallCandidate) -> float:
        # Para segmentos colineales separados, Shapely.distance devuelve el gap
        # longitudinal. Aquí necesitamos únicamente la separación NORMAL entre
        # sus ejes; el gap se evalúa aparte en _projection_gap().
        angle = math.radians(first.angle_deg)
        nx, ny = -math.sin(angle), math.cos(angle)
        c1 = ((first.start.x + first.end.x) / 2.0) * nx + ((first.start.y + first.end.y) / 2.0) * ny
        c2 = ((second.start.x + second.end.x) / 2.0) * nx + ((second.start.y + second.end.y) / 2.0) * ny
        return abs(c1 - c2)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _axis(candidate: WallCandidate) -> tuple[float, float]:
        angle = math.radians(candidate.angle_deg)
        return math.cos(angle), math.sin(angle)

    def _interval(self, candidate: WallCandidate, ux: float, uy: float) -> tuple[float, float]:
        values = [
            candidate.start.x * ux + candidate.start.y * uy,
            candidate.end.x * ux + candidate.end.y * uy,
        ]
        return min(values), max(values)

    def _overlap_ratio(self, first: WallCandidate, second: WallCandidate) -> float:
        ux, uy = self._axis(first)
        a0, a1 = self._interval(first, ux, uy)
        b0, b1 = self._interval(second, ux, uy)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        return overlap / max(1e-6, min(a1 - a0, b1 - b0))

    def _projection_gap(self, first: WallCandidate, second: WallCandidate) -> float:
        ux, uy = self._axis(first)
        a0, a1 = self._interval(first, ux, uy)
        b0, b1 = self._interval(second, ux, uy)
        if a1 < b0:
            return b0 - a1
        if b1 < a0:
            return a0 - b1
        return 0.0

    @staticmethod
    def _median(values: Sequence[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        return float(ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0)
