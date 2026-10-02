from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from app.quantia_spatialV1.adaptive_reconstruction.contracts import SingleLineWall

from .contracts import CanonicalWallMapping


class WallHypothesisCanonicalizer:
    """Publica una sola centerline por muro fisico sin reconstruir el plano.

    V3 usa tres pasos independientes y seleccionables:
    1) hipotesis estrictamente duplicadas -> una representacion;
    2) bundles casi coincidentes -> una centerline aun si discrepa el rol topologico;
    3) dos caras paralelas compatibles -> centerline entre ambas.

    No conecta fragmentos separados, no cierra openings y no crea muros nuevos.
    """

    def canonicalize(
        self,
        *,
        walls: Sequence[SingleLineWall],
        px_per_m: float,
        enable_duplicates: bool = True,
        enable_near_coincident: bool = True,
        enable_wall_faces: bool = True,
    ) -> tuple[list[SingleLineWall], list[CanonicalWallMapping]]:
        walls = list(walls)
        if not walls:
            return [], []

        if enable_duplicates:
            duplicate_walls, duplicate_sources = self._collapse_duplicate_hypotheses(walls, px_per_m=px_per_m)
        else:
            duplicate_walls = list(walls)
            duplicate_sources = {wall.id: {wall.id} for wall in walls}

        if enable_near_coincident:
            near_walls, near_sources = self._collapse_near_coincident_bundles(
                duplicate_walls,
                source_map=duplicate_sources,
                px_per_m=px_per_m,
            )
        else:
            near_walls = list(duplicate_walls)
            near_sources = {wall.id: set(duplicate_sources.get(wall.id, {wall.id})) for wall in duplicate_walls}

        if enable_wall_faces:
            face_walls, face_sources = self._collapse_wall_faces(
                near_walls,
                duplicate_sources=near_sources,
                px_per_m=px_per_m,
            )
        else:
            face_walls = list(near_walls)
            face_sources = {wall.id: set(near_sources.get(wall.id, {wall.id})) for wall in near_walls}

        mappings: list[CanonicalWallMapping] = []
        for wall in face_walls:
            source_ids = sorted(face_sources.get(wall.id, {wall.id}))
            source_candidates = sorted({
                cid
                for original in walls
                if original.id in source_ids
                for cid in original.source_candidate_ids
            })
            reason = "single_hypothesis"
            topology = dict(wall.topology_support or {})
            if topology.get("canonicalization_reason"):
                reason = str(topology["canonicalization_reason"])
            elif len(source_ids) > 1:
                reason = "parallel_overlap_consensus"
            mappings.append(
                CanonicalWallMapping(
                    canonical_wall_id=wall.id,
                    source_wall_ids=source_ids,
                    source_candidate_ids=source_candidates,
                    reason=reason,
                )
            )

        face_walls.sort(
            key=lambda wall: (
                round(min(wall.start_px[1], wall.end_px[1]), 3),
                round(min(wall.start_px[0], wall.end_px[0]), 3),
                wall.id,
            )
        )
        mappings.sort(key=lambda item: item.canonical_wall_id)
        return face_walls, mappings

    def _collapse_duplicate_hypotheses(
        self,
        walls: Sequence[SingleLineWall],
        *,
        px_per_m: float,
    ) -> tuple[list[SingleLineWall], dict[str, set[str]]]:
        remaining = list(sorted(walls, key=self._length, reverse=True))
        output: list[SingleLineWall] = []
        sources: dict[str, set[str]] = {}
        canonical_index = 1

        while remaining:
            reference = remaining.pop(0)
            group = [reference]
            keep: list[SingleLineWall] = []
            for item in remaining:
                if self._same_physical_wall(reference, item, px_per_m=px_per_m):
                    group.append(item)
                else:
                    keep.append(item)
            remaining = keep

            if len(group) == 1:
                canonical = reference
            else:
                canonical = self._merge_consensus(
                    group,
                    canonical_index=canonical_index,
                    reason="near_duplicate_hypotheses",
                )
                canonical_index += 1
            output.append(canonical)
            sources[canonical.id] = {item.id for item in group}
        return output, sources

    def _collapse_near_coincident_bundles(
        self,
        walls: Sequence[SingleLineWall],
        *,
        source_map: dict[str, set[str]],
        px_per_m: float,
    ) -> tuple[list[SingleLineWall], dict[str, set[str]]]:
        remaining = list(sorted(walls, key=self._length, reverse=True))
        output: list[SingleLineWall] = []
        sources: dict[str, set[str]] = {}
        canonical_index = 1

        while remaining:
            reference = remaining.pop(0)
            group = [reference]
            keep: list[SingleLineWall] = []
            for item in remaining:
                if self._near_coincident_metrics(reference, item, px_per_m=px_per_m) is not None:
                    group.append(item)
                else:
                    keep.append(item)
            remaining = keep

            if len(group) == 1:
                canonical = reference
            else:
                canonical = self._merge_centerline_bundle(
                    group,
                    canonical_index=canonical_index,
                )
                canonical_index += 1
            output.append(canonical)
            merged_sources: set[str] = set()
            for item in group:
                merged_sources.update(source_map.get(item.id, {item.id}))
            sources[canonical.id] = merged_sources
        return output, sources

    def _collapse_wall_faces(
        self,
        walls: Sequence[SingleLineWall],
        *,
        duplicate_sources: dict[str, set[str]],
        px_per_m: float,
    ) -> tuple[list[SingleLineWall], dict[str, set[str]]]:
        candidates: list[tuple[float, int, int, float]] = []
        walls = list(walls)
        for i, first in enumerate(walls):
            for j in range(i + 1, len(walls)):
                second = walls[j]
                metrics = self._face_pair_metrics(first, second, px_per_m=px_per_m)
                if metrics is None:
                    continue
                overlap, distance, length_similarity = metrics
                distance_m = distance / px_per_m
                score = 2.0 * overlap + length_similarity - 0.30 * abs(distance_m - 0.15)
                candidates.append((score, i, j, distance))
        candidates.sort(reverse=True)

        used: set[int] = set()
        output: list[SingleLineWall] = []
        sources: dict[str, set[str]] = {}
        pair_index = 1
        for _, i, j, distance in candidates:
            if i in used or j in used:
                continue
            first, second = walls[i], walls[j]
            canonical = self._merge_face_pair(
                first,
                second,
                separation_px=distance,
                canonical_index=pair_index,
            )
            pair_index += 1
            used.update({i, j})
            output.append(canonical)
            sources[canonical.id] = set(duplicate_sources.get(first.id, {first.id})) | set(
                duplicate_sources.get(second.id, {second.id})
            )

        for index, wall in enumerate(walls):
            if index in used:
                continue
            output.append(wall)
            sources[wall.id] = set(duplicate_sources.get(wall.id, {wall.id}))
        return output, sources

    @classmethod
    def _same_physical_wall(cls, first: SingleLineWall, second: SingleLineWall, *, px_per_m: float) -> bool:
        if cls._angle_diff(cls._angle(first), cls._angle(second)) > 3.0:
            return False
        ux, uy = cls._axis(first)
        a0, a1 = cls._interval(first, ux, uy)
        b0, b1 = cls._interval(second, ux, uy)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1e-6, min(a1 - a0, b1 - b0))
        if overlap / denom < 0.60:
            return False

        distance = abs(cls._offset(first, ux, uy) - cls._offset(second, ux, uy))
        min_thickness = max(1.0, min(first.thickness_px, second.thickness_px))
        if distance > max(0.06 * px_per_m, 0.65 * min_thickness):
            return False
        if first.role != "REVIEW" and second.role != "REVIEW" and first.role != second.role:
            return False
        return True

    @classmethod
    def _near_coincident_metrics(
        cls,
        first: SingleLineWall,
        second: SingleLineWall,
        *,
        px_per_m: float,
    ) -> tuple[float, float, float] | None:
        if cls._angle_diff(cls._angle(first), cls._angle(second)) > 3.0:
            return None
        ux, uy = cls._axis(first)
        a0, a1 = cls._interval(first, ux, uy)
        b0, b1 = cls._interval(second, ux, uy)
        len_a = max(1e-6, a1 - a0)
        len_b = max(1e-6, b1 - b0)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        overlap_ratio = overlap / min(len_a, len_b)
        length_similarity = min(len_a, len_b) / max(len_a, len_b)
        if overlap_ratio < 0.75 or length_similarity < 0.55:
            return None
        distance = abs(cls._offset(first, ux, uy) - cls._offset(second, ux, uy))
        if distance / max(px_per_m, 1e-6) > 0.09:
            return None
        return overlap_ratio, distance, length_similarity

    @classmethod
    def _face_pair_metrics(
        cls,
        first: SingleLineWall,
        second: SingleLineWall,
        *,
        px_per_m: float,
    ) -> tuple[float, float, float] | None:
        if cls._angle_diff(cls._angle(first), cls._angle(second)) > 2.5:
            return None
        if first.role != "REVIEW" and second.role != "REVIEW" and first.role != second.role:
            return None

        ux, uy = cls._axis(first)
        a0, a1 = cls._interval(first, ux, uy)
        b0, b1 = cls._interval(second, ux, uy)
        len_a = max(1e-6, a1 - a0)
        len_b = max(1e-6, b1 - b0)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        overlap_ratio = overlap / min(len_a, len_b)
        length_similarity = min(len_a, len_b) / max(len_a, len_b)
        if overlap_ratio < 0.82 or length_similarity < 0.78:
            return None

        distance = abs(cls._offset(first, ux, uy) - cls._offset(second, ux, uy))
        distance_m = distance / max(px_per_m, 1e-6)
        if not 0.065 <= distance_m <= 0.30:
            return None

        thin_a = first.thickness_px / px_per_m <= 0.10
        thin_b = second.thickness_px / px_per_m <= 0.10
        if not (thin_a or thin_b):
            return None
        return overlap_ratio, distance, length_similarity

    @classmethod
    def _merge_consensus(
        cls,
        items: Sequence[SingleLineWall],
        *,
        canonical_index: int,
        reason: str,
    ) -> SingleLineWall:
        reference = max(items, key=cls._length)
        ux, uy = cls._axis(reference)
        nx, ny = -uy, ux
        weighted = [
            (
                item,
                max(0.05, item.confidence)
                * cls._length(item)
                * (1.35 if item.f03_seed_protected else 1.0),
            )
            for item in items
        ]
        offsets = [(cls._offset(item, ux, uy), weight) for item, weight in weighted]
        starts: list[tuple[float, float]] = []
        ends: list[tuple[float, float]] = []
        for item, weight in weighted:
            lo, hi = cls._interval(item, ux, uy)
            starts.append((lo, weight))
            ends.append((hi, weight))

        offset = cls._weighted_median(offsets)
        lo = cls._weighted_median(starts)
        hi = cls._weighted_median(ends)
        start = (lo * ux + offset * nx, lo * uy + offset * ny)
        end = (hi * ux + offset * nx, hi * uy + offset * ny)
        thickness = cls._weighted_average([(item.thickness_px, weight) for item, weight in weighted])
        return cls._wall_from_group(
            items,
            wall_id=cls._canonical_id("CW", canonical_index, items),
            start=start,
            end=end,
            thickness=max(1.0, float(thickness)),
            reason=reason,
        )

    @classmethod
    def _merge_centerline_bundle(
        cls,
        items: Sequence[SingleLineWall],
        *,
        canonical_index: int,
    ) -> SingleLineWall:
        reference = max(items, key=cls._length)
        ux, uy = cls._axis(reference)
        nx, ny = -uy, ux
        offsets = [cls._offset(item, ux, uy) for item in items]
        weighted = [
            (
                item,
                max(0.05, item.confidence)
                * cls._length(item)
                * (1.35 if item.f03_seed_protected else 1.0),
            )
            for item in items
        ]
        starts: list[tuple[float, float]] = []
        ends: list[tuple[float, float]] = []
        for item, weight in weighted:
            lo, hi = cls._interval(item, ux, uy)
            starts.append((lo, weight))
            ends.append((hi, weight))
        offset = (min(offsets) + max(offsets)) * 0.5
        lo = cls._weighted_median(starts)
        hi = cls._weighted_median(ends)
        start = (lo * ux + offset * nx, lo * uy + offset * ny)
        end = (hi * ux + offset * nx, hi * uy + offset * ny)
        bundle_span = max(offsets) - min(offsets)
        thickness = max(bundle_span, max(item.thickness_px for item in items))
        return cls._wall_from_group(
            items,
            wall_id=cls._canonical_id("CB", canonical_index, items),
            start=start,
            end=end,
            thickness=max(1.0, float(thickness)),
            reason="near_coincident_centerline_bundle",
        )

    @classmethod
    def _merge_face_pair(
        cls,
        first: SingleLineWall,
        second: SingleLineWall,
        *,
        separation_px: float,
        canonical_index: int,
    ) -> SingleLineWall:
        items = [first, second]
        reference = max(items, key=cls._length)
        ux, uy = cls._axis(reference)
        nx, ny = -uy, ux
        offsets = [cls._offset(item, ux, uy) for item in items]
        offset = sum(offsets) / 2.0
        intervals = [cls._interval(item, ux, uy) for item in items]
        lo = sum(pair[0] for pair in intervals) / len(intervals)
        hi = sum(pair[1] for pair in intervals) / len(intervals)
        start = (lo * ux + offset * nx, lo * uy + offset * ny)
        end = (hi * ux + offset * nx, hi * uy + offset * ny)
        inferred_thickness = max(first.thickness_px, second.thickness_px, separation_px)
        return cls._wall_from_group(
            items,
            wall_id=cls._canonical_id("CF", canonical_index, items),
            start=start,
            end=end,
            thickness=max(1.0, float(inferred_thickness)),
            reason="parallel_wall_faces_to_centerline",
        )

    @classmethod
    def _wall_from_group(
        cls,
        items: Sequence[SingleLineWall],
        *,
        wall_id: str,
        start: tuple[float, float],
        end: tuple[float, float],
        thickness: float,
        reason: str,
    ) -> SingleLineWall:
        confidence = max(item.confidence for item in items)
        topology_support = {
            "canonicalized_from_wall_ids": sorted(item.id for item in items),
            "hypothesis_count": len(items),
            "canonicalization_reason": reason,
            "source_topology": [item.topology_support for item in items],
        }
        return SingleLineWall(
            id=wall_id,
            start_px=start,
            end_px=end,
            thickness_px=thickness,
            role=cls._role(items),
            confidence=max(0.0, min(1.0, confidence)),
            source_candidate_ids=sorted({cid for item in items for cid in item.source_candidate_ids}),
            evidence_ids=sorted({eid for item in items for eid in item.evidence_ids}),
            source_names=sorted({name for item in items for name in item.source_names}),
            f03_seed_protected=any(item.f03_seed_protected for item in items),
            context_state="POST_FILTER_CANONICAL",
            topology_support=topology_support,
        )

    @staticmethod
    def _canonical_id(prefix: str, index: int, items: Sequence[SingleLineWall]) -> str:
        ident = hashlib.sha1("|".join(sorted(item.id for item in items)).encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{index:03d}_{ident}"

    @staticmethod
    def _length(wall: SingleLineWall) -> float:
        return math.hypot(wall.end_px[0] - wall.start_px[0], wall.end_px[1] - wall.start_px[1])

    @staticmethod
    def _angle(wall: SingleLineWall) -> float:
        return math.degrees(math.atan2(wall.end_px[1] - wall.start_px[1], wall.end_px[0] - wall.start_px[0])) % 180.0

    @classmethod
    def _axis(cls, wall: SingleLineWall) -> tuple[float, float]:
        angle = math.radians(cls._angle(wall))
        return math.cos(angle), math.sin(angle)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _interval(wall: SingleLineWall, ux: float, uy: float) -> tuple[float, float]:
        values = [
            wall.start_px[0] * ux + wall.start_px[1] * uy,
            wall.end_px[0] * ux + wall.end_px[1] * uy,
        ]
        return min(values), max(values)

    @staticmethod
    def _offset(wall: SingleLineWall, ux: float, uy: float) -> float:
        nx, ny = -uy, ux
        mx = (wall.start_px[0] + wall.end_px[0]) * 0.5
        my = (wall.start_px[1] + wall.end_px[1]) * 0.5
        return mx * nx + my * ny

    @staticmethod
    def _weighted_average(values: Sequence[tuple[float, float]]) -> float:
        total = sum(weight for _, weight in values)
        if total <= 1e-9:
            return sum(value for value, _ in values) / max(len(values), 1)
        return sum(value * weight for value, weight in values) / total

    @staticmethod
    def _weighted_median(values: Sequence[tuple[float, float]]) -> float:
        ordered = sorted((float(value), max(0.0, float(weight))) for value, weight in values)
        total = sum(weight for _, weight in ordered)
        if total <= 1e-9:
            return ordered[len(ordered) // 2][0]
        threshold = total * 0.5
        acc = 0.0
        for value, weight in ordered:
            acc += weight
            if acc >= threshold:
                return value
        return ordered[-1][0]

    @staticmethod
    def _role(items: Sequence[SingleLineWall]) -> str:
        weighted = {"PERIMETER": 0.0, "DIVIDER": 0.0, "REVIEW": 0.0}
        for item in items:
            weight = max(0.05, item.confidence) * (1.25 if item.f03_seed_protected else 1.0)
            weighted[item.role] += weight
        non_review = max(("PERIMETER", "DIVIDER"), key=lambda key: weighted[key])
        if weighted[non_review] > weighted["REVIEW"] * 0.75 and weighted[non_review] > 0:
            return non_review
        return "REVIEW"
