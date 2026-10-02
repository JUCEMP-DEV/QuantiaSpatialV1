from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence

from app.quantia_spatialV1.adaptive_reconstruction.contracts import SingleLineWall

from .contracts import PatternEvidence


class PostFilterGeometryPatternAnalyzer:
    """Detecta motivos geométricos no-muro sobre el resultado adaptativo.

    La capa trabaja exclusivamente con el WallGraph ya reconstruido. No cambia
    candidatos ni vuelve a ejecutar F03/Mask/Topology.

    V2 es conservadora: un patrón se publica como evidencia solo cuando su firma
    geométrica es suficientemente fuerte. La decisión final sigue en FilterEngine.
    """

    def analyze(self, *, walls: Sequence[SingleLineWall], px_per_m: float) -> list[PatternEvidence]:
        walls = list(walls)
        if not walls:
            return []

        repetitive = self._repetitive_parallel_patterns(walls=walls, px_per_m=px_per_m)
        grid = self._orthogonal_grid_patterns(repetitive=repetitive)
        hubs = self._angular_hub_patterns(walls=walls, px_per_m=px_per_m)

        grid_members = {wid for item in grid for wid in item.wall_ids}
        output = [*grid]
        for item in repetitive:
            if any(wid in grid_members for wid in item.wall_ids):
                continue
            output.append(item)
        output.extend(hubs)
        output.sort(key=lambda item: (item.kind, item.id))
        return output

    def _repetitive_parallel_patterns(
        self,
        *,
        walls: Sequence[SingleLineWall],
        px_per_m: float,
    ) -> list[PatternEvidence]:
        if len(walls) < 5:
            return []

        angle_buckets: dict[int, list[SingleLineWall]] = defaultdict(list)
        for wall in walls:
            length_m = self._length(wall) / px_per_m
            if length_m < 0.10:
                continue
            angle = self._angle(wall)
            bucket = int(round(angle / 3.0)) % 60
            angle_buckets[bucket].append(wall)

        patterns: list[PatternEvidence] = []
        for bucket_walls in angle_buckets.values():
            if len(bucket_walls) < 5:
                continue
            reference = max(bucket_walls, key=self._length)
            ux, uy = self._axis(reference)

            # Separa zonas longitudinalmente antes de buscar periodicidad.
            # Se usa agrupacion directa contra un ancla, no componentes
            # transitivos: una linea larga intermedia no puede unir dos zonas
            # arquitectonicas separadas.
            span_components: list[list[SingleLineWall]] = []
            local_remaining = list(sorted(bucket_walls, key=self._length, reverse=True))
            while local_remaining:
                seed = local_remaining.pop(0)
                a0, a1 = self._interval(seed, ux, uy)
                len_a = max(1e-6, a1 - a0)
                center_a = (a0 + a1) * 0.5
                group = [seed]
                keep: list[SingleLineWall] = []
                for second in local_remaining:
                    b0, b1 = self._interval(second, ux, uy)
                    len_b = max(1e-6, b1 - b0)
                    overlap = max(0.0, min(a1, b1) - max(a0, b0))
                    center_b = (b0 + b1) * 0.5
                    direct_match = (
                        overlap / min(len_a, len_b) >= 0.50
                        and abs(center_a - center_b) <= 0.75 * min(len_a, len_b)
                    )
                    if direct_match:
                        group.append(second)
                    else:
                        keep.append(second)
                local_remaining = keep
                if len(group) >= 5:
                    span_components.append(group)

            for local_walls in span_components:
                ordered = sorted(
                    ((self._offset(w, ux, uy), w) for w in local_walls),
                    key=lambda item: item[0],
                )
                offset_groups: list[list[tuple[float, SingleLineWall]]] = []
                duplicate_tol = max(2.0, 0.05 * px_per_m)
                for offset, wall in ordered:
                    if not offset_groups:
                        offset_groups.append([(offset, wall)])
                        continue
                    center = statistics.fmean(item[0] for item in offset_groups[-1])
                    if abs(offset - center) <= duplicate_tol:
                        offset_groups[-1].append((offset, wall))
                    else:
                        offset_groups.append([(offset, wall)])
                if len(offset_groups) < 5:
                    continue

                representatives = [
                    (statistics.fmean(item[0] for item in group), max((item[1] for item in group), key=self._length), group)
                    for group in offset_groups
                ]
                consumed: set[int] = set()
                candidates: list[tuple[int, int, float, dict]] = []
                for start_index in range(0, len(representatives) - 4):
                    for end_index in range(start_index + 4, len(representatives)):
                        subset = representatives[start_index : end_index + 1]
                        offsets = [item[0] for item in subset]
                        gaps = [b - a for a, b in zip(offsets, offsets[1:])]
                        median_gap = statistics.median(gaps)
                        if not 0.07 * px_per_m <= median_gap <= 0.45 * px_per_m:
                            continue
                        mean_gap = statistics.fmean(gaps)
                        spacing_cv = statistics.pstdev(gaps) / max(mean_gap, 1e-6)
                        if spacing_cv > 0.22:
                            continue
                        members = [item[1] for item in subset]
                        intervals = [self._interval(item, ux, uy) for item in members]
                        common = max(0.0, min(item[1] for item in intervals) - max(item[0] for item in intervals))
                        median_length = statistics.median(item[1] - item[0] for item in intervals)
                        common_span_ratio = common / max(median_length, 1e-6)
                        if common_span_ratio < 0.50:
                            continue
                        lengths = [self._length(item) for item in members]
                        length_similarity = min(lengths) / max(max(lengths), 1e-6)
                        if length_similarity < 0.45:
                            continue
                        offset_span = offsets[-1] - offsets[0]
                        metrics = {
                            "angle_deg": self._angle(reference),
                            "member_count": len(members),
                            "median_spacing_px": median_gap,
                            "median_spacing_m": median_gap / px_per_m,
                            "spacing_cv": spacing_cv,
                            "median_length_px": statistics.median(lengths),
                            "median_length_m": statistics.median(lengths) / px_per_m,
                            "length_similarity": length_similarity,
                            "common_span_ratio": common_span_ratio,
                            "offset_span_px": offset_span,
                            "offset_span_m": offset_span / px_per_m,
                        }
                        score = len(members) + common_span_ratio - spacing_cv
                        candidates.append((start_index, end_index, score, metrics))

                candidates.sort(key=lambda item: item[2], reverse=True)
                for start_index, end_index, _, metrics in candidates:
                    indexes = set(range(start_index, end_index + 1))
                    if indexes & consumed:
                        continue
                    consumed.update(indexes)
                    selected_groups = [representatives[index][2] for index in range(start_index, end_index + 1)]
                    all_members = [entry[1] for group in selected_groups for entry in group]
                    xs = [point for wall in all_members for point in (wall.start_px[0], wall.end_px[0])]
                    ys = [point for wall in all_members for point in (wall.start_px[1], wall.end_px[1])]
                    metrics = dict(metrics)
                    metrics["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
                    oblique = not self._is_axis_aligned(float(metrics["angle_deg"]), tolerance=7.0)
                    metrics["oblique"] = oblique

                    offset_span_m = float(metrics["offset_span_m"])
                    median_length_m = float(metrics["median_length_m"])
                    # Un haz muy estrecho de lineas largas suele ser varias
                    # hipotesis/caras del mismo muro y se deja al canonicalizer.
                    if offset_span_m < 0.50 and median_length_m > 0.40:
                        continue

                    confidence = min(
                        0.97,
                        0.66
                        + 0.04 * min(int(metrics["member_count"]) - 4, 5)
                        + 0.13 * max(0.0, 1.0 - float(metrics["spacing_cv"]) / 0.22)
                        + 0.08 * float(metrics["common_span_ratio"]),
                    )
                    ids = sorted({wall.id for wall in all_members})
                    if oblique:
                        output_class = "EXCLUDED_GRAPHIC"
                        family = None
                        reason = "familia diagonal repetitiva compatible con hatch/representacion grafica"
                    elif offset_span_m < 0.50 and median_length_m <= 0.40:
                        output_class = "EXCLUDED_GRAPHIC"
                        family = None
                        reason = "familia compacta de trazos repetitivos compatible con ticks/simbologia grafica"
                    else:
                        output_class = "ARCHITECTURAL_ELEMENT"
                        family = "REPETITIVE_LINEAR_ELEMENT"
                        reason = "familia local repetitiva compatible con escalera/cortina/elemento arquitectonico"
                    patterns.append(
                        PatternEvidence(
                            id=self._id("RP", ids),
                            kind="REPETITIVE_PARALLEL",
                            wall_ids=ids,
                            confidence=confidence,
                            output_class_hint=output_class,
                            architectural_family_hint=family,
                            reason=reason,
                            metrics=metrics,
                        )
                    )
        return self._dedupe_patterns(patterns)

    def _orthogonal_grid_patterns(self, *, repetitive: Sequence[PatternEvidence]) -> list[PatternEvidence]:
        output: list[PatternEvidence] = []
        for i, first in enumerate(repetitive):
            for second in repetitive[i + 1:]:
                a = float(first.metrics.get("angle_deg", 0.0))
                b = float(second.metrics.get("angle_deg", 0.0))
                if abs(self._angle_diff(a, b) - 90.0) > 7.0:
                    continue
                bbox_a = first.metrics.get("bbox")
                bbox_b = second.metrics.get("bbox")
                if not bbox_a or not bbox_b:
                    continue
                overlap = self._bbox_overlap_ratio(tuple(bbox_a), tuple(bbox_b))
                if overlap < 0.30:
                    continue
                ids = sorted(set(first.wall_ids) | set(second.wall_ids))
                confidence = min(0.98, 0.72 + 0.16 * overlap + 0.06 * min(first.confidence, second.confidence))
                output.append(
                    PatternEvidence(
                        id=self._id("GRID", ids),
                        kind="ORTHOGONAL_GRID",
                        wall_ids=ids,
                        confidence=confidence,
                        output_class_hint="EXCLUDED_GRAPHIC",
                        architectural_family_hint=None,
                        reason="dos familias repetitivas casi ortogonales forman una reticula grafica",
                        metrics={
                            "angle_a_deg": a,
                            "angle_b_deg": b,
                            "bbox_overlap_ratio": overlap,
                            "source_pattern_ids": [first.id, second.id],
                        },
                    )
                )
        return self._dedupe_patterns(output)

    def _angular_hub_patterns(
        self,
        *,
        walls: Sequence[SingleLineWall],
        px_per_m: float,
    ) -> list[PatternEvidence]:
        tol = max(4.0, 0.10 * px_per_m)
        endpoints: list[tuple[float, float, SingleLineWall]] = []
        for wall in walls:
            endpoints.append((wall.start_px[0], wall.start_px[1], wall))
            endpoints.append((wall.end_px[0], wall.end_px[1], wall))

        clusters: list[list[tuple[float, float, SingleLineWall]]] = []
        for endpoint in endpoints:
            ex, ey, _ = endpoint
            target = None
            for cluster in clusters:
                cx = sum(item[0] for item in cluster) / len(cluster)
                cy = sum(item[1] for item in cluster) / len(cluster)
                if math.hypot(ex - cx, ey - cy) <= tol:
                    target = cluster
                    break
            if target is None:
                clusters.append([endpoint])
            else:
                target.append(endpoint)

        output: list[PatternEvidence] = []
        for cluster in clusters:
            members = {item[2].id: item[2] for item in cluster}
            if len(members) < 3:
                continue
            angles = sorted(self._angle(item) for item in members.values())
            unique = self._unique_angles(angles, tolerance=12.0)
            if len(unique) < 3:
                continue
            review_ratio = sum(item.role == "REVIEW" for item in members.values()) / len(members)
            mean_conf = sum(item.confidence for item in members.values()) / len(members)
            oblique_count = sum(not self._is_axis_aligned(self._angle(item), tolerance=7.0) for item in members.values())
            if review_ratio < 0.67 or oblique_count < 2:
                continue
            confidence = min(
                0.92,
                0.56
                + 0.06 * min(len(members), 5)
                + 0.08 * min(oblique_count, 3)
                + 0.08 * review_ratio
                - 0.10 * max(0.0, mean_conf - 0.60),
            )
            if confidence < 0.78:
                continue
            ids = sorted(members)
            output.append(
                PatternEvidence(
                    id=self._id("HUB", ids),
                    kind="ANGULAR_HUB",
                    wall_ids=ids,
                    confidence=confidence,
                    output_class_hint="ARCHITECTURAL_ELEMENT",
                    architectural_family_hint="ROOF_COVER_CANDIDATE",
                    reason="varias direcciones no paralelas convergen en un nodo local sin firma topologica fuerte de muro",
                    metrics={
                        "member_count": len(members),
                        "unique_direction_count": len(unique),
                        "oblique_member_count": oblique_count,
                        "review_ratio": review_ratio,
                        "mean_confidence": mean_conf,
                        "endpoint_tolerance_px": tol,
                    },
                )
            )
        return self._dedupe_patterns(output)

    @classmethod
    def _parallel_family_compatible(cls, first: SingleLineWall, second: SingleLineWall, *, px_per_m: float) -> bool:
        a = cls._angle(first)
        b = cls._angle(second)
        if cls._angle_diff(a, b) > 3.0:
            return False
        ux, uy = cls._axis(first)
        a0, a1 = cls._interval(first, ux, uy)
        b0, b1 = cls._interval(second, ux, uy)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        min_len = max(1e-6, min(a1 - a0, b1 - b0))
        if overlap / min_len < 0.55:
            return False
        separation = abs(cls._offset(first, ux, uy) - cls._offset(second, ux, uy))
        return 0.04 * px_per_m <= separation <= 0.45 * px_per_m

    @classmethod
    def _parallel_signature(cls, walls: Sequence[SingleLineWall], *, px_per_m: float) -> tuple[float, dict] | None:
        reference = max(walls, key=cls._length)
        ux, uy = cls._axis(reference)
        offsets = sorted(cls._offset(item, ux, uy) for item in walls)
        gaps = [b - a for a, b in zip(offsets, offsets[1:]) if b - a > 1e-6]
        if len(gaps) < 4:
            return None
        median_gap = statistics.median(gaps)
        if median_gap < 0.04 * px_per_m or median_gap > 0.45 * px_per_m:
            return None
        mean_gap = statistics.fmean(gaps)
        spacing_cv = statistics.pstdev(gaps) / max(mean_gap, 1e-6)
        if spacing_cv > 0.25:
            return None

        lengths = [cls._length(item) for item in walls]
        median_length = statistics.median(lengths)
        length_similarity = min(lengths) / max(max(lengths), 1e-6)
        if length_similarity < 0.45:
            return None

        xs = [point for wall in walls for point in (wall.start_px[0], wall.end_px[0])]
        ys = [point for wall in walls for point in (wall.start_px[1], wall.end_px[1])]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        angle = cls._angle(reference)
        oblique = not cls._is_axis_aligned(angle, tolerance=7.0)
        confidence = min(
            0.96,
            0.64
            + 0.06 * min(len(walls) - 4, 4)
            + 0.14 * max(0.0, 1.0 - spacing_cv / 0.25)
            + 0.08 * length_similarity,
        )
        return confidence, {
            "angle_deg": angle,
            "member_count": len(walls),
            "median_spacing_px": median_gap,
            "median_spacing_m": median_gap / px_per_m,
            "spacing_cv": spacing_cv,
            "median_length_px": median_length,
            "median_length_m": median_length / px_per_m,
            "length_similarity": length_similarity,
            "bbox": list(bbox),
            "oblique": oblique,
        }

    @staticmethod
    def _dedupe_patterns(patterns: Sequence[PatternEvidence]) -> list[PatternEvidence]:
        best: dict[tuple[str, tuple[str, ...]], PatternEvidence] = {}
        for item in patterns:
            key = (item.kind, tuple(sorted(item.wall_ids)))
            current = best.get(key)
            if current is None or item.confidence > current.confidence:
                best[key] = item
        return list(best.values())

    @staticmethod
    def _bbox_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        if inter <= 0.0:
            return 0.0
        area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
        return inter / min(area_a, area_b)

    @staticmethod
    def _unique_angles(angles: Sequence[float], *, tolerance: float) -> list[float]:
        unique: list[float] = []
        for angle in angles:
            if not any(PostFilterGeometryPatternAnalyzer._angle_diff(angle, existing) <= tolerance for existing in unique):
                unique.append(angle)
        return unique

    @staticmethod
    def _is_axis_aligned(angle: float, *, tolerance: float) -> bool:
        return min(abs(angle), abs(angle - 90.0), abs(angle - 180.0)) <= tolerance

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
    def _id(prefix: str, values: Sequence[str]) -> str:
        digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:12]
        return f"PF_{prefix}_{digest}"
