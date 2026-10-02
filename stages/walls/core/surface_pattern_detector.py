from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from shapely.geometry import LineString, box

from .context_models import ContextRegion, RepetitiveAxisProfile
from .drawing_model import DrawingBBox, DrawingLine, DrawingModel
from .metric_normalized_context import MetricNormalizedContext


@dataclass(frozen=True)
class _SegmentRun:
    angle_deg: float
    cross: float
    segments: tuple[DrawingLine, ...]
    along_centers: tuple[float, ...]
    median_length: float
    median_gap: float
    length_cv: float
    gap_cv: float
    lo: float
    hi: float

    @property
    def pitch(self) -> float:
        return self.median_length + max(0.0, self.median_gap)


class SurfacePatternDetector:
    """Detecta acabados modulares dibujados como segmentos repetidos.

    Complementa ``RepeatedCellDetector``. No exige que cada celda esté cerrada:
    reconoce filas/columnas de segmentos sólidos repetidos con longitud y paso
    regulares y exige una segunda evidencia local (otra fila paralela compatible
    o separadores ortogonales). Esto cubre mosaicos/baldosas dibujados manualmente
    y evita confundir una única línea fragmentada con un acabado.

    Cuando existe escala métrica, publica además la dimensión física estimada del
    módulo y usa 0.15 m como soporte positivo de un módulo de acabado grande. La
    ausencia de escala nunca se convierte en una exclusión por sí sola.
    """

    VERSION = "SURFACE_PATTERN_DETECTOR_V6_2"
    AXIS_TOLERANCE_DEG = 5.5
    MIN_SEGMENTS_PER_RUN = 4

    def detect(
        self,
        *,
        drawing: DrawingModel,
        lines: Sequence[DrawingLine],
        polygon,
        scale_context: MetricNormalizedContext | None = None,
    ) -> list[ContextRegion]:
        metric = scale_context or MetricNormalizedContext(drawing=drawing)
        min_dim = metric.reference_min_dim_native_px
        minimum_length = max(metric.to_native_px(5.0), 0.0045 * min_dim)
        maximum_length = max(metric.to_native_px(36.0), 0.18 * min_dim)
        track_tolerance = max(metric.to_native_px(2.0), 0.0028 * min_dim)

        axis_lines = [
            line
            for line in lines
            if line.kind == "LINE"
            and not line.dashed
            and minimum_length <= line.length_px <= maximum_length
            and min(self._axis_diff(line.angle_deg, 0.0), self._axis_diff(line.angle_deg, 90.0))
            <= self.AXIS_TOLERANCE_DEG
        ]
        if len(axis_lines) < self.MIN_SEGMENTS_PER_RUN:
            return []

        runs: list[_SegmentRun] = []
        for target_angle in (0.0, 90.0):
            oriented = [
                line
                for line in axis_lines
                if self._axis_diff(line.angle_deg, target_angle) <= self.AXIS_TOLERANCE_DEG
            ]
            runs.extend(
                self._segment_runs(
                    lines=oriented,
                    target_angle=target_angle,
                    track_tolerance=track_tolerance,
                    min_dim=min_dim,
                )
            )

        if not runs:
            return []

        regions: list[ContextRegion] = []
        used: set[int] = set()
        for index, run in enumerate(runs):
            if index in used:
                continue
            compatible = [index]
            for other_index, other in enumerate(runs):
                if other_index == index or other_index in used:
                    continue
                if not self._parallel_run_compatible(run, other, min_dim=min_dim):
                    continue
                compatible.append(other_index)

            selected_runs = [runs[item] for item in compatible]
            all_segments = [segment for item in selected_runs for segment in item.segments]
            bbox = self._lines_bbox(all_segments)
            orthogonal_support = self._orthogonal_support(
                primary=run,
                all_lines=axis_lines,
                bbox=bbox,
                tolerance=track_tolerance,
            )
            parallel_support = len(selected_runs) >= 2
            if not parallel_support and orthogonal_support < 3:
                continue

            region = self._region_from_runs(
                runs=selected_runs,
                orthogonal_support=orthogonal_support,
                drawing=drawing,
                polygon=polygon,
                metric=metric,
            )
            if region is None:
                continue
            regions.append(region)
            used.update(compatible)

        return self._dedupe(regions)

    def _segment_runs(
        self,
        *,
        lines: Sequence[DrawingLine],
        target_angle: float,
        track_tolerance: float,
        min_dim: float,
    ) -> list[_SegmentRun]:
        tracks: list[list[DrawingLine]] = []
        for line in sorted(lines, key=lambda item: self._cross_coordinate(item, target_angle)):
            cross = self._cross_coordinate(line, target_angle)
            destination: list[DrawingLine] | None = None
            for track in reversed(tracks[-8:]):
                reference = median([self._cross_coordinate(item, target_angle) for item in track])
                if abs(cross - reference) <= track_tolerance:
                    destination = track
                    break
            if destination is None:
                tracks.append([line])
            else:
                destination.append(line)

        runs: list[_SegmentRun] = []
        for track in tracks:
            if len(track) < self.MIN_SEGMENTS_PER_RUN:
                continue
            intervals = []
            for line in track:
                lo, hi = self._along_interval(line, target_angle)
                intervals.append((lo, hi, line))
            intervals.sort(key=lambda item: (item[0], item[1]))

            # Separa clusters cuando existe un vacío mucho mayor que el patrón
            # local; el análisis de regularidad posterior decide cada cluster.
            clusters: list[list[tuple[float, float, DrawingLine]]] = []
            current: list[tuple[float, float, DrawingLine]] = []
            for item in intervals:
                if not current:
                    current = [item]
                    continue
                previous = current[-1]
                prev_len = previous[1] - previous[0]
                this_len = item[1] - item[0]
                gap = item[0] - previous[1]
                split_threshold = max(0.06 * min_dim, 2.8 * max(prev_len, this_len))
                if gap > split_threshold:
                    clusters.append(current)
                    current = [item]
                else:
                    current.append(item)
            if current:
                clusters.append(current)

            for cluster in clusters:
                run = self._make_run(cluster=cluster, target_angle=target_angle)
                if run is not None:
                    runs.append(run)
        return runs

    def _make_run(
        self,
        *,
        cluster: Sequence[tuple[float, float, DrawingLine]],
        target_angle: float,
    ) -> _SegmentRun | None:
        if len(cluster) < self.MIN_SEGMENTS_PER_RUN:
            return None

        lengths = [hi - lo for lo, hi, _ in cluster]
        median_length = float(median(lengths))
        if median_length <= 1e-6:
            return None

        # Descarta fragmentos anómalos y conserva una secuencia de módulos de
        # longitud comparable. Un muro fragmentado irregular no suele cumplirlo.
        filtered = [
            item
            for item, length in zip(cluster, lengths)
            if 0.68 <= length / median_length <= 1.42
        ]
        if len(filtered) < self.MIN_SEGMENTS_PER_RUN:
            return None

        lengths = [hi - lo for lo, hi, _ in filtered]
        centers = [(lo + hi) / 2.0 for lo, hi, _ in filtered]
        gaps = [
            max(0.0, current[0] - previous[1])
            for previous, current in zip(filtered[:-1], filtered[1:])
        ]
        positive_gaps = [gap for gap in gaps if gap > 1e-6]
        if len(positive_gaps) < len(filtered) - 2:
            return None

        length_cv = self._cv(lengths)
        gap_cv = self._cv(positive_gaps)
        median_gap = float(median(positive_gaps)) if positive_gaps else 0.0
        if length_cv > 0.24 or gap_cv > 0.38:
            return None
        if median_gap > 1.60 * median(lengths):
            return None

        cross = float(median([self._cross_coordinate(line, target_angle) for _, _, line in filtered]))
        return _SegmentRun(
            angle_deg=target_angle,
            cross=cross,
            segments=tuple(line for _, _, line in filtered),
            along_centers=tuple(float(value) for value in centers),
            median_length=float(median(lengths)),
            median_gap=median_gap,
            length_cv=length_cv,
            gap_cv=gap_cv,
            lo=float(min(lo for lo, _, _ in filtered)),
            hi=float(max(hi for _, hi, _ in filtered)),
        )

    def _parallel_run_compatible(self, first: _SegmentRun, second: _SegmentRun, *, min_dim: float) -> bool:
        if self._axis_diff(first.angle_deg, second.angle_deg) > self.AXIS_TOLERANCE_DEG:
            return False
        separation = abs(first.cross - second.cross)
        if separation <= 1e-6 or separation > max(0.14 * min_dim, 2.8 * max(first.median_length, second.median_length)):
            return False
        length_ratio = min(first.median_length, second.median_length) / max(first.median_length, second.median_length)
        if length_ratio < 0.72:
            return False
        pitch_ratio = min(first.pitch, second.pitch) / max(first.pitch, second.pitch)
        if pitch_ratio < 0.72:
            return False
        overlap = max(0.0, min(first.hi, second.hi) - max(first.lo, second.lo))
        union = max(first.hi, second.hi) - min(first.lo, second.lo)
        if overlap / max(union, 1e-6) < 0.42:
            return False

        phase_tolerance = 0.32 * max(first.pitch, second.pitch)
        matches = 0
        for center in first.along_centers:
            if min(abs(center - other) for other in second.along_centers) <= phase_tolerance:
                matches += 1
        return matches / max(1, min(len(first.along_centers), len(second.along_centers))) >= 0.55

    def _orthogonal_support(
        self,
        *,
        primary: _SegmentRun,
        all_lines: Sequence[DrawingLine],
        bbox: DrawingBBox,
        tolerance: float,
    ) -> int:
        target = 90.0 if self._axis_diff(primary.angle_deg, 0.0) <= self.AXIS_TOLERANCE_DEG else 0.0
        expanded = box(
            bbox.x_min - tolerance,
            bbox.y_min - tolerance,
            bbox.x_max + tolerance,
            bbox.y_max + tolerance,
        )
        supports = 0
        for line in all_lines:
            if self._axis_diff(line.angle_deg, target) > self.AXIS_TOLERANCE_DEG:
                continue
            geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
            if not geom.intersects(expanded):
                continue
            supports += 1
        return supports

    def _region_from_runs(
        self,
        *,
        runs: Sequence[_SegmentRun],
        orthogonal_support: int,
        drawing: DrawingModel,
        polygon,
        metric: MetricNormalizedContext,
    ) -> ContextRegion | None:
        segments = [segment for run in runs for segment in run.segments]
        bbox = self._lines_bbox(segments)
        # V6.3: la zona contextual debe envolver el patrón, no la habitación.
        # El 10% de la longitud de una pista podía añadir decenas de píxeles y
        # absorber muros reales adyacentes. El margen queda ligado al stroke y
        # solo conserva una fracción pequeña del tamaño modular observado.
        median_pattern_length = float(median([run.median_length for run in runs]))
        pad = max(metric.to_native_px(1.5), 0.04 * median_pattern_length)
        bbox = DrawingBBox(
            x_min=max(0.0, bbox.x_min - pad),
            y_min=max(0.0, bbox.y_min - pad),
            x_max=min(float(drawing.width_px), bbox.x_max + pad),
            y_max=min(float(drawing.height_px), bbox.y_max + pad),
        )
        geom = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        if polygon.intersection(geom).area / max(geom.area, 1e-6) < 0.70:
            return None

        length_cv = float(median([run.length_cv for run in runs]))
        gap_cv = float(median([run.gap_cv for run in runs]))
        regularity = max(0.0, 1.0 - 0.55 * min(1.0, length_cv / 0.24) - 0.45 * min(1.0, gap_cv / 0.38))
        count = sum(len(run.segments) for run in runs)
        count_score = min(1.0, (count - 3) / 9.0)
        parallel_score = min(1.0, len(runs) / 2.0)
        orthogonal_score = min(1.0, orthogonal_support / 6.0)

        inferred_cross = None
        if len(runs) >= 2:
            cross_gaps = sorted(abs(a.cross - b.cross) for i, a in enumerate(runs) for b in runs[i + 1 :] if abs(a.cross - b.cross) > 1e-6)
            if cross_gaps:
                inferred_cross = float(median(cross_gaps))
        inferred_along = float(median([run.pitch for run in runs]))
        along_m = metric.to_meters(inferred_along)
        cross_m = metric.to_meters(inferred_cross) if inferred_cross is not None else None
        physical_size_support = None
        if along_m is not None and cross_m is not None:
            physical_size_support = bool(min(along_m, cross_m) >= 0.15)

        physical_score = 0.0
        if physical_size_support is True:
            physical_score = 1.0
        elif physical_size_support is False:
            physical_score = -0.35

        confidence = (
            0.36 * regularity
            + 0.24 * count_score
            + 0.22 * max(parallel_score, orthogonal_score)
            + 0.18 * min(1.0, 0.55 * parallel_score + 0.45 * orthogonal_score)
            + 0.08 * physical_score
        )
        confidence = max(0.0, min(0.98, confidence))
        if confidence < 0.58:
            return None

        member_ids = sorted({
            item
            for line in segments
            for item in (
                {line.id}
                | {str(item) for item in line.evidence_ids if item}
                | set(line.metadata.get("physical_stroke_member_line_ids") or [])
                | set(line.metadata.get("context_equivalent_line_ids") or [])
            )
        })
        pattern_segments = [
            [float(line.start.x), float(line.start.y), float(line.end.x), float(line.end.y)]
            for line in segments
        ]

        profiles: list[RepetitiveAxisProfile] = []
        for angle in (0.0, 90.0):
            same = [run for run in runs if self._axis_diff(run.angle_deg, angle) <= self.AXIS_TOLERANCE_DEG]
            if not same:
                continue
            coords = sorted({round(run.cross, 4) for run in same})
            if len(coords) >= 2:
                spacing = float(median([b - a for a, b in zip(coords[:-1], coords[1:]) if b - a > 1e-6]))
                profile_coords = coords
            else:
                # Una única fila/columna sigue siendo útil por member ids y por
                # geometry support; el perfil usa centros modulares para poder
                # reconocer candidatos geométricos derivados del patrón.
                spacing = float(median([run.pitch for run in same]))
                profile_coords = list(same[0].along_centers)
                angle = 90.0 if angle == 0.0 else 0.0
            if len(profile_coords) < 3:
                continue
            profiles.append(
                RepetitiveAxisProfile(
                    angle_deg=angle,
                    spacing_px=max(0.1, spacing),
                    spacing_cv=max(run.gap_cv for run in same),
                    member_count=len(profile_coords),
                    median_length_px=max(1.0, float(median([run.median_length for run in same]))),
                    common_span_ratio=1.0,
                    length_similarity=max(0.0, 1.0 - float(median([run.length_cv for run in same]))),
                    track_coordinates=[float(value) for value in profile_coords],
                    member_line_ids=member_ids,
                )
            )

        digest = hashlib.sha1(("SURFACE|" + "|".join(member_ids)).encode()).hexdigest()[:16]
        return ContextRegion(
            id=f"CTX__FLOOR_FINISH_GRID_REGION__{digest}",
            region_type="FLOOR_FINISH_GRID_REGION",
            bbox=bbox,
            confidence=confidence,
            quarantine_enabled=confidence >= 0.70,
            profiles=profiles,
            member_line_ids=member_ids,
            semantic_support=0.0,
            metadata={
                "detector": self.VERSION,
                "source": "SEGMENTED_MODULAR_SURFACE_PATTERN",
                "run_count": len(runs),
                "segment_count": count,
                "orthogonal_support_count": orthogonal_support,
                "median_segment_length_px": float(median([run.median_length for run in runs])),
                "median_gap_px": float(median([run.median_gap for run in runs])),
                "length_cv": length_cv,
                "gap_cv": gap_cv,
                "estimated_module_along_px": inferred_along,
                "estimated_module_cross_px": inferred_cross,
                "estimated_module_along_m": along_m,
                "estimated_module_cross_m": cross_m,
                "minimum_large_tile_dimension_m": 0.15,
                "physical_large_tile_support": physical_size_support,
                "pattern_segments": pattern_segments,
                "member_tolerance_px": pad,
            },
        )

    @staticmethod
    def _cross_coordinate(line: DrawingLine, target_angle: float) -> float:
        if target_angle == 0.0:
            return 0.5 * (line.start.y + line.end.y)
        return 0.5 * (line.start.x + line.end.x)

    @staticmethod
    def _along_interval(line: DrawingLine, target_angle: float) -> tuple[float, float]:
        if target_angle == 0.0:
            return tuple(sorted((float(line.start.x), float(line.end.x))))
        return tuple(sorted((float(line.start.y), float(line.end.y))))

    @staticmethod
    def _axis_diff(angle: float, target: float) -> float:
        diff = abs((angle - target) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _lines_bbox(lines: Sequence[DrawingLine]) -> DrawingBBox:
        xs = [coord for line in lines for coord in (line.start.x, line.end.x)]
        ys = [coord for line in lines for coord in (line.start.y, line.end.y)]
        return DrawingBBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    @staticmethod
    def _cv(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        if abs(mean) <= 1e-9:
            return 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return math.sqrt(variance) / abs(mean)

    @staticmethod
    def _bbox_overlap_ratio(first: DrawingBBox, second: DrawingBBox) -> float:
        x0 = max(first.x_min, second.x_min)
        y0 = max(first.y_min, second.y_min)
        x1 = min(first.x_max, second.x_max)
        y1 = min(first.y_max, second.y_max)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        a = max(1e-6, (first.x_max - first.x_min) * (first.y_max - first.y_min))
        b = max(1e-6, (second.x_max - second.x_min) * (second.y_max - second.y_min))
        return inter / min(a, b)

    def _dedupe(self, regions: Sequence[ContextRegion]) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        for region in sorted(regions, key=lambda item: -item.confidence):
            if any(self._bbox_overlap_ratio(region.bbox, other.bbox) >= 0.82 for other in output):
                continue
            output.append(region)
        return output
