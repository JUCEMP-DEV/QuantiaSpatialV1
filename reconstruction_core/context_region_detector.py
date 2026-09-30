from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from shapely.geometry import LineString, box

from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import EditablePerimeterModel

from .context_models import ContextRegion, RepetitiveAxisProfile
from .context_region_classifier import ContextRegionClassifier
from .context_region_assembler import ContextRegionAssembler
from .drawing_model import DrawingBBox, DrawingLine, DrawingModel
from .perimeter_adapter import perimeter_polygon
from .physical_stroke_normalizer import PhysicalStrokeNormalizer
from .repeated_cell_detector import RepeatedCellDetector
from .surface_pattern_detector import SurfacePatternDetector
from .level_scale_normalizer import LevelScaleProfile
from .metric_normalized_context import MetricNormalizedContext
from .element_context_detector import ElementContextDetector


@dataclass(frozen=True)
class _ObservedTrack:
    line: DrawingLine
    line_ids: tuple[str, ...]
    cross: float
    lo: float
    hi: float


@dataclass(frozen=True)
class _Profile:
    model: RepetitiveAxisProfile
    bbox: DrawingBBox
    confidence: float
    tracks: tuple[_ObservedTrack, ...]
    spacing_to_length: float


class ContextRegionDetector:
    """Detecta regiones gráficas repetitivas antes del CandidateGraph.

    La detección se hace sobre primitivas del DrawingModel, no sobre el score de
    WallCandidate. De esta forma una hipótesis no puede "autovalidarse" usando
    las mismas evidencias que la generaron.

    La salida tampoco borra geometría: publica regiones contextuales que después
    serán evaluadas por CandidateContextGate.
    """

    ANGLE_TOLERANCE_DEG = 3.5
    GRID_ORTHOGONAL_TOLERANCE_DEG = 6.0

    def __init__(
        self,
        *,
        stroke_normalizer: PhysicalStrokeNormalizer | None = None,
        region_assembler: ContextRegionAssembler | None = None,
        repeated_cell_detector: RepeatedCellDetector | None = None,
        surface_pattern_detector: SurfacePatternDetector | None = None,
        classifier: ContextRegionClassifier | None = None,
        element_context_detector: ElementContextDetector | None = None,
    ) -> None:
        self.stroke_normalizer = stroke_normalizer or PhysicalStrokeNormalizer()
        self.region_assembler = region_assembler or ContextRegionAssembler()
        self.repeated_cell_detector = repeated_cell_detector or RepeatedCellDetector()
        self.surface_pattern_detector = surface_pattern_detector or SurfacePatternDetector()
        self.classifier = classifier or ContextRegionClassifier()
        self.element_context_detector = element_context_detector or ElementContextDetector()

    def detect(
        self,
        *,
        drawing: DrawingModel,
        perimeter: EditablePerimeterModel,
        scale_profile: LevelScaleProfile | None = None,
    ) -> list[ContextRegion]:
        polygon = perimeter_polygon(perimeter)
        metric = MetricNormalizedContext(drawing=drawing, profile=scale_profile)
        observed = self._observed_lines(drawing=drawing, polygon=polygon, metric=metric)

        # Camino V5: periodicidad lineal. Se conserva, pero V6 ya no deja la
        # salida en clases matemáticas genéricas: primero ensambla y después
        # clasifica su función gráfica.
        repetitive_regions: list[ContextRegion] = []
        if len(observed) >= 3:
            components = self._parallel_components(observed, drawing=drawing, metric=metric)
            profiles: list[_Profile] = []
            for component in components:
                profiles.extend(self._regular_profiles(component, drawing=drawing, metric=metric))

            if profiles:
                parallel_regions = [
                    self._parallel_region(profile=profile, drawing=drawing)
                    for profile in profiles
                ]
                grid_regions = self._grid_regions(profiles=profiles, drawing=drawing, metric=metric)

                filtered_parallel: list[ContextRegion] = []
                for region in parallel_regions:
                    covered = any(
                        self._bbox_overlap_ratio(region.bbox, grid.bbox) >= 0.72
                        and set(region.member_line_ids).issubset(set(grid.member_line_ids))
                        for grid in grid_regions
                    )
                    if not covered:
                        filtered_parallel.append(region)

                assembled = self.region_assembler.assemble(
                    regions=[*grid_regions, *filtered_parallel],
                    drawing=drawing,
                    scale_context=metric,
                )
                repetitive_regions = self.classifier.classify_repetitive(
                    regions=assembled,
                    drawing=drawing,
                    scale_context=metric,
                )

        # Nuevo V6: detecta celdas rectangulares repetidas aunque no existan dos
        # familias de líneas largas que se crucen. Es el camino que cubre mosaico,
        # acabado y mobiliario modular.
        cell_regions = self.repeated_cell_detector.detect(
            drawing=drawing,
            lines=observed,
            polygon=polygon,
            scale_context=metric,
        )

        # V6.2: segundo camino para acabados modulares. Detecta filas/columnas de
        # segmentos repetidos aunque las celdas no estén completamente cerradas.
        # Esta firma es típica de baldosas/mosaicos dibujados manualmente y no se
        # basa en nombres de caso ni en posiciones conocidas.
        surface_regions = self.surface_pattern_detector.detect(
            drawing=drawing,
            lines=observed,
            polygon=polygon,
            scale_context=metric,
        )

        # Referencias normalizadas (ejes/cotas) se clasifican por su propia firma
        # gráfica y textual; no compiten como simple repetición.
        reference_regions = self.classifier.detect_reference_regions(
            drawing=drawing,
            polygon=polygon,
            scale_context=metric,
        )

        # Element Context Isolation V4 conserva información funcional que antes
        # se perdía antes del CandidateGraph (puertas/openings, ventanas, columnas,
        # fixtures, furniture y texto/símbolos). Estas regiones NO sustituyen a
        # los detectores geométricos V6; aportan provenance adicional.
        element_regions = self.element_context_detector.detect(
            drawing=drawing,
            scale_profile=scale_profile,
            architectural_polygon=polygon,
        )

        # F02 es verdad operativa inmutable. Sus tramos se publican como regiones
        # de protección para impedir que un hatch/retícula que toque el borde
        # degrade un muro perimetral real.
        protected_regions = self._wall_protected_regions(
            perimeter=perimeter,
            drawing=drawing,
            metric=metric,
        )

        regions = self._dedupe_context_regions(
            [*protected_regions, *element_regions, *surface_regions, *cell_regions, *reference_regions, *repetitive_regions]
        )
        return [self._attach_scale_metadata(region, metric=metric) for region in regions]

    def _observed_lines(self, *, drawing: DrawingModel, polygon, metric: MetricNormalizedContext) -> list[DrawingLine]:
        min_dim = metric.reference_min_dim_native_px
        minimum_length = max(metric.to_native_px(8.0), min_dim * 0.010)
        boundary_tol = max(metric.to_native_px(2.0), min_dim * 0.006)

        usable: list[DrawingLine] = []
        for line in drawing.lines:
            if line.kind != "LINE" or line.dashed or line.length_px < minimum_length:
                continue
            geom = self._line(line)
            if geom.length <= 1e-6:
                continue
            inside = polygon.buffer(metric.to_native_px(1.0), cap_style=2, join_style=2).intersection(geom).length
            if inside / geom.length < 0.65:
                continue
            near_boundary = polygon.boundary.buffer(
                boundary_tol,
                cap_style=2,
                join_style=2,
            ).intersection(geom).length
            if near_boundary / geom.length >= 0.82:
                continue
            usable.append(line)

        return self.stroke_normalizer.normalize(
            lines=usable,
            drawing=drawing,
            scale_context=metric,
        )

    def _parallel_components(
        self,
        lines: Sequence[DrawingLine],
        *,
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> list[list[DrawingLine]]:
        # Bin angular evita un O(N²) global entre orientaciones incompatibles.
        bins: dict[int, list[DrawingLine]] = {}
        for line in lines:
            key = int(round(line.angle_deg / 5.0)) % 36
            bins.setdefault(key, []).append(line)

        components: list[list[DrawingLine]] = []
        visited: set[str] = set()
        by_id = {line.id: line for line in lines}

        for line in lines:
            if line.id in visited:
                continue
            queue = [line]
            visited.add(line.id)
            component: list[DrawingLine] = []
            while queue:
                current = queue.pop()
                component.append(current)
                current_bin = int(round(current.angle_deg / 5.0)) % 36
                neighbor_bins = {
                    current_bin,
                    (current_bin - 1) % 36,
                    (current_bin + 1) % 36,
                }
                candidates = [item for key in neighbor_bins for item in bins.get(key, [])]
                for other in candidates:
                    if other.id in visited:
                        continue
                    if not self._same_parallel_family(current, other, drawing=drawing, metric=metric):
                        continue
                    visited.add(other.id)
                    queue.append(by_id[other.id])
            if len(component) >= 3:
                components.append(component)
        return components

    def _same_parallel_family(
        self,
        first: DrawingLine,
        second: DrawingLine,
        *,
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> bool:
        if self._angle_diff(first.angle_deg, second.angle_deg) > self.ANGLE_TOLERANCE_DEG:
            return False
        length_ratio = min(first.length_px, second.length_px) / max(first.length_px, second.length_px)
        if length_ratio < 0.48:
            return False
        if self._overlap_ratio(first, second) < 0.48:
            return False

        min_dim = metric.reference_min_dim_native_px
        distance = self._normal_distance(first, second)
        max_distance = min(
            min_dim * 0.12,
            max(first.length_px, second.length_px) * 0.85,
        )
        return distance <= max(metric.to_native_px(8.0), max_distance)

    def _regular_profiles(
        self,
        lines: Sequence[DrawingLine],
        *,
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> list[_Profile]:
        if len(lines) < 3:
            return []
        ref = max(lines, key=lambda item: item.length_px)
        angle = math.radians(ref.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux

        raw_tracks: list[_ObservedTrack] = []
        for line in lines:
            midpoint_x = (line.start.x + line.end.x) / 2.0
            midpoint_y = (line.start.y + line.end.y) / 2.0
            cross = midpoint_x * nx + midpoint_y * ny
            values = [
                line.start.x * ux + line.start.y * uy,
                line.end.x * ux + line.end.y * uy,
            ]
            ids = set(line.metadata.get("context_equivalent_line_ids") or [])
            ids.update(line.metadata.get("physical_stroke_member_line_ids") or [])
            ids.update(str(item) for item in line.evidence_ids if item)
            ids.add(line.id)
            raw_tracks.append(
                _ObservedTrack(
                    line=line,
                    line_ids=tuple(sorted(ids)),
                    cross=float(cross),
                    lo=float(min(values)),
                    hi=float(max(values)),
                )
            )

        min_dim = metric.reference_min_dim_native_px
        merge_tol = max(metric.to_native_px(1.5), min_dim * 0.0014)
        tracks: list[_ObservedTrack] = []
        for track in sorted(raw_tracks, key=lambda item: item.cross):
            if tracks and abs(track.cross - tracks[-1].cross) <= merge_tol:
                previous = tracks[-1]
                chosen = track if track.line.length_px > previous.line.length_px else previous
                tracks[-1] = _ObservedTrack(
                    line=chosen.line,
                    line_ids=tuple(sorted(set(previous.line_ids) | set(track.line_ids))),
                    cross=(previous.cross + track.cross) / 2.0,
                    lo=min(previous.lo, track.lo),
                    hi=max(previous.hi, track.hi),
                )
            else:
                tracks.append(track)

        if len(tracks) < 3:
            return []

        candidates: list[tuple[float, int, int, _Profile]] = []
        for start in range(0, len(tracks) - 2):
            for end in range(start + 3, len(tracks) + 1):
                subset = tracks[start:end]
                profile = self._profile_from_tracks(
                    subset,
                    angle_deg=ref.angle_deg,
                    drawing=drawing,
                    metric=metric,
                )
                if profile is None:
                    continue
                # Favorece la secuencia más larga y después la más regular.
                rank = len(subset) * 10.0 + profile.confidence
                candidates.append((rank, start, end, profile))

        if not candidates:
            return []

        # Selección de runs máximos sin reutilizar los mismos tracks en perfiles
        # casi idénticos. Esto mantiene la salida interpretable y estable.
        used: set[int] = set()
        output: list[_Profile] = []
        for _, start, end, profile in sorted(candidates, key=lambda item: -item[0]):
            indexes = set(range(start, end))
            if len(indexes & used) / len(indexes) > 0.55:
                continue
            output.append(profile)
            used.update(indexes)
        return output

    def _profile_from_tracks(
        self,
        tracks: Sequence[_ObservedTrack],
        *,
        angle_deg: float,
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> _Profile | None:
        if len(tracks) < 3:
            return None
        coords = [item.cross for item in tracks]
        gaps = [coords[index + 1] - coords[index] for index in range(len(coords) - 1)]
        if any(gap <= 1e-6 for gap in gaps):
            return None
        spacing = float(median(gaps))
        min_dim = metric.reference_min_dim_native_px
        minimum_spacing = max(metric.to_native_px(1.5), min_dim * 0.0015)
        if spacing < minimum_spacing:
            return None

        mean_gap = sum(gaps) / len(gaps)
        variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
        spacing_cv = math.sqrt(variance) / max(mean_gap, 1e-6)
        if spacing_cv > 0.30:
            return None
        if any(gap < 0.50 * spacing or gap > 1.65 * spacing for gap in gaps):
            return None

        lengths = [item.hi - item.lo for item in tracks]
        median_length = float(median(lengths))
        if median_length <= 1e-6:
            return None
        spacing_to_length = spacing / median_length
        if spacing_to_length > 0.90:
            return None

        common_lo = max(item.lo for item in tracks)
        common_hi = min(item.hi for item in tracks)
        common_length = max(0.0, common_hi - common_lo)
        common_span_ratio = common_length / median_length
        if common_span_ratio < 0.45:
            return None

        mean_length = sum(lengths) / len(lengths)
        length_variance = sum((value - mean_length) ** 2 for value in lengths) / len(lengths)
        length_cv = math.sqrt(length_variance) / max(mean_length, 1e-6)
        length_similarity = max(0.0, min(1.0, 1.0 - length_cv / 0.40))
        if length_similarity < 0.48:
            return None

        spacing_score = max(0.0, min(1.0, 1.0 - spacing_cv / 0.30))
        span_score = max(0.0, min(1.0, common_span_ratio / 0.78))
        count_score = max(0.0, min(1.0, (len(tracks) - 2) / 6.0))
        density_score = max(0.0, min(1.0, (0.75 - spacing_to_length) / 0.55))
        confidence = (
            0.32 * spacing_score
            + 0.24 * span_score
            + 0.18 * length_similarity
            + 0.14 * count_score
            + 0.12 * density_score
        )

        member_ids = sorted({line_id for track in tracks for line_id in track.line_ids})
        profile_model = RepetitiveAxisProfile(
            angle_deg=float(angle_deg % 180.0),
            spacing_px=spacing,
            spacing_cv=float(spacing_cv),
            member_count=len(tracks),
            median_length_px=median_length,
            common_span_ratio=float(max(0.0, min(1.0, common_span_ratio))),
            length_similarity=float(length_similarity),
            track_coordinates=[float(value) for value in coords],
            member_line_ids=member_ids,
        )
        return _Profile(
            model=profile_model,
            bbox=self._tracks_bbox(tracks),
            confidence=float(max(0.0, min(1.0, confidence))),
            tracks=tuple(tracks),
            spacing_to_length=float(spacing_to_length),
        )

    def _parallel_region(self, *, profile: _Profile, drawing: DrawingModel) -> ContextRegion:
        semantic_support = self._stair_semantic_support(profile.bbox, drawing=drawing)
        stair_signature = (
            min(
                abs(profile.model.angle_deg % 180.0),
                abs((profile.model.angle_deg % 180.0) - 90.0),
                abs((profile.model.angle_deg % 180.0) - 180.0),
            ) <= 8.0
            and profile.model.member_count >= 7
            and profile.spacing_to_length <= 0.35
            and profile.model.spacing_cv <= 0.22
        )
        region_type = (
            "STAIR_FLIGHT_REGION"
            if semantic_support >= 0.45 or stair_signature
            else "REPETITIVE_PARALLEL_REGION"
        )
        confidence = min(1.0, profile.confidence + 0.16 * semantic_support)

        # Un patrón paralelo aislado solo puede cuarentenar cuando es denso y
        # tiene al menos cinco tracks. Tres líneas regulares quedan disponibles
        # como evidencia para formar una retícula, pero no excluyen por sí solas.
        quarantine_enabled = (
            profile.model.member_count >= 5
            and profile.spacing_to_length <= 0.42
            and confidence >= 0.78
        ) or (
            semantic_support >= 0.60
            and profile.model.member_count >= 4
            and confidence >= 0.74
        )

        ident = self._region_id(region_type, [profile.model])
        return ContextRegion(
            id=ident,
            region_type=region_type,
            bbox=self._pad_bbox(profile.bbox, 0.30 * profile.model.spacing_px, drawing=drawing),
            confidence=confidence,
            quarantine_enabled=quarantine_enabled,
            profiles=[profile.model],
            member_line_ids=list(profile.model.member_line_ids),
            semantic_support=semantic_support,
            metadata={
                "detector": "DRAWING_MODEL_REPETITION_V5",
                "spacing_to_length": profile.spacing_to_length,
                "source": "RAW_PRIMITIVE_PATTERN",
            },
        )

    def _grid_regions(
        self,
        *,
        profiles: Sequence[_Profile],
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        for index, first in enumerate(profiles):
            if first.model.member_count < 3:
                continue
            for second in profiles[index + 1 :]:
                if second.model.member_count < 3:
                    continue
                orthogonal_error = abs(90.0 - self._angle_diff(first.model.angle_deg, second.model.angle_deg))
                if orthogonal_error > self.GRID_ORTHOGONAL_TOLERANCE_DEG:
                    continue
                intersection = self._bbox_intersection(first.bbox, second.bbox)
                if intersection is None:
                    continue
                if self._bbox_area(intersection) <= 1.0:
                    continue

                crossing_ratio = self._crossing_ratio(first, second, metric=metric)
                if crossing_ratio < 0.45:
                    continue

                confidence = max(
                    0.0,
                    min(
                        1.0,
                        0.33 * first.confidence
                        + 0.33 * second.confidence
                        + 0.34 * crossing_ratio,
                    ),
                )
                quarantine_enabled = (
                    confidence >= 0.76
                    and crossing_ratio >= 0.52
                    and first.model.spacing_cv <= 0.28
                    and second.model.spacing_cv <= 0.28
                )
                profiles_model = [first.model, second.model]
                ident = self._region_id("REPETITIVE_GRID_REGION", profiles_model)
                member_ids = sorted(
                    set(first.model.member_line_ids) | set(second.model.member_line_ids)
                )
                pad = 0.18 * min(first.model.spacing_px, second.model.spacing_px)
                output.append(
                    ContextRegion(
                        id=ident,
                        region_type="REPETITIVE_GRID_REGION",
                        bbox=self._pad_bbox(intersection, pad, drawing=drawing),
                        confidence=confidence,
                        quarantine_enabled=quarantine_enabled,
                        profiles=profiles_model,
                        member_line_ids=member_ids,
                        semantic_support=0.0,
                        metadata={
                            "detector": "DRAWING_MODEL_REPETITION_V5",
                            "crossing_ratio": crossing_ratio,
                            "orthogonal_error_deg": orthogonal_error,
                            "source": "RAW_PRIMITIVE_GRID",
                        },
                    )
                )

        # Dedupe de retículas equivalentes.
        unique: list[ContextRegion] = []
        for region in sorted(output, key=lambda item: -item.confidence):
            if any(
                self._bbox_overlap_ratio(region.bbox, existing.bbox) >= 0.82
                and len(set(region.member_line_ids) & set(existing.member_line_ids))
                / max(1, min(len(region.member_line_ids), len(existing.member_line_ids)))
                >= 0.72
                for existing in unique
            ):
                continue
            unique.append(region)
        return unique

    def _crossing_ratio(
        self,
        first: _Profile,
        second: _Profile,
        *,
        metric: MetricNormalizedContext,
    ) -> float:
        total = len(first.tracks) * len(second.tracks)
        if total <= 0:
            return 0.0
        tolerance = max(metric.to_native_px(1.5), 0.12 * min(first.model.spacing_px, second.model.spacing_px))
        count = 0
        for a in first.tracks:
            line_a = self._line(a.line)
            for b in second.tracks:
                line_b = self._line(b.line)
                if line_a.distance(line_b) <= tolerance:
                    count += 1
        return count / total

    def _stair_semantic_support(self, bbox: DrawingBBox, *, drawing: DrawingModel) -> float:
        tokens = ("stair", "stairs", "staircase", "step", "steps", "escalera", "escaleras", "pelda")
        best = 0.0
        for observation in drawing.semantic_observations:
            text = " ".join(
                [
                    observation.family or "",
                    observation.text or "",
                    str(observation.payload),
                ]
            ).lower()
            if not any(token in text for token in tokens):
                continue
            confidence = observation.confidence if observation.confidence is not None else 0.75
            # Una observación semántica sin geometría solo demuestra que el
            # objeto existe en el LevelView; no puede localizar esta región.
            if observation.bbox is None:
                continue
            overlap = self._bbox_overlap_ratio(bbox, observation.bbox)
            if overlap > 0.0:
                best = max(best, min(1.0, confidence * (0.55 + 0.45 * overlap)))
        return max(0.0, min(1.0, best))

    def _wall_protected_regions(
        self,
        *,
        perimeter: EditablePerimeterModel,
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> list[ContextRegion]:
        vertices = {vertex.id: vertex.point_px for vertex in perimeter.vertices}
        min_dim = metric.reference_min_dim_native_px
        tolerance = max(metric.to_native_px(3.0), 0.006 * min_dim)
        output: list[ContextRegion] = []
        for wall in perimeter.walls:
            start = vertices.get(wall.start_vertex_id)
            end = vertices.get(wall.end_vertex_id)
            if start is None or end is None:
                continue
            dx = float(end.x - start.x)
            dy = float(end.y - start.y)
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            bbox = DrawingBBox(
                x_min=max(0.0, min(start.x, end.x) - tolerance),
                y_min=max(0.0, min(start.y, end.y) - tolerance),
                x_max=min(float(drawing.width_px), max(start.x, end.x) + tolerance),
                y_max=min(float(drawing.height_px), max(start.y, end.y) + tolerance),
            )
            digest = hashlib.sha1(
                f"WALL_PROTECTED|{wall.id}|{start.x:.3f}|{start.y:.3f}|{end.x:.3f}|{end.y:.3f}".encode()
            ).hexdigest()[:16]
            output.append(
                ContextRegion(
                    id=f"CTX__WALL_PROTECTED_REGION__{digest}",
                    region_type="WALL_PROTECTED_REGION",
                    bbox=bbox,
                    confidence=1.0,
                    quarantine_enabled=False,
                    profiles=[],
                    member_line_ids=list(wall.evidence_ids),
                    semantic_support=0.0,
                    metadata={
                        "detector": "F02_WALL_PROTECTION_V6",
                        "wall_id": wall.id,
                        "segment": [
                            [float(start.x), float(start.y)],
                            [float(end.x), float(end.y)],
                        ],
                        "angle_deg": angle,
                        "tolerance_px": tolerance,
                        "source": "F02_IMMUTABLE_PERIMETER",
                    },
                )
            )
        return output

    @staticmethod
    def _attach_scale_metadata(
        region: ContextRegion,
        *,
        metric: MetricNormalizedContext,
    ) -> ContextRegion:
        profile = metric.profile
        metadata = dict(region.metadata)
        metadata.update(
            {
                "scale_normalization": "PROJECT_LEVEL_SCALE_V6_1",
                "scale_resolved": metric.resolved,
                "scale_factor_to_canonical": metric.scale_factor,
                "local_m_per_px": metric.local_m_per_px,
                "project_reference_min_dim_native_px": metric.reference_min_dim_native_px,
            }
        )
        if profile is not None:
            metadata["scale_state"] = profile.state
            metadata["canonical_m_per_px"] = profile.canonical_m_per_px
            metadata["project_reference_min_dim_normalized_px"] = (
                profile.project_reference_min_dim_normalized_px
            )
        return region.model_copy(update={"metadata": metadata})

    def _dedupe_context_regions(self, regions: Sequence[ContextRegion]) -> list[ContextRegion]:
        # Prioridad semántica: una celda detectada directamente es más informativa
        # que una retícula inferida; WALL_PROTECTED siempre se conserva.
        priority = {
            "WALL_PROTECTED_REGION": 100,
            "FLOOR_FINISH_GRID_REGION": 90,
            "FURNITURE_MODULE_REGION": 85,
            "STAIR_FLIGHT_REGION": 80,
            "HATCH_FILL_REGION": 75,
            "AXIS_GRID_REGION": 70,
            "DIMENSION_REGION": 70,
            "SYMBOL_FIXTURE_REGION": 65,
            "UNKNOWN_REPETITIVE_REGION": 40,
            "REPETITIVE_GRID_REGION": 30,
            "REPETITIVE_PARALLEL_REGION": 20,
        }
        output: list[ContextRegion] = []
        def region_priority(item: ContextRegion) -> tuple[int, int, float, str]:
            detector = str(item.metadata.get("detector", ""))
            direct_surface = 2 if detector == "SURFACE_PATTERN_DETECTOR_V6_2" else 0
            direct_cell = 1 if detector == "REPEATED_CELL_DETECTOR_V6" else 0
            return (
                -priority.get(item.region_type, 0),
                -(direct_surface + direct_cell),
                -item.confidence,
                item.id,
            )

        ordered = sorted(regions, key=region_priority)
        for region in ordered:
            if region.region_type == "WALL_PROTECTED_REGION":
                output.append(region)
                continue
            duplicate = False
            for existing in output:
                if existing.region_type == "WALL_PROTECTED_REGION":
                    continue
                overlap = self._bbox_overlap_ratio(region.bbox, existing.bbox)
                if overlap < 0.78:
                    continue
                shared = len(set(region.member_line_ids) & set(existing.member_line_ids))
                denom = max(1, min(len(region.member_line_ids), len(existing.member_line_ids)))
                if shared / denom >= 0.45 or region.region_type == existing.region_type:
                    duplicate = True
                    break
            if not duplicate:
                output.append(region)
        return output

    @staticmethod
    def _region_id(region_type: str, profiles: Sequence[RepetitiveAxisProfile]) -> str:
        payload = [region_type]
        for profile in profiles:
            payload.extend(
                [
                    f"{profile.angle_deg:.3f}",
                    f"{profile.spacing_px:.3f}",
                    ",".join(profile.member_line_ids),
                ]
            )
        digest = hashlib.sha1("|".join(payload).encode()).hexdigest()[:16]
        return f"CTX__{region_type}__{digest}"

    @staticmethod
    def _tracks_bbox(tracks: Sequence[_ObservedTrack]) -> DrawingBBox:
        xs = [coord for track in tracks for coord in (track.line.start.x, track.line.end.x)]
        ys = [coord for track in tracks for coord in (track.line.start.y, track.line.end.y)]
        return DrawingBBox(
            x_min=float(min(xs)),
            y_min=float(min(ys)),
            x_max=float(max(xs)),
            y_max=float(max(ys)),
        )

    @staticmethod
    def _pad_bbox(bbox: DrawingBBox, pad: float, *, drawing: DrawingModel) -> DrawingBBox:
        return DrawingBBox(
            x_min=max(0.0, bbox.x_min - pad),
            y_min=max(0.0, bbox.y_min - pad),
            x_max=min(float(drawing.width_px), bbox.x_max + pad),
            y_max=min(float(drawing.height_px), bbox.y_max + pad),
        )

    @staticmethod
    def _bbox_intersection(first: DrawingBBox, second: DrawingBBox) -> DrawingBBox | None:
        x_min = max(first.x_min, second.x_min)
        y_min = max(first.y_min, second.y_min)
        x_max = min(first.x_max, second.x_max)
        y_max = min(first.y_max, second.y_max)
        if x_max <= x_min or y_max <= y_min:
            return None
        return DrawingBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    @staticmethod
    def _bbox_area(bbox: DrawingBBox) -> float:
        return max(0.0, bbox.x_max - bbox.x_min) * max(0.0, bbox.y_max - bbox.y_min)

    @classmethod
    def _bbox_overlap_ratio(cls, first: DrawingBBox, second: DrawingBBox) -> float:
        intersection = cls._bbox_intersection(first, second)
        if intersection is None:
            return 0.0
        area = cls._bbox_area(intersection)
        denom = max(1e-6, min(cls._bbox_area(first), cls._bbox_area(second)))
        return max(0.0, min(1.0, area / denom))

    @staticmethod
    def _line(line: DrawingLine) -> LineString:
        return LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _normal_distance(first: DrawingLine, second: DrawingLine) -> float:
        angle = math.radians(first.angle_deg)
        nx, ny = -math.sin(angle), math.cos(angle)
        first_mid = ((first.start.x + first.end.x) / 2.0, (first.start.y + first.end.y) / 2.0)
        second_mid = ((second.start.x + second.end.x) / 2.0, (second.start.y + second.end.y) / 2.0)
        return abs(first_mid[0] * nx + first_mid[1] * ny - second_mid[0] * nx - second_mid[1] * ny)

    @staticmethod
    def _overlap_ratio(first: DrawingLine, second: DrawingLine) -> float:
        angle = math.radians(first.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        a = [first.start.x * ux + first.start.y * uy, first.end.x * ux + first.end.y * uy]
        b = [second.start.x * ux + second.start.y * uy, second.end.x * ux + second.end.y * uy]
        a0, a1 = min(a), max(a)
        b0, b1 = min(b), max(b)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        return max(0.0, min(1.0, overlap / max(1e-6, min(a1 - a0, b1 - b0))))
