from __future__ import annotations

import hashlib
import math
from statistics import median
from typing import Sequence

from shapely.geometry import box

from .context_models import ContextRegion, RepetitiveAxisProfile
from .drawing_model import DrawingBBox, DrawingModel
from app.quantia_spatialV1.core.scale.metric_normalized_context import MetricNormalizedContext


class ContextRegionAssembler:
    """Une micro-regiones periódicas que pertenecen al mismo objeto gráfico.

    El detector local puede producir varios runs de una misma escalera cuando
    existen puertas, símbolos, cambios de fuente o cortes de raster. Esta capa
    ensambla únicamente regiones compatibles por orientación, modulación y
    proximidad. No crea candidatos de muro ni modifica A1.2.
    """

    ANGLE_TOLERANCE_DEG = 5.0
    SPACING_RELATIVE_TOLERANCE = 0.28

    def assemble(
        self,
        *,
        regions: Sequence[ContextRegion],
        drawing: DrawingModel,
        scale_context: MetricNormalizedContext | None = None,
    ) -> list[ContextRegion]:
        if not regions:
            return []

        components: list[list[ContextRegion]] = []
        visited: set[str] = set()
        by_id = {region.id: region for region in regions}

        for region in regions:
            if region.id in visited:
                continue
            queue = [region]
            visited.add(region.id)
            component: list[ContextRegion] = []
            while queue:
                current = queue.pop()
                component.append(current)
                for other in regions:
                    if other.id in visited:
                        continue
                    if not self._compatible(current, other, drawing=drawing, scale_context=scale_context):
                        continue
                    visited.add(other.id)
                    queue.append(by_id[other.id])
            components.append(component)

        assembled = [
            self._merge_component(component=component, drawing=drawing, scale_context=scale_context)
            if len(component) > 1
            else component[0]
            for component in components
        ]
        return sorted(
            assembled,
            key=lambda item: (
                not item.quarantine_enabled,
                -item.confidence,
                item.region_type,
                item.id,
            ),
        )

    def _compatible(
        self,
        first: ContextRegion,
        second: ContextRegion,
        *,
        drawing: DrawingModel,
        scale_context: MetricNormalizedContext | None = None,
    ) -> bool:
        if first.region_type != second.region_type:
            return False
        if first.region_type == "REPETITIVE_GRID_REGION":
            # Las retículas ya se deduplican en el detector; fusionarlas a larga
            # distancia puede absorber habitaciones reales adyacentes.
            return self._bbox_overlap_ratio(first.bbox, second.bbox) >= 0.55

        first_profile = self._dominant_profile(first)
        second_profile = self._dominant_profile(second)
        if self._angle_diff(first_profile.angle_deg, second_profile.angle_deg) > self.ANGLE_TOLERANCE_DEG:
            return False

        spacing_ratio = min(first_profile.spacing_px, second_profile.spacing_px) / max(
            first_profile.spacing_px,
            second_profile.spacing_px,
        )
        if spacing_ratio < 1.0 - self.SPACING_RELATIVE_TOLERANCE:
            return False

        spacing = float(median([first_profile.spacing_px, second_profile.spacing_px]))
        metric = scale_context or MetricNormalizedContext(drawing=drawing)
        min_dim = metric.reference_min_dim_native_px
        pad = max(1.6 * spacing, min_dim * 0.006)
        a = box(first.bbox.x_min, first.bbox.y_min, first.bbox.x_max, first.bbox.y_max)
        b = box(second.bbox.x_min, second.bbox.y_min, second.bbox.x_max, second.bbox.y_max)
        if not a.buffer(pad, cap_style=2, join_style=2).intersects(b):
            return False

        # Debe existir solape longitudinal o continuidad inmediata. Esto evita
        # unir dos patrones de igual separación situados en zonas distintas.
        angle = math.radians(first_profile.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        first_interval = self._bbox_projection(first.bbox, ux=ux, uy=uy)
        second_interval = self._bbox_projection(second.bbox, ux=ux, uy=uy)
        longitudinal_overlap = self._interval_overlap_ratio(first_interval, second_interval)
        longitudinal_gap = self._interval_gap(first_interval, second_interval)
        median_length = float(median([first_profile.median_length_px, second_profile.median_length_px]))
        return longitudinal_overlap >= 0.34 or longitudinal_gap <= max(1.5 * spacing, 0.18 * median_length)

    def _merge_component(
        self,
        *,
        component: Sequence[ContextRegion],
        drawing: DrawingModel,
        scale_context: MetricNormalizedContext | None = None,
    ) -> ContextRegion:
        profiles = self._dedupe_profiles(
            [profile for region in component for profile in region.profiles]
        )
        member_ids = sorted({line_id for region in component for line_id in region.member_line_ids})
        bbox = DrawingBBox(
            x_min=min(region.bbox.x_min for region in component),
            y_min=min(region.bbox.y_min for region in component),
            x_max=max(region.bbox.x_max for region in component),
            y_max=max(region.bbox.y_max for region in component),
        )
        confidence = max(region.confidence for region in component)
        semantic_support = max(region.semantic_support for region in component)

        aggregate_track_count = self._aggregate_track_count(profiles, scale_context=scale_context)
        dominant = self._dominant_profile_from_profiles(profiles)
        spacing_to_length = dominant.spacing_px / max(dominant.median_length_px, 1e-6)

        quarantine_enabled = any(region.quarantine_enabled for region in component)
        if component[0].region_type == "STAIR_FLIGHT_REGION":
            quarantine_enabled = quarantine_enabled or (
                aggregate_track_count >= 6
                and spacing_to_length <= 0.42
                and confidence >= 0.76
            )
        elif component[0].region_type == "REPETITIVE_PARALLEL_REGION":
            quarantine_enabled = quarantine_enabled or (
                aggregate_track_count >= 7
                and spacing_to_length <= 0.36
                and confidence >= 0.80
            )

        payload = "|".join(sorted(region.id for region in component))
        digest = hashlib.sha1(payload.encode()).hexdigest()[:16]
        return ContextRegion(
            id=f"CTX_ASSEMBLED__{component[0].region_type}__{digest}",
            region_type=component[0].region_type,
            bbox=bbox,
            confidence=float(max(0.0, min(1.0, confidence))),
            quarantine_enabled=bool(quarantine_enabled),
            profiles=profiles,
            member_line_ids=member_ids,
            semantic_support=float(max(0.0, min(1.0, semantic_support))),
            metadata={
                "detector": "CONTEXT_REGION_ASSEMBLER_V5",
                "source_region_ids": sorted(region.id for region in component),
                "source_region_count": len(component),
                "aggregate_track_count": aggregate_track_count,
                "spacing_to_length": spacing_to_length,
                "source": "ASSEMBLED_PERIODIC_CONTEXT",
            },
        )

    def _aggregate_track_count(
        self,
        profiles: Sequence[RepetitiveAxisProfile],
        *,
        scale_context: MetricNormalizedContext | None = None,
    ) -> int:
        if not profiles:
            return 0
        dominant = self._dominant_profile_from_profiles(profiles)
        coordinates: list[float] = []
        for profile in profiles:
            if self._angle_diff(profile.angle_deg, dominant.angle_deg) > self.ANGLE_TOLERANCE_DEG:
                continue
            if min(profile.spacing_px, dominant.spacing_px) / max(profile.spacing_px, dominant.spacing_px) < 0.72:
                continue
            coordinates.extend(profile.track_coordinates)
        if not coordinates:
            return 0
        min_tol = scale_context.to_native_px(1.0) if scale_context is not None else 1.0
        tolerance = max(min_tol, 0.30 * dominant.spacing_px)
        collapsed: list[float] = []
        for value in sorted(coordinates):
            if collapsed and abs(value - collapsed[-1]) <= tolerance:
                collapsed[-1] = (collapsed[-1] + value) / 2.0
            else:
                collapsed.append(value)
        return len(collapsed)

    def _dedupe_profiles(
        self,
        profiles: Sequence[RepetitiveAxisProfile],
    ) -> list[RepetitiveAxisProfile]:
        output: list[RepetitiveAxisProfile] = []
        for profile in sorted(profiles, key=lambda item: (-item.member_count, item.spacing_cv)):
            duplicate = False
            for existing in output:
                if self._angle_diff(profile.angle_deg, existing.angle_deg) > self.ANGLE_TOLERANCE_DEG:
                    continue
                spacing_ratio = min(profile.spacing_px, existing.spacing_px) / max(
                    profile.spacing_px,
                    existing.spacing_px,
                )
                if spacing_ratio < 0.82:
                    continue
                shared = len(set(profile.member_line_ids) & set(existing.member_line_ids))
                denom = max(1, min(len(profile.member_line_ids), len(existing.member_line_ids)))
                if shared / denom >= 0.72:
                    duplicate = True
                    break
            if not duplicate:
                output.append(profile)
        return output

    @staticmethod
    def _dominant_profile(region: ContextRegion) -> RepetitiveAxisProfile:
        return max(region.profiles, key=lambda item: (item.member_count, -item.spacing_cv))

    @staticmethod
    def _dominant_profile_from_profiles(
        profiles: Sequence[RepetitiveAxisProfile],
    ) -> RepetitiveAxisProfile:
        return max(profiles, key=lambda item: (item.member_count, -item.spacing_cv))

    @staticmethod
    def _bbox_projection(bbox: DrawingBBox, *, ux: float, uy: float) -> tuple[float, float]:
        points = (
            (bbox.x_min, bbox.y_min),
            (bbox.x_min, bbox.y_max),
            (bbox.x_max, bbox.y_min),
            (bbox.x_max, bbox.y_max),
        )
        values = [x * ux + y * uy for x, y in points]
        return min(values), max(values)

    @staticmethod
    def _interval_overlap_ratio(first: tuple[float, float], second: tuple[float, float]) -> float:
        overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
        return overlap / max(1e-6, min(first[1] - first[0], second[1] - second[0]))

    @staticmethod
    def _interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
        if first[1] < second[0]:
            return second[0] - first[1]
        if second[1] < first[0]:
            return first[0] - second[1]
        return 0.0

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _bbox_overlap_ratio(first: DrawingBBox, second: DrawingBBox) -> float:
        x_min = max(first.x_min, second.x_min)
        y_min = max(first.y_min, second.y_min)
        x_max = min(first.x_max, second.x_max)
        y_max = min(first.y_max, second.y_max)
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        intersection = (x_max - x_min) * (y_max - y_min)
        a = max(1e-6, (first.x_max - first.x_min) * (first.y_max - first.y_min))
        b = max(1e-6, (second.x_max - second.x_min) * (second.y_max - second.y_min))
        return max(0.0, min(1.0, intersection / min(a, b)))
