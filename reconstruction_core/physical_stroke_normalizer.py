from __future__ import annotations

import hashlib
import math
from statistics import median
from typing import Sequence

from shapely.geometry import LineString

from .drawing_model import DrawingLine, DrawingModel, DrawingPoint
from .metric_normalized_context import MetricNormalizedContext


class PhysicalStrokeNormalizer:
    """Colapsa múltiples caras raster de un mismo trazo gráfico.

    Esta capa SOLO se usa para detectar contexto repetitivo. No altera el
    DrawingModel original ni los WallCandidate de A1.2.

    El objetivo es evitar que las dos caras producidas por antialiasing, LSD o
    contornos de una misma línea se interpreten como dos peldaños/miembros de una
    retícula. La distancia de colapso se deriva de la escala del LevelView y se
    limita a una fracción pequeña del dibujo; no intenta estimar espesor de muro.
    """

    ANGLE_TOLERANCE_DEG = 2.75
    MIN_OVERLAP_RATIO = 0.72
    MICRO_GAP_RATIO = 0.0080
    MICRO_GAP_MIN_PX = 1.5

    def normalize(
        self,
        *,
        lines: Sequence[DrawingLine],
        drawing: DrawingModel,
        scale_context: MetricNormalizedContext | None = None,
    ) -> list[DrawingLine]:
        if not lines:
            return []

        micro_gap = self._micro_gap_threshold(lines=lines, drawing=drawing, scale_context=scale_context)
        groups: list[list[DrawingLine]] = []

        # El orden por longitud hace estable la agrupación y favorece que una
        # detección completa represente al trazo frente a fragmentos menores.
        for line in sorted(lines, key=lambda item: (-item.length_px, item.id)):
            target: list[DrawingLine] | None = None
            for group in groups:
                if self._belongs_to_group(line=line, group=group, micro_gap=micro_gap):
                    target = group
                    break
            if target is None:
                groups.append([line])
            else:
                target.append(line)

        return [
            self._collapse_group(group=group, micro_gap=micro_gap)
            for group in groups
        ]

    def _belongs_to_group(
        self,
        *,
        line: DrawingLine,
        group: Sequence[DrawingLine],
        micro_gap: float,
    ) -> bool:
        # Comparamos con varios miembros para no depender de una única cara que
        # haya quedado ligeramente desplazada por rasterización.
        for reference in group[:4]:
            if self._angle_diff(line.angle_deg, reference.angle_deg) > self.ANGLE_TOLERANCE_DEG:
                continue
            if self._normal_distance(line, reference) > micro_gap:
                continue
            if self._overlap_ratio(line, reference) < self.MIN_OVERLAP_RATIO:
                continue
            return True
        return False

    def _micro_gap_threshold(
        self,
        *,
        lines: Sequence[DrawingLine],
        drawing: DrawingModel,
        scale_context: MetricNormalizedContext | None = None,
    ) -> float:
        metric = scale_context or MetricNormalizedContext(drawing=drawing)
        min_dim = metric.reference_min_dim_native_px
        min_px = metric.to_native_px(self.MICRO_GAP_MIN_PX)
        scale_ceiling = max(min_px, min_dim * self.MICRO_GAP_RATIO)

        # Si existen anchos de trazo explícitos, los usamos como evidencia de la
        # rasterización, pero nunca permitimos que amplíen el límite de escala.
        widths = [
            float(line.stroke_width_px)
            for line in lines
            if line.stroke_width_px is not None and line.stroke_width_px > 0.0
        ]
        if not widths:
            return scale_ceiling

        graphic_width = float(median(widths))
        return max(
            min_px,
            min(scale_ceiling, max(1.5 * graphic_width, min_px)),
        )

    def _collapse_group(
        self,
        *,
        group: Sequence[DrawingLine],
        micro_gap: float,
    ) -> DrawingLine:
        if len(group) == 1:
            output = group[0].model_copy(deep=True)
            members = set(output.metadata.get("physical_stroke_member_line_ids") or [])
            members.add(output.id)
            output.metadata["physical_stroke_member_line_ids"] = sorted(members)
            output.metadata["physical_stroke_collapsed_count"] = 1
            output.metadata["physical_stroke_micro_gap_px"] = float(micro_gap)
            return output

        reference = max(group, key=lambda item: item.length_px)
        angle = math.radians(reference.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux

        projected: list[tuple[float, float, float, float]] = []
        for line in group:
            starts = line.start.x * ux + line.start.y * uy
            ends = line.end.x * ux + line.end.y * uy
            lo, hi = min(starts, ends), max(starts, ends)
            mx = (line.start.x + line.end.x) / 2.0
            my = (line.start.y + line.end.y) / 2.0
            cross = mx * nx + my * ny
            projected.append((lo, hi, cross, line.length_px))

        lo = float(median(item[0] for item in projected))
        hi = float(median(item[1] for item in projected))
        cross = self._weighted_mean(
            [item[2] for item in projected],
            [item[3] for item in projected],
        )
        if hi <= lo + 1e-6:
            lo = min(item[0] for item in projected)
            hi = max(item[1] for item in projected)

        start = DrawingPoint(x=lo * ux + cross * nx, y=lo * uy + cross * ny)
        end = DrawingPoint(x=hi * ux + cross * nx, y=hi * uy + cross * ny)
        length = math.hypot(end.x - start.x, end.y - start.y)

        member_ids: set[str] = set()
        for line in group:
            member_ids.add(line.id)
            member_ids.update(line.metadata.get("physical_stroke_member_line_ids") or [])
            member_ids.update(line.metadata.get("context_equivalent_line_ids") or [])

        payload = "|".join(sorted(member_ids))
        digest = hashlib.sha1(payload.encode()).hexdigest()[:16]
        sources = sorted({source for line in group for source in line.sources})
        evidence_ids = sorted({evidence for line in group for evidence in line.evidence_ids})
        confidences = [line.confidence for line in group if line.confidence is not None]
        widths = [line.stroke_width_px for line in group if line.stroke_width_px is not None]

        metadata = dict(reference.metadata)
        metadata.update(
            {
                "physical_stroke_member_line_ids": sorted(member_ids),
                "physical_stroke_collapsed_count": len(group),
                "physical_stroke_micro_gap_px": float(micro_gap),
                "physical_stroke_normalizer": "V5",
            }
        )

        return DrawingLine(
            id=f"PHYSICAL_STROKE__{digest}",
            kind="LINE",
            start=start,
            end=end,
            length_px=max(length, 1e-6),
            angle_deg=float(reference.angle_deg % 180.0),
            stroke_width_px=float(median(widths)) if widths else reference.stroke_width_px,
            dashed=all(line.dashed for line in group),
            evidence_ids=evidence_ids,
            sources=sources or list(reference.sources),
            confidence=max(confidences) if confidences else None,
            metadata=metadata,
        )

    @staticmethod
    def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 1e-9:
            return float(sum(values) / max(1, len(values)))
        return float(
            sum(value * max(0.0, weight) for value, weight in zip(values, weights)) / total
        )

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _normal_distance(first: DrawingLine, second: DrawingLine) -> float:
        angle = math.radians(first.angle_deg)
        nx, ny = -math.sin(angle), math.cos(angle)
        a = ((first.start.x + first.end.x) / 2.0, (first.start.y + first.end.y) / 2.0)
        b = ((second.start.x + second.end.x) / 2.0, (second.start.y + second.end.y) / 2.0)
        return abs((a[0] - b[0]) * nx + (a[1] - b[1]) * ny)

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
