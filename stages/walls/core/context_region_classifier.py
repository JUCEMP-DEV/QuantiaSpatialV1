from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from statistics import median
from typing import Sequence

from shapely.geometry import LineString, Point, box

from .context_models import ContextRegion
from .drawing_model import DrawingBBox, DrawingLine, DrawingModel
from app.quantia_spatialV1.core.scale.metric_normalized_context import MetricNormalizedContext


class ContextRegionClassifier:
    """Clasificación funcional V6.1 de regiones gráficas.

    V6.1 mantiene las clases funcionales de V6, pero normaliza umbrales contra
    una referencia común de proyecto cuando F02 resolvió escala métrica. También
    limita DIMENSION_REGION a una cadena local texto -> línea de cota ->
    extensiones, evitando cajas que absorban gran parte de una planta.
    """

    AXIS_ANGLE_TOLERANCE_DEG = 5.0
    DIMENSION_PERPENDICULAR_TOLERANCE_DEG = 12.0

    def classify_repetitive(
        self,
        *,
        regions: Sequence[ContextRegion],
        drawing: DrawingModel,
        scale_context: MetricNormalizedContext | None = None,
    ) -> list[ContextRegion]:
        metric = scale_context or MetricNormalizedContext(drawing=drawing)
        output: list[ContextRegion] = []
        for region in regions:
            if region.region_type == "STAIR_FLIGHT_REGION":
                output.append(self._with_detector(region, "CONTEXT_CLASSIFIER_V6_1"))
                continue
            if region.region_type == "REPETITIVE_GRID_REGION":
                output.append(self._classify_grid(region, drawing=drawing, metric=metric))
                continue
            if region.region_type == "REPETITIVE_PARALLEL_REGION":
                output.append(self._classify_parallel(region, drawing=drawing, metric=metric))
                continue
            output.append(region)
        return output

    def detect_reference_regions(
        self,
        *,
        drawing: DrawingModel,
        polygon,
        scale_context: MetricNormalizedContext | None = None,
    ) -> list[ContextRegion]:
        metric = scale_context or MetricNormalizedContext(drawing=drawing)
        return [
            *self._axis_grid_regions(drawing=drawing, polygon=polygon, metric=metric),
            *self._dimension_regions(drawing=drawing, polygon=polygon, metric=metric),
        ]

    def _classify_parallel(
        self,
        region: ContextRegion,
        *,
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> ContextRegion:
        if not region.profiles:
            return region.model_copy(
                update={"region_type": "UNKNOWN_REPETITIVE_REGION", "quarantine_enabled": False}
            )
        profile = max(region.profiles, key=lambda item: (item.member_count, -item.spacing_cv))
        spacing_to_length = profile.spacing_px / max(profile.median_length_px, 1e-6)
        diagonal = self._distance_to_axis(profile.angle_deg) >= 12.0
        compact = self._bbox_area(region.bbox) / max(1.0, metric.reference_area_native_px2) <= 0.18

        if (
            diagonal
            and profile.member_count >= 5
            and spacing_to_length <= 0.32
            and profile.spacing_cv <= 0.24
            and compact
        ):
            confidence = min(
                1.0,
                0.72 + 0.18 * region.confidence + 0.10 * min(1.0, profile.member_count / 10.0),
            )
            return region.model_copy(
                update={
                    "region_type": "HATCH_FILL_REGION",
                    "confidence": confidence,
                    "quarantine_enabled": confidence >= 0.78,
                    "metadata": {
                        **region.metadata,
                        "classifier": "CONTEXT_CLASSIFIER_V6_1",
                        "classification_reason": "DENSE_DIAGONAL_PARALLEL_PATTERN",
                    },
                }
            )

        stair_like = (
            self._distance_to_axis(profile.angle_deg) <= 6.0
            and profile.member_count >= 7
            and spacing_to_length <= 0.35
            and profile.spacing_cv <= 0.22
            and compact
        )
        if stair_like:
            confidence = max(region.confidence, 0.78)
            return region.model_copy(
                update={
                    "region_type": "STAIR_FLIGHT_REGION",
                    "confidence": confidence,
                    "quarantine_enabled": confidence >= 0.78,
                    "metadata": {
                        **region.metadata,
                        "classifier": "CONTEXT_CLASSIFIER_V6_1",
                        "classification_reason": "ORTHOGONAL_DENSE_STEP_PATTERN",
                    },
                }
            )

        return region.model_copy(
            update={
                "region_type": "UNKNOWN_REPETITIVE_REGION",
                "quarantine_enabled": False,
                "metadata": {
                    **region.metadata,
                    "classifier": "CONTEXT_CLASSIFIER_V6_1",
                    "classification_reason": "REPETITIVE_BUT_FUNCTION_UNRESOLVED",
                },
            }
        )

    def _classify_grid(
        self,
        region: ContextRegion,
        *,
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> ContextRegion:
        area_ratio = self._bbox_area(region.bbox) / max(1.0, metric.reference_area_native_px2)
        crossing = float(region.metadata.get("crossing_ratio", 0.0) or 0.0)
        member_count = sum(profile.member_count for profile in region.profiles)
        local = area_ratio <= 0.22

        if local and member_count >= 6 and crossing >= 0.45:
            confidence = min(
                1.0,
                0.55 * region.confidence
                + 0.25 * crossing
                + 0.20 * min(1.0, member_count / 12.0),
            )
            return region.model_copy(
                update={
                    "region_type": "FLOOR_FINISH_GRID_REGION",
                    "confidence": confidence,
                    "quarantine_enabled": confidence >= 0.76,
                    "metadata": {
                        **region.metadata,
                        "classifier": "CONTEXT_CLASSIFIER_V6_1",
                        "classification_reason": "LOCAL_ORTHOGONAL_REPETITIVE_GRID",
                    },
                }
            )

        return region.model_copy(
            update={
                "region_type": "UNKNOWN_REPETITIVE_REGION",
                "quarantine_enabled": False,
                "metadata": {
                    **region.metadata,
                    "classifier": "CONTEXT_CLASSIFIER_V6_1",
                    "classification_reason": "GRID_FUNCTION_UNRESOLVED",
                },
            }
        )

    def _axis_grid_regions(
        self,
        *,
        drawing: DrawingModel,
        polygon,
        metric: MetricNormalizedContext,
    ) -> list[ContextRegion]:
        min_dim = metric.reference_min_dim_native_px
        minimum_length = max(metric.to_native_px(30.0), 0.22 * min_dim)
        boundary_buffer = metric.to_native_px(2.0)
        candidates = [
            line
            for line in drawing.lines
            if line.kind == "LINE"
            and line.dashed
            and line.length_px >= minimum_length
            and self._line(line).intersection(polygon.buffer(boundary_buffer)).length
            / max(line.length_px, 1e-6)
            >= 0.35
        ]
        if len(candidates) < 2:
            return []

        bins: dict[int, list[DrawingLine]] = defaultdict(list)
        for line in candidates:
            bins[int(round(line.angle_deg / 5.0)) % 36].append(line)

        regions: list[ContextRegion] = []
        for lines in bins.values():
            if len(lines) < 2:
                continue
            angle = float(median([line.angle_deg for line in lines])) % 180.0
            bbox = self._lines_bbox(lines)
            extent = max(bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min)
            if extent < 0.30 * min_dim:
                continue
            member_ids = sorted({line.id for line in lines})
            confidence = min(0.96, 0.72 + 0.06 * min(4, len(lines)))
            digest = hashlib.sha1(("AXIS|" + "|".join(member_ids)).encode()).hexdigest()[:16]
            regions.append(
                ContextRegion(
                    id=f"CTX__AXIS_GRID_REGION__{digest}",
                    region_type="AXIS_GRID_REGION",
                    bbox=self._pad_bbox(
                        bbox,
                        max(metric.to_native_px(3.0), 0.006 * min_dim),
                        drawing=drawing,
                    ),
                    confidence=confidence,
                    quarantine_enabled=False,
                    profiles=[],
                    member_line_ids=member_ids,
                    semantic_support=0.0,
                    metadata={
                        "detector": "CONTEXT_CLASSIFIER_V6_1",
                        "angle_deg": angle,
                        "line_count": len(lines),
                        "source": "DASHED_LONG_REFERENCE_LINES",
                        "negative_mask": False,
                        "selection_contract": "AXIS_IS_LOCATOR_NOT_EXCLUSION_V1_4",
                    },
                )
            )
        return regions

    def _dimension_regions(
        self,
        *,
        drawing: DrawingModel,
        polygon,
        metric: MetricNormalizedContext,
    ) -> list[ContextRegion]:
        """Detecta cadenas locales de acotación sin propagar la bbox por cruces.

        V6 elegía la línea más larga cerca del texto y luego absorbía cualquier
        línea que tocara su buffer. En planos densos eso podía convertir una cota
        local en una región que cubría casi toda la planta. V6.1 selecciona el
        anchor por proximidad/proyección y solo admite extensiones aproximadamente
        perpendiculares cerca de sus extremos.
        """

        min_dim = metric.reference_min_dim_native_px
        text_gap = max(metric.to_native_px(8.0), 0.012 * min_dim)
        minimum_anchor = max(metric.to_native_px(18.0), 0.025 * min_dim)
        endpoint_radius = max(metric.to_native_px(10.0), 0.022 * min_dim)
        max_extension_length = max(metric.to_native_px(40.0), 0.34 * min_dim)
        bbox_pad = max(metric.to_native_px(3.0), 0.004 * min_dim)
        output: list[ContextRegion] = []

        for text in drawing.texts:
            if text.bbox is None or not self._dimension_text(text.text):
                continue

            text_geom = box(text.bbox.x_min, text.bbox.y_min, text.bbox.x_max, text.bbox.y_max)
            text_center = text_geom.centroid
            anchors = []
            for line in drawing.lines:
                if line.kind != "LINE" or line.length_px < minimum_anchor:
                    continue
                geom = self._line(line)
                distance = geom.distance(text_geom)
                if distance > text_gap:
                    continue
                projection = geom.project(text_center)
                center_offset = abs(projection - 0.5 * geom.length) / max(0.5 * geom.length, 1e-6)
                # El texto de una cota suele estar sobre/cerca de la propia línea,
                # no necesariamente en el centro exacto, pero una línea estructural
                # enorme que pasa cerca queda penalizada.
                rank = (
                    distance / max(text_gap, 1e-6)
                    + 0.35 * min(2.0, center_offset)
                    + 0.08 * min(3.0, line.length_px / max(min_dim, 1e-6))
                )
                anchors.append((rank, line))
            if not anchors:
                continue

            anchors.sort(key=lambda item: (item[0], item[1].length_px, item[1].id))
            accepted: tuple[DrawingLine, list[DrawingLine], float] | None = None
            for _, anchor in anchors[:8]:
                extensions = self._dimension_extension_lines(
                    anchor=anchor,
                    drawing=drawing,
                    endpoint_radius=endpoint_radius,
                    max_extension_length=max_extension_length,
                )
                if len(extensions) < 2 and not anchor.dashed:
                    continue
                accepted = (anchor, extensions, self._line(anchor).distance(text_geom))
                break
            if accepted is None:
                continue

            anchor, extensions, text_distance = accepted
            members = [anchor, *extensions]
            bbox = self._bounded_dimension_bbox(
                text_bbox=text.bbox,
                lines=members,
                drawing=drawing,
                pad=bbox_pad,
                min_dim=min_dim,
            )
            if bbox is None:
                continue

            member_ids = sorted({line.id for line in members})
            extension_score = min(1.0, len(extensions) / 2.0)
            proximity_score = max(0.0, 1.0 - text_distance / max(text_gap, 1e-6))
            confidence = min(0.94, 0.70 + 0.12 * extension_score + 0.08 * proximity_score)
            digest = hashlib.sha1((f"DIM|{text.id}|" + "|".join(member_ids)).encode()).hexdigest()[:16]
            output.append(
                ContextRegion(
                    id=f"CTX__DIMENSION_REGION__{digest}",
                    region_type="DIMENSION_REGION",
                    bbox=bbox,
                    confidence=confidence,
                    quarantine_enabled=confidence >= 0.80,
                    profiles=[],
                    member_line_ids=member_ids,
                    semantic_support=0.0,
                    metadata={
                        "detector": "CONTEXT_CLASSIFIER_V6_1",
                        "text_id": text.id,
                        "text": text.text,
                        "anchor_line_id": anchor.id,
                        "extension_line_ids": [line.id for line in extensions],
                        "anchor_length_px": anchor.length_px,
                        "bbox_area_ratio": self._bbox_area(bbox)
                        / max(1.0, drawing.width_px * drawing.height_px),
                        "source": "LOCAL_DIMENSION_CHAIN",
                    },
                )
            )
        return self._dedupe_by_overlap(output)

    def _dimension_extension_lines(
        self,
        *,
        anchor: DrawingLine,
        drawing: DrawingModel,
        endpoint_radius: float,
        max_extension_length: float,
    ) -> list[DrawingLine]:
        anchor_geom = self._line(anchor)
        endpoints = [Point(anchor.start.x, anchor.start.y), Point(anchor.end.x, anchor.end.y)]
        output: list[DrawingLine] = []
        for line in drawing.lines:
            if line.id == anchor.id or line.kind != "LINE":
                continue
            if line.length_px > max_extension_length:
                continue
            orthogonal_error = abs(90.0 - self._angle_diff(anchor.angle_deg, line.angle_deg))
            if orthogonal_error > self.DIMENSION_PERPENDICULAR_TOLERANCE_DEG:
                continue
            geom = self._line(line)
            if min(geom.distance(endpoint) for endpoint in endpoints) > endpoint_radius:
                continue
            if geom.distance(anchor_geom) > endpoint_radius:
                continue
            output.append(line)

        # Máximo dos extensiones principales: una por extremo. Esto evita que
        # cruces estructurales cercanos expandan la región de cota.
        selected: list[DrawingLine] = []
        for endpoint in endpoints:
            candidates = sorted(
                output,
                key=lambda line: (
                    self._line(line).distance(endpoint),
                    line.length_px,
                    line.id,
                ),
            )
            if candidates:
                chosen = candidates[0]
                if chosen.id not in {item.id for item in selected}:
                    selected.append(chosen)
        return selected

    def _bounded_dimension_bbox(
        self,
        *,
        text_bbox: DrawingBBox,
        lines: Sequence[DrawingLine],
        drawing: DrawingModel,
        pad: float,
        min_dim: float,
    ) -> DrawingBBox | None:
        xs = [text_bbox.x_min, text_bbox.x_max]
        ys = [text_bbox.y_min, text_bbox.y_max]
        for line in lines:
            xs.extend([line.start.x, line.end.x])
            ys.extend([line.start.y, line.end.y])
        bbox = DrawingBBox(
            x_min=max(0.0, min(xs) - pad),
            y_min=max(0.0, min(ys) - pad),
            x_max=min(float(drawing.width_px), max(xs) + pad),
            y_max=min(float(drawing.height_px), max(ys) + pad),
        )
        width = bbox.x_max - bbox.x_min
        height = bbox.y_max - bbox.y_min
        # Una cota puede ser larga, pero debe permanecer relativamente delgada.
        # Si ambas dimensiones crecen demasiado, se trata como asociación ambigua.
        if width > 0.72 * min_dim and height > 0.30 * min_dim:
            return None
        if height > 0.72 * min_dim and width > 0.30 * min_dim:
            return None
        area_ratio = self._bbox_area(bbox) / max(1.0, drawing.width_px * drawing.height_px)
        if area_ratio > 0.18:
            return None
        return bbox

    @staticmethod
    def _dimension_text(text: str) -> bool:
        normalized = text.strip().lower().replace(" ", "")
        if not normalized or len(normalized) > 18:
            return False
        return bool(
            re.fullmatch(r"[±~]?\d+(?:[.,]\d+)?(?:m|cm|mm)?", normalized)
            or re.fullmatch(r"\d+[x×]\d+(?:[.,]\d+)?", normalized)
        )

    @staticmethod
    def _with_detector(region: ContextRegion, detector: str) -> ContextRegion:
        return region.model_copy(update={"metadata": {**region.metadata, "classifier": detector}})

    @staticmethod
    def _distance_to_axis(angle: float) -> float:
        return min(
            abs(angle % 180.0),
            abs((angle % 180.0) - 90.0),
            abs((angle % 180.0) - 180.0),
        )

    @staticmethod
    def _line(line: DrawingLine) -> LineString:
        return LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])

    @staticmethod
    def _lines_bbox(lines: Sequence[DrawingLine]) -> DrawingBBox:
        xs = [value for line in lines for value in (line.start.x, line.end.x)]
        ys = [value for line in lines for value in (line.start.y, line.end.y)]
        return DrawingBBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    @staticmethod
    def _pad_bbox(bbox: DrawingBBox, pad: float, *, drawing: DrawingModel) -> DrawingBBox:
        return DrawingBBox(
            x_min=max(0.0, bbox.x_min - pad),
            y_min=max(0.0, bbox.y_min - pad),
            x_max=min(float(drawing.width_px), bbox.x_max + pad),
            y_max=min(float(drawing.height_px), bbox.y_max + pad),
        )

    @staticmethod
    def _bbox_area(bbox: DrawingBBox) -> float:
        return max(0.0, bbox.x_max - bbox.x_min) * max(0.0, bbox.y_max - bbox.y_min)

    @staticmethod
    def _bbox_overlap_ratio(first: DrawingBBox, second: DrawingBBox) -> float:
        x0 = max(first.x_min, second.x_min)
        y0 = max(first.y_min, second.y_min)
        x1 = min(first.x_max, second.x_max)
        y1 = min(first.y_max, second.y_max)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        a = max(1e-6, ContextRegionClassifier._bbox_area(first))
        b = max(1e-6, ContextRegionClassifier._bbox_area(second))
        return inter / min(a, b)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    def _dedupe_by_overlap(self, regions: Sequence[ContextRegion]) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        for region in sorted(regions, key=lambda item: -item.confidence):
            if any(self._bbox_overlap_ratio(region.bbox, other.bbox) >= 0.80 for other in output):
                continue
            output.append(region)
        return output
