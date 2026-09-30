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
class _Cell:
    bbox: DrawingBBox
    member_line_ids: tuple[str, ...]

    @property
    def width(self) -> float:
        return self.bbox.x_max - self.bbox.x_min

    @property
    def height(self) -> float:
        return self.bbox.y_max - self.bbox.y_min

    @property
    def cx(self) -> float:
        return (self.bbox.x_min + self.bbox.x_max) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox.y_min + self.bbox.y_max) / 2.0


class RepeatedCellDetector:
    """Detecta módulos rectangulares repetidos sin depender de líneas largas cruzadas.

    V5 solo detectaba una retícula cuando dos familias H/V largas se cruzaban con
    suficiente densidad. Muchos planos dibujan acabados, mosaicos, mobiliario o
    módulos como rectángulos independientes. V6 detecta primero la *celda física*
    y después agrupa celdas semejantes por tamaño, alineación y proximidad.

    No conoce nombres de casos ni dimensiones absolutas del proyecto. Todos los
    límites principales se normalizan contra el tamaño del LevelView y contra el
    tamaño mediano de la propia familia.
    """

    AXIS_TOLERANCE_DEG = 5.5

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
        minimum = max(metric.to_native_px(5.0), 0.005 * min_dim)
        maximum = max(metric.to_native_px(28.0), 0.24 * min_dim)
        junction_tol = max(metric.to_native_px(2.0), 0.0035 * min_dim)

        horizontal = [
            line for line in lines
            if self._axis_diff(line.angle_deg, 0.0) <= self.AXIS_TOLERANCE_DEG
            and minimum <= line.length_px <= 3.0 * maximum
        ]
        vertical = [
            line for line in lines
            if self._axis_diff(line.angle_deg, 90.0) <= self.AXIS_TOLERANCE_DEG
            and minimum <= line.length_px <= 3.0 * maximum
        ]
        if len(horizontal) < 2 or len(vertical) < 2:
            return []

        cells = self._rectangular_cells(
            horizontal=horizontal,
            vertical=vertical,
            junction_tol=junction_tol,
            minimum=minimum,
            maximum=maximum,
            polygon=polygon,
        )
        if len(cells) < 4:
            return []

        components = self._cell_components(cells=cells, min_dim=min_dim)
        regions: list[ContextRegion] = []
        for component in components:
            if len(component) < 4:
                continue
            region = self._region_from_cells(component=component, drawing=drawing, metric=metric)
            if region is not None:
                regions.append(region)
        return self._dedupe_regions(regions)

    def _rectangular_cells(
        self,
        *,
        horizontal: Sequence[DrawingLine],
        vertical: Sequence[DrawingLine],
        junction_tol: float,
        minimum: float,
        maximum: float,
        polygon,
    ) -> list[_Cell]:
        candidates: list[_Cell] = []
        hs = sorted(horizontal, key=lambda line: self._midpoint(line)[1])

        for i, top in enumerate(hs[:-1]):
            top_y = self._midpoint(top)[1]
            tx0, tx1 = sorted((top.start.x, top.end.x))
            for bottom in hs[i + 1 :]:
                bottom_y = self._midpoint(bottom)[1]
                height = bottom_y - top_y
                if height < minimum:
                    continue
                if height > maximum:
                    break
                bx0, bx1 = sorted((bottom.start.x, bottom.end.x))
                overlap_x0 = max(tx0, bx0)
                overlap_x1 = min(tx1, bx1)
                if overlap_x1 - overlap_x0 < minimum:
                    continue

                connectors: list[tuple[float, DrawingLine]] = []
                for vertical_line in vertical:
                    x = self._midpoint(vertical_line)[0]
                    if x < overlap_x0 - junction_tol or x > overlap_x1 + junction_tol:
                        continue
                    vy0, vy1 = sorted((vertical_line.start.y, vertical_line.end.y))
                    if vy0 > top_y + junction_tol or vy1 < bottom_y - junction_tol:
                        continue
                    connectors.append((x, vertical_line))
                if len(connectors) < 2:
                    continue

                connectors.sort(key=lambda item: item[0])
                for (left_x, left), (right_x, right) in zip(connectors[:-1], connectors[1:]):
                    width = right_x - left_x
                    if width < minimum or width > maximum:
                        continue
                    aspect = width / max(height, 1e-6)
                    if aspect < 0.22 or aspect > 4.5:
                        continue
                    cell_bbox = DrawingBBox(
                        x_min=float(left_x),
                        y_min=float(top_y),
                        x_max=float(right_x),
                        y_max=float(bottom_y),
                    )
                    cell_geom = box(cell_bbox.x_min, cell_bbox.y_min, cell_bbox.x_max, cell_bbox.y_max)
                    if polygon.intersection(cell_geom).area / max(cell_geom.area, 1e-6) < 0.82:
                        continue
                    members = set()
                    for line in (top, bottom, left, right):
                        members.add(line.id)
                        members.update(str(item) for item in line.evidence_ids if item)
                        members.update(line.metadata.get("physical_stroke_member_line_ids") or [])
                        members.update(line.metadata.get("context_equivalent_line_ids") or [])
                    candidates.append(_Cell(cell_bbox, tuple(sorted(members))))

        unique: list[_Cell] = []
        for cell in sorted(candidates, key=lambda item: item.width * item.height):
            if any(self._bbox_iou(cell.bbox, other.bbox) >= 0.86 for other in unique):
                continue
            unique.append(cell)
        return unique

    def _cell_components(self, *, cells: Sequence[_Cell], min_dim: float) -> list[list[_Cell]]:
        visited: set[int] = set()
        components: list[list[_Cell]] = []
        for index, cell in enumerate(cells):
            if index in visited:
                continue
            queue = [index]
            visited.add(index)
            component: list[_Cell] = []
            while queue:
                current_index = queue.pop()
                current = cells[current_index]
                component.append(current)
                for other_index, other in enumerate(cells):
                    if other_index in visited:
                        continue
                    if not self._compatible_cells(current, other, min_dim=min_dim):
                        continue
                    visited.add(other_index)
                    queue.append(other_index)
            components.append(component)
        return components

    def _compatible_cells(self, first: _Cell, second: _Cell, *, min_dim: float) -> bool:
        width_ratio = min(first.width, second.width) / max(first.width, second.width)
        height_ratio = min(first.height, second.height) / max(first.height, second.height)
        if width_ratio < 0.66 or height_ratio < 0.66:
            return False

        dx = abs(first.cx - second.cx)
        dy = abs(first.cy - second.cy)
        max_gap = max(0.012 * min_dim, 1.65 * max(first.width, first.height, second.width, second.height))
        if math.hypot(dx, dy) > max_gap:
            return False

        row_aligned = dy <= 0.42 * max(first.height, second.height)
        col_aligned = dx <= 0.42 * max(first.width, second.width)
        adjacent_x = dx <= 1.85 * max(first.width, second.width)
        adjacent_y = dy <= 1.85 * max(first.height, second.height)
        return (row_aligned and adjacent_x) or (col_aligned and adjacent_y)

    def _region_from_cells(
        self,
        *,
        component: Sequence[_Cell],
        drawing: DrawingModel,
        metric: MetricNormalizedContext,
    ) -> ContextRegion | None:
        widths = [cell.width for cell in component]
        heights = [cell.height for cell in component]
        median_w = float(median(widths))
        median_h = float(median(heights))
        width_cv = self._cv(widths)
        height_cv = self._cv(heights)
        if width_cv > 0.34 or height_cv > 0.34:
            return None

        xs = [cell.cx for cell in component]
        ys = [cell.cy for cell in component]
        col_centers = self._collapse_coordinates(xs, max(metric.to_native_px(2.0), 0.42 * median_w))
        row_centers = self._collapse_coordinates(ys, max(metric.to_native_px(2.0), 0.42 * median_h))
        row_count = len(row_centers)
        col_count = len(col_centers)
        if max(row_count, col_count) < 3:
            return None

        bbox = DrawingBBox(
            x_min=min(cell.bbox.x_min for cell in component),
            y_min=min(cell.bbox.y_min for cell in component),
            x_max=max(cell.bbox.x_max for cell in component),
            y_max=max(cell.bbox.y_max for cell in component),
        )
        region_area = max(1e-6, (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min))
        cell_area = sum(cell.width * cell.height for cell in component)
        coverage = min(1.0, cell_area / region_area)

        regularity = max(0.0, 1.0 - (width_cv + height_cv) / 0.68)
        count_score = min(1.0, (len(component) - 3) / 9.0)
        lattice_score = min(1.0, (row_count + col_count - 3) / 7.0)
        confidence = 0.42 * regularity + 0.34 * count_score + 0.24 * lattice_score

        if len(component) >= 6 and row_count >= 2 and col_count >= 2:
            region_type = "FLOOR_FINISH_GRID_REGION"
            quarantine_enabled = confidence >= 0.72
        else:
            region_type = "FURNITURE_MODULE_REGION"
            quarantine_enabled = confidence >= 0.84 and len(component) >= 5

        member_ids = sorted({line_id for cell in component for line_id in cell.member_line_ids})
        profiles: list[RepetitiveAxisProfile] = []
        vertical_tracks = self._cell_edge_coordinates(component, axis="x", minimum_tolerance=metric.to_native_px(1.5))
        horizontal_tracks = self._cell_edge_coordinates(component, axis="y", minimum_tolerance=metric.to_native_px(1.5))
        if len(vertical_tracks) >= 3:
            profiles.append(
                self._profile_from_coordinates(
                    angle_deg=90.0,
                    coords=vertical_tracks,
                    median_length=max(median_h, 1.0),
                    member_ids=member_ids,
                )
            )
        if len(horizontal_tracks) >= 3:
            profiles.append(
                self._profile_from_coordinates(
                    angle_deg=0.0,
                    coords=horizontal_tracks,
                    median_length=max(median_w, 1.0),
                    member_ids=member_ids,
                )
            )

        digest = hashlib.sha1(
            (region_type + "|" + "|".join(member_ids)).encode()
        ).hexdigest()[:16]
        pad = max(metric.to_native_px(1.5), 0.08 * min(median_w, median_h))
        return ContextRegion(
            id=f"CTX__{region_type}__{digest}",
            region_type=region_type,
            bbox=DrawingBBox(
                x_min=max(0.0, bbox.x_min - pad),
                y_min=max(0.0, bbox.y_min - pad),
                x_max=min(float(drawing.width_px), bbox.x_max + pad),
                y_max=min(float(drawing.height_px), bbox.y_max + pad),
            ),
            confidence=float(max(0.0, min(1.0, confidence))),
            quarantine_enabled=quarantine_enabled,
            profiles=profiles,
            member_line_ids=member_ids,
            semantic_support=0.0,
            metadata={
                "detector": "REPEATED_CELL_DETECTOR_V6",
                "cell_count": len(component),
                "row_count": row_count,
                "column_count": col_count,
                "median_cell_width_px": median_w,
                "median_cell_height_px": median_h,
                "cell_width_cv": width_cv,
                "cell_height_cv": height_cv,
                "cell_coverage_ratio": coverage,
                "source": "REPEATED_RECTANGULAR_CELLS",
            },
        )

    def _profile_from_coordinates(
        self,
        *,
        angle_deg: float,
        coords: Sequence[float],
        median_length: float,
        member_ids: Sequence[str],
    ) -> RepetitiveAxisProfile:
        ordered = sorted(coords)
        gaps = [b - a for a, b in zip(ordered[:-1], ordered[1:]) if b - a > 1e-6]
        spacing = float(median(gaps)) if gaps else 1.0
        spacing_cv = self._cv(gaps) if gaps else 0.0
        return RepetitiveAxisProfile(
            angle_deg=angle_deg,
            spacing_px=max(0.1, spacing),
            spacing_cv=max(0.0, spacing_cv),
            member_count=len(ordered),
            median_length_px=max(1.0, median_length),
            common_span_ratio=1.0,
            length_similarity=1.0,
            track_coordinates=[float(value) for value in ordered],
            member_line_ids=list(member_ids),
        )

    @staticmethod
    def _cell_edge_coordinates(
        cells: Sequence[_Cell],
        *,
        axis: str,
        minimum_tolerance: float = 1.5,
    ) -> list[float]:
        raw: list[float] = []
        for cell in cells:
            if axis == "x":
                raw.extend([cell.bbox.x_min, cell.bbox.x_max])
            else:
                raw.extend([cell.bbox.y_min, cell.bbox.y_max])
        reference = median([cell.width if axis == "x" else cell.height for cell in cells])
        return RepeatedCellDetector._collapse_coordinates(raw, max(minimum_tolerance, 0.08 * reference))

    @staticmethod
    def _collapse_coordinates(values: Sequence[float], tolerance: float) -> list[float]:
        output: list[float] = []
        for value in sorted(values):
            if output and abs(value - output[-1]) <= tolerance:
                output[-1] = (output[-1] + value) / 2.0
            else:
                output.append(float(value))
        return output

    @staticmethod
    def _cv(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        if mean <= 1e-9:
            return 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return math.sqrt(variance) / mean

    @staticmethod
    def _axis_diff(angle: float, target: float) -> float:
        diff = abs((angle - target) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _midpoint(line: DrawingLine) -> tuple[float, float]:
        return (
            (line.start.x + line.end.x) / 2.0,
            (line.start.y + line.end.y) / 2.0,
        )

    @staticmethod
    def _bbox_iou(first: DrawingBBox, second: DrawingBBox) -> float:
        x0 = max(first.x_min, second.x_min)
        y0 = max(first.y_min, second.y_min)
        x1 = min(first.x_max, second.x_max)
        y1 = min(first.y_max, second.y_max)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        a = max(1e-6, (first.x_max - first.x_min) * (first.y_max - first.y_min))
        b = max(1e-6, (second.x_max - second.x_min) * (second.y_max - second.y_min))
        return inter / max(a + b - inter, 1e-6)

    def _dedupe_regions(self, regions: Sequence[ContextRegion]) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        for region in sorted(regions, key=lambda item: (-item.confidence, -self._bbox_area(item.bbox))):
            if any(self._bbox_iou(region.bbox, existing.bbox) >= 0.72 for existing in output):
                continue
            output.append(region)
        return output

    @staticmethod
    def _bbox_area(bbox: DrawingBBox) -> float:
        return max(0.0, bbox.x_max - bbox.x_min) * max(0.0, bbox.y_max - bbox.y_min)
