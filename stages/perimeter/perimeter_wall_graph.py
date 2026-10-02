from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import LineString, Polygon

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView

from .perimeter_models import (
    PerimeterCandidate,
    PerimeterEvidenceRef,
    PerimeterPointPx,
    PerimeterPolygonPx,
)


GraphState = Literal["RESOLVED", "REVIEW", "UNRESOLVED"]
GraphOrientation = Literal["HORIZONTAL", "VERTICAL"]


class PerimeterGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    point: PerimeterPointPx


class PerimeterGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    start_node_id: str = Field(min_length=1)
    end_node_id: str = Field(min_length=1)
    orientation: GraphOrientation
    length_px: float = Field(gt=0.0)
    evidence_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class PerimeterWallGraphResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    state: GraphState
    candidate: PerimeterCandidate | None = None
    root_contour_evidence_id: str | None = None
    nodes: list[PerimeterGraphNode] = Field(default_factory=list)
    edges: list[PerimeterGraphEdge] = Field(default_factory=list)
    line_evidence_count: int = Field(default=0, ge=0)
    vector_support_count: int = Field(default=0, ge=0)
    raster_support_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _AxisLine:
    evidence: RawEvidence
    orientation: GraphOrientation
    axis: float
    start: float
    end: float


@dataclass(slots=True)
class _ProfileRun:
    start: int
    end: int
    axis: float

    @property
    def length(self) -> int:
        return self.end - self.start


class PerimeterWallGraphBuilder:
    """
    Reconstruye un ciclo exterior de muros a partir de evidencia propia del motor.

    Entradas:
        - RASTER_CONTOUR raíz: envolvente observada del componente estructural;
        - RASTER_LINE continuity_merged: ejes gráficos H/V observados;
        - VECTOR_LINE: corroboración vectorial cuando existe.

    Estrategia:
        1. selecciona el componente estructural con mayor extensión bidimensional;
        2. obtiene perfiles laterales del contorno raíz;
        3. elimina excursiones locales mediante ruptura natural de persistencia
           (sin umbral fijo de px);
        4. convierte los cambios persistentes en un ciclo ortogonal;
        5. ancla cada lado al eje H/V observado más próximo con traslape real;
        6. publica nodos/aristas y evidencia por tramo.

    Gemini no participa.
    No convierte px -> m.
    No usa el bbox del LevelView como perímetro.
    """

    def build(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> PerimeterWallGraphResult:
        raw = [item for item in evidence if item.source != "GEMINI"]
        self._validate(level_view=level_view, evidence=raw)

        root_contours = self._root_contours(raw)
        if not root_contours:
            return PerimeterWallGraphResult(
                level_view_id=level_view.id,
                state="UNRESOLVED",
                notes=["No existe RASTER_CONTOUR raíz utilizable para reconstruir el ciclo exterior."],
            )

        ranked = sorted(
            root_contours,
            key=lambda item: self._two_dimensional_extent(item),
            reverse=True,
        )
        root = ranked[0]
        if len(ranked) > 1 and math.isclose(
            self._two_dimensional_extent(ranked[0]),
            self._two_dimensional_extent(ranked[1]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            return PerimeterWallGraphResult(
                level_view_id=level_view.id,
                state="REVIEW",
                root_contour_evidence_id=root.id,
                notes=[
                    "Existen varios componentes raíz con la misma extensión bidimensional; no se fuerza selección."
                ],
            )

        params = self._opencv_parameters(root)
        vertical_kernel = max(3, int(params.get("directional_kernel_vertical") or 3))
        axis_tolerance = max(0.0, float(params.get("axis_tolerance_px") or 0.0))

        mask, bbox = self._contour_mask(level_view=level_view, contour=root)
        if mask is None:
            return PerimeterWallGraphResult(
                level_view_id=level_view.id,
                state="UNRESOLVED",
                root_contour_evidence_id=root.id,
                notes=["El contorno raíz no pudo rasterizarse como componente estructural."],
            )

        left_runs, right_runs = self._stable_side_profiles(
            mask=mask,
            y_min=bbox[1],
            y_max=bbox[3],
            median_window=vertical_kernel,
            axis_tolerance=axis_tolerance,
        )
        polygon_points = self._polygon_from_profiles(
            left_runs=left_runs,
            right_runs=right_runs,
            y_min=bbox[1],
            y_max=bbox[3],
        )
        if len(polygon_points) < 4:
            return PerimeterWallGraphResult(
                level_view_id=level_view.id,
                state="UNRESOLVED",
                root_contour_evidence_id=root.id,
                notes=["Los perfiles persistentes no produjeron un ciclo ortogonal suficiente."],
            )

        axis_lines = self._axis_lines(raw)
        snapped = self._snap_cycle_to_observed_axes(
            points=polygon_points,
            axis_lines=axis_lines,
        )
        snapped = self._simplify_orthogonal_ring(snapped)

        try:
            polygon = Polygon(snapped)
        except Exception:
            polygon = Polygon()
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
            return PerimeterWallGraphResult(
                level_view_id=level_view.id,
                state="UNRESOLVED",
                root_contour_evidence_id=root.id,
                notes=["El ciclo estructural reconstruido no forma un polígono válido."],
            )

        nodes, edges, supporting = self._build_graph(
            level_view=level_view,
            points=snapped,
            axis_lines=axis_lines,
            root_contour=root,
            axis_tolerance=axis_tolerance,
        )

        unsupported_edges = [edge for edge in edges if not edge.evidence_ids]
        sources = sorted({source for edge in edges for source in edge.sources})
        geometry_source = "HYBRID" if {"PYMUPDF", "OPENCV"}.issubset(sources) else "RASTER"

        candidate = PerimeterCandidate(
            id=self._candidate_id(level_view_id=level_view.id, points=snapped),
            level_view_id=level_view.id,
            geometry=PerimeterPolygonPx(
                points=[PerimeterPointPx(x=float(x), y=float(y)) for x, y in snapped]
            ),
            geometry_source=geometry_source,
            evidence=[self._evidence_ref(item) for item in supporting],
            semantic_evidence_ids=[],
            semantic_localized_evidence_ids=[],
            interior_ring_count=0,
            confirmed=False,
        )

        if unsupported_edges:
            state: GraphState = "REVIEW"
            notes = [
                f"El ciclo cerró geométricamente, pero {len(unsupported_edges)} tramo(s) no tienen línea H/V específica asociada.",
                "Se conserva el ciclo para revisión/fine tuning; Gemini no intervino.",
            ]
        else:
            state = "RESOLVED"
            notes = [
                "Ciclo exterior reconstruido y cerrado con soporte geométrico por tramo.",
                "La selección proviene del componente estructural + ejes H/V observados; Gemini no intervino.",
            ]

        return PerimeterWallGraphResult(
            level_view_id=level_view.id,
            state=state,
            candidate=candidate,
            root_contour_evidence_id=root.id,
            nodes=nodes,
            edges=edges,
            line_evidence_count=len(axis_lines),
            vector_support_count=sum(1 for item in supporting if item.source == "PYMUPDF"),
            raster_support_count=sum(1 for item in supporting if item.source == "OPENCV"),
            notes=notes,
        )

    @staticmethod
    def _validate(*, level_view: LevelView, evidence: Sequence[RawEvidence]) -> None:
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(f"RawEvidence {item.id} pertenece a otro LevelView.")
            if item.confirmed:
                raise ValueError(f"RawEvidence {item.id} llegó confirmed=True.")

    @staticmethod
    def _root_contours(evidence: Sequence[RawEvidence]) -> list[RawEvidence]:
        result: list[RawEvidence] = []
        for item in evidence:
            if item.kind != "RASTER_CONTOUR" or item.source != "OPENCV":
                continue
            if item.metadata.get("closed") is not True:
                continue
            hierarchy = item.metadata.get("hierarchy")
            parent = hierarchy.get("parent") if isinstance(hierarchy, dict) else None
            if parent is not None:
                continue
            if item.geometry.geometry_type != "POLYLINE" or len(item.geometry.points) < 3:
                continue
            result.append(item)
        return result

    @staticmethod
    def _two_dimensional_extent(evidence: RawEvidence) -> float:
        points = evidence.geometry.points
        xs = [float(point.x) for point in points]
        ys = [float(point.y) for point in points]
        return min(max(xs) - min(xs), max(ys) - min(ys))

    @staticmethod
    def _opencv_parameters(evidence: RawEvidence) -> dict[str, object]:
        opencv = evidence.metadata.get("opencv")
        return dict(opencv) if isinstance(opencv, dict) else {}

    @staticmethod
    def _contour_mask(
        *, level_view: LevelView, contour: RawEvidence
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
        points = [(int(point.x), int(point.y)) for point in contour.geometry.points]
        if len(set(points)) < 3:
            return None, (0, 0, 0, 0)
        array = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(
            (level_view.raster_height_px, level_view.raster_width_px), dtype=np.uint8
        )
        cv2.fillPoly(mask, [array], 1)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return mask, (min(xs), min(ys), max(xs), max(ys))

    def _stable_side_profiles(
        self,
        *,
        mask: np.ndarray,
        y_min: int,
        y_max: int,
        median_window: int,
        axis_tolerance: float,
    ) -> tuple[list[_ProfileRun], list[_ProfileRun]]:
        height = mask.shape[0]
        left = np.full(height, np.nan, dtype=np.float64)
        right = np.full(height, np.nan, dtype=np.float64)
        valid = np.zeros(height, dtype=bool)

        for y in range(max(0, y_min), min(height - 1, y_max) + 1):
            xs = np.flatnonzero(mask[y])
            if xs.size == 0:
                continue
            valid[y] = True
            left[y] = float(xs.min())
            right[y] = float(xs.max())

        left_runs = self._persistent_runs(
            values=left,
            valid=valid,
            median_window=median_window,
            axis_tolerance=axis_tolerance,
        )
        right_runs = self._persistent_runs(
            values=right,
            valid=valid,
            median_window=median_window,
            axis_tolerance=axis_tolerance,
        )
        return left_runs, right_runs

    def _persistent_runs(
        self,
        *,
        values: np.ndarray,
        valid: np.ndarray,
        median_window: int,
        axis_tolerance: float,
    ) -> list[_ProfileRun]:
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return []

        filled = values.copy()
        missing = np.flatnonzero(~valid)
        if missing.size:
            filled[missing] = np.interp(missing, indices, values[indices])

        smoothed = self._median_filter_1d(filled, median_window)
        runs: list[_ProfileRun] = []
        start = 0
        mean = float(smoothed[0])
        count = 1

        for index, value in enumerate(smoothed[1:], start=1):
            value = float(value)
            if abs(value - mean) <= axis_tolerance:
                mean = ((mean * count) + value) / float(count + 1)
                count += 1
                continue
            runs.append(_ProfileRun(start=start, end=index, axis=mean))
            start = index
            mean = value
            count = 1
        runs.append(_ProfileRun(start=start, end=len(smoothed), axis=mean))

        significant = self._natural_break_runs(runs)
        significant = self._merge_adjacent_axis_runs(
            significant, axis_tolerance=axis_tolerance
        )
        return self._cover_profile(significant, total_length=len(smoothed))

    @staticmethod
    def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
        window = max(3, int(window))
        if window % 2 == 0:
            window += 1
        radius = window // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        return np.asarray(
            [np.median(padded[index : index + window]) for index in range(len(values))],
            dtype=np.float64,
        )

    @staticmethod
    def _natural_break_runs(runs: Sequence[_ProfileRun]) -> list[_ProfileRun]:
        if len(runs) <= 2:
            return [ _ProfileRun(r.start, r.end, r.axis) for r in runs ]

        lengths = sorted({run.length for run in runs if run.length > 0})
        if len(lengths) <= 1:
            return [ _ProfileRun(r.start, r.end, r.axis) for r in runs ]

        gaps: list[tuple[float, int, int]] = []
        for lower, upper in zip(lengths, lengths[1:]):
            ratio = float(upper) / float(lower)
            gaps.append((ratio, lower, upper))

        ratios = [item[0] for item in gaps]
        strongest = max(gaps, key=lambda item: item[0])
        median_ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
        if strongest[0] <= median_ratio:
            return [ _ProfileRun(r.start, r.end, r.axis) for r in runs ]

        minimum_persistent_length = strongest[2]
        selected = [
            _ProfileRun(run.start, run.end, run.axis)
            for run in runs
            if run.length >= minimum_persistent_length
        ]
        return selected or [max(runs, key=lambda run: run.length)]

    @staticmethod
    def _merge_adjacent_axis_runs(
        runs: Sequence[_ProfileRun], *, axis_tolerance: float
    ) -> list[_ProfileRun]:
        ordered = sorted(runs, key=lambda item: item.start)
        result: list[_ProfileRun] = []
        for run in ordered:
            if not result or abs(run.axis - result[-1].axis) > axis_tolerance:
                result.append(_ProfileRun(run.start, run.end, run.axis))
                continue
            previous = result[-1]
            total = previous.length + run.length
            axis = (
                (previous.axis * previous.length) + (run.axis * run.length)
            ) / float(total)
            result[-1] = _ProfileRun(previous.start, run.end, axis)
        return result

    @staticmethod
    def _cover_profile(
        runs: Sequence[_ProfileRun], *, total_length: int
    ) -> list[_ProfileRun]:
        ordered = sorted(runs, key=lambda item: item.start)
        if not ordered:
            return []
        result: list[_ProfileRun] = []
        for index, run in enumerate(ordered):
            if index == 0:
                start = 0
            else:
                start = int(round((ordered[index - 1].end + run.start) / 2.0))
            if index == len(ordered) - 1:
                end = total_length
            else:
                end = int(round((run.end + ordered[index + 1].start) / 2.0))
            result.append(_ProfileRun(start=start, end=end, axis=run.axis))
        return result

    @staticmethod
    def _polygon_from_profiles(
        *,
        left_runs: Sequence[_ProfileRun],
        right_runs: Sequence[_ProfileRun],
        y_min: int,
        y_max: int,
    ) -> list[tuple[float, float]]:
        def side_points(runs: Sequence[_ProfileRun], *, reverse: bool) -> list[tuple[float, float]]:
            result: list[tuple[float, float]] = []
            for run in runs:
                start = max(run.start, y_min)
                end = min(run.end - 1, y_max)
                if end < start:
                    continue
                axis = float(run.axis)
                if not result:
                    result.append((axis, float(start)))
                else:
                    previous_x, previous_y = result[-1]
                    if previous_y != float(start):
                        result.append((previous_x, float(start)))
                    if previous_x != axis:
                        result.append((axis, float(start)))
                result.append((axis, float(end)))
            return list(reversed(result)) if reverse else result

        points = [
            *side_points(left_runs, reverse=False),
            *side_points(right_runs, reverse=True),
        ]
        return PerimeterWallGraphBuilder._simplify_orthogonal_ring(points)

    def _axis_lines(self, evidence: Sequence[RawEvidence]) -> list[_AxisLine]:
        result: list[_AxisLine] = []
        for item in evidence:
            if item.kind not in {"RASTER_LINE", "VECTOR_LINE"}:
                continue
            if item.kind == "RASTER_LINE" and item.metadata.get("stage") != "continuity_merged":
                continue
            if item.geometry.geometry_type != "SEGMENT" or len(item.geometry.points) != 2:
                continue
            p0, p1 = item.geometry.points
            dx = float(p1.x - p0.x)
            dy = float(p1.y - p0.y)
            if dy == 0.0 and dx != 0.0:
                result.append(
                    _AxisLine(
                        evidence=item,
                        orientation="HORIZONTAL",
                        axis=float(p0.y),
                        start=min(float(p0.x), float(p1.x)),
                        end=max(float(p0.x), float(p1.x)),
                    )
                )
            elif dx == 0.0 and dy != 0.0:
                result.append(
                    _AxisLine(
                        evidence=item,
                        orientation="VERTICAL",
                        axis=float(p0.x),
                        start=min(float(p0.y), float(p1.y)),
                        end=max(float(p0.y), float(p1.y)),
                    )
                )
        return result

    def _snap_cycle_to_observed_axes(
        self,
        *,
        points: Sequence[tuple[float, float]],
        axis_lines: Sequence[_AxisLine],
    ) -> list[tuple[float, float]]:
        points = self._simplify_orthogonal_ring(points)
        if len(points) < 4:
            return list(points)

        edge_axes: list[tuple[GraphOrientation, float]] = []
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            if start[1] == end[1]:
                orientation: GraphOrientation = "HORIZONTAL"
                axis = self._nearest_supported_axis(
                    orientation=orientation,
                    current_axis=float(start[1]),
                    start=min(start[0], end[0]),
                    end=max(start[0], end[0]),
                    axis_lines=axis_lines,
                )
            elif start[0] == end[0]:
                orientation = "VERTICAL"
                axis = self._nearest_supported_axis(
                    orientation=orientation,
                    current_axis=float(start[0]),
                    start=min(start[1], end[1]),
                    end=max(start[1], end[1]),
                    axis_lines=axis_lines,
                )
            else:
                return list(points)
            edge_axes.append((orientation, axis))

        snapped: list[tuple[float, float]] = []
        for index in range(len(points)):
            previous_orientation, previous_axis = edge_axes[index - 1]
            current_orientation, current_axis = edge_axes[index]
            if previous_orientation == current_orientation:
                return list(points)
            if previous_orientation == "VERTICAL":
                x = previous_axis
                y = current_axis
            else:
                x = current_axis
                y = previous_axis
            snapped.append((float(x), float(y)))
        return snapped

    @staticmethod
    def _nearest_supported_axis(
        *,
        orientation: GraphOrientation,
        current_axis: float,
        start: float,
        end: float,
        axis_lines: Sequence[_AxisLine],
    ) -> float:
        grouped: dict[float, float] = {}
        for line in axis_lines:
            if line.orientation != orientation:
                continue
            overlap = max(0.0, min(end, line.end) - max(start, line.start))
            if overlap <= 0.0:
                continue
            grouped[line.axis] = grouped.get(line.axis, 0.0) + overlap
        if not grouped:
            return current_axis
        # La cercanía al perfil estructural domina; el traslape desempata.
        axis, _ = min(
            grouped.items(),
            key=lambda item: (abs(item[0] - current_axis), -item[1], item[0]),
        )
        return float(axis)

    def _build_graph(
        self,
        *,
        level_view: LevelView,
        points: Sequence[tuple[float, float]],
        axis_lines: Sequence[_AxisLine],
        root_contour: RawEvidence,
        axis_tolerance: float,
    ) -> tuple[list[PerimeterGraphNode], list[PerimeterGraphEdge], list[RawEvidence]]:
        nodes: list[PerimeterGraphNode] = []
        edges: list[PerimeterGraphEdge] = []
        supporting: dict[str, RawEvidence] = {root_contour.id: root_contour}

        for index, point in enumerate(points):
            nodes.append(
                PerimeterGraphNode(
                    id=f"{level_view.id}__PERIMETER_GRAPH_NODE_{index + 1}",
                    point=PerimeterPointPx(x=float(point[0]), y=float(point[1])),
                )
            )

        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            if start[1] == end[1]:
                orientation: GraphOrientation = "HORIZONTAL"
                axis = float(start[1])
                interval = (min(start[0], end[0]), max(start[0], end[0]))
            else:
                orientation = "VERTICAL"
                axis = float(start[0])
                interval = (min(start[1], end[1]), max(start[1], end[1]))

            edge_evidence: list[RawEvidence] = []
            for line in axis_lines:
                if line.orientation != orientation:
                    continue
                if abs(line.axis - axis) > axis_tolerance:
                    continue
                overlap = max(0.0, min(interval[1], line.end) - max(interval[0], line.start))
                if overlap <= 0.0:
                    continue
                edge_evidence.append(line.evidence)
                supporting[line.evidence.id] = line.evidence

            # El contorno raíz es evidencia geométrica del ciclo completo y se
            # conserva, pero no sustituye la ausencia de línea H/V específica.
            sources = sorted({item.source for item in edge_evidence})
            edges.append(
                PerimeterGraphEdge(
                    id=f"{level_view.id}__PERIMETER_GRAPH_EDGE_{index + 1}",
                    start_node_id=nodes[index].id,
                    end_node_id=nodes[(index + 1) % len(nodes)].id,
                    orientation=orientation,
                    length_px=math.hypot(end[0] - start[0], end[1] - start[1]),
                    evidence_ids=sorted({item.id for item in edge_evidence}),
                    sources=sources,
                )
            )

        return nodes, edges, list(supporting.values())

    @staticmethod
    def _simplify_orthogonal_ring(
        points: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        cleaned: list[tuple[float, float]] = []
        for point in points:
            normalized = (float(point[0]), float(point[1]))
            if not cleaned or normalized != cleaned[-1]:
                cleaned.append(normalized)
        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()

        changed = True
        while changed and len(cleaned) >= 4:
            changed = False
            result: list[tuple[float, float]] = []
            for index, current in enumerate(cleaned):
                previous = cleaned[index - 1]
                following = cleaned[(index + 1) % len(cleaned)]
                if (
                    previous[0] == current[0] == following[0]
                    or previous[1] == current[1] == following[1]
                ):
                    changed = True
                    continue
                result.append(current)
            if len(result) < 4:
                break
            cleaned = result
        return cleaned

    @staticmethod
    def _evidence_ref(item: RawEvidence) -> PerimeterEvidenceRef:
        semantic_category = item.metadata.get("semantic_category")
        description = None
        if item.kind == "RASTER_CONTOUR":
            description = "Contorno raíz del componente estructural raster."
        elif item.kind in {"RASTER_LINE", "VECTOR_LINE"}:
            description = "Línea H/V que soporta un tramo del ciclo exterior."
        return PerimeterEvidenceRef(
            evidence_id=item.id,
            source=item.source,
            kind=item.kind,
            semantic_category=(str(semantic_category) if semantic_category else None),
            text=item.text,
            description=description,
            confidence=item.confidence,
        )

    @staticmethod
    def _candidate_id(
        *, level_view_id: str, points: Sequence[tuple[float, float]]
    ) -> str:
        payload = "|".join(f"{x:.6f},{y:.6f}" for x, y in points).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"{level_view_id}__PERIMETER_GRAPH_OUTER_CYCLE__{digest}"
