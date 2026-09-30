from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict, deque
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView, PixelPoint


GridAxisOrientation = Literal["horizontal", "vertical"]
DimensionChainOrientation = Literal["HORIZONTAL", "VERTICAL"]
AxisGridState = Literal["RESOLVED", "PARTIAL", "UNRESOLVED", "CONFLICT"]


class ResolvedGridAxis(BaseModel):
    """
    Eje arquitectónico localizado en px mediante reconciliación híbrida.

    La cadena de cotas Gemini aporta únicamente identidad, orden y proporción
    relativa. La posición final en px solo se publica si una transformación
    afín de TODA la cadena encuentra soporte geométrico repetido en las líneas
    observadas por OpenCV/PyMuPDF.

    No convierte ni publica escala métrica.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_view_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    orientation: GridAxisOrientation
    coordinate_px: float
    start: PixelPoint
    end: PixelPoint
    semantic_chain_position: float = Field(ge=0.0)
    semantic_chain_orientation: DimensionChainOrientation
    source_dimension_evidence_ids: list[str] = Field(default_factory=list)
    source_axis_evidence_ids: list[str] = Field(default_factory=list)
    source_geometric_evidence_ids: list[str] = Field(default_factory=list)
    support_coordinate_px: float | None = None
    residual_px: float | None = Field(default=None, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confirmed: Literal[False] = False


class AxisGridFitDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_orientation: DimensionChainOrientation
    axis_orientation: GridAxisOrientation
    labels: list[str] = Field(default_factory=list)
    chain_dimension_evidence_ids: list[str] = Field(default_factory=list)
    chain_span_value: float = Field(default=0.0, ge=0.0)
    general_span_value: float | None = Field(default=None, gt=0.0)
    general_span_consistent: bool | None = None
    candidate_coordinate_count: int = Field(default=0, ge=0)
    matched_axis_count: int = Field(default=0, ge=0)
    axis_count: int = Field(default=0, ge=0)
    matched_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    median_residual_px: float | None = Field(default=None, ge=0.0)
    match_tolerance_px: float = Field(default=0.0, ge=0.0)
    affine_offset_px: float | None = None
    affine_pixels_per_semantic_unit: float | None = Field(default=None, gt=0.0)
    score: float | None = None
    state: AxisGridState = "UNRESOLVED"
    warnings: list[str] = Field(default_factory=list)


class AxisGridResolutionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_semantic_evidence_count: int = Field(default=0, ge=0)
    axis_semantic_evidence_count: int = Field(default=0, ge=0)
    raw_line_evidence_count: int = Field(default=0, ge=0)
    dimension_chain_count: int = Field(default=0, ge=0)
    resolved_chain_count: int = Field(default=0, ge=0)
    vertical_axis_count: int = Field(default=0, ge=0)
    horizontal_axis_count: int = Field(default=0, ge=0)
    rejected_dimension_evidence_ids: list[str] = Field(default_factory=list)
    conflicting_axis_labels: list[str] = Field(default_factory=list)
    fits: list[AxisGridFitDiagnostics] = Field(default_factory=list)


class AxisGridResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    axes: list[ResolvedGridAxis] = Field(default_factory=list)
    state: AxisGridState
    diagnostics: AxisGridResolutionDiagnostics
    warnings: list[str] = Field(default_factory=list)
    confirmed: Literal[False] = False

    def axes_by_orientation(
        self,
        orientation: GridAxisOrientation,
    ) -> list[ResolvedGridAxis]:
        return sorted(
            [axis for axis in self.axes if axis.orientation == orientation],
            key=lambda axis: (axis.coordinate_px, axis.label, axis.id),
        )


class _DimensionEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    orientation: DimensionChainOrientation
    start_label: str
    end_label: str
    value: float = Field(gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class _DimensionChain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orientation: DimensionChainOrientation
    labels: list[str] = Field(min_length=2)
    positions: list[float] = Field(min_length=2)
    edge_evidence_ids: list[str] = Field(default_factory=list)


class _CoordinateSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinate_px: float
    weight: float = Field(gt=0.0)
    evidence_ids: list[str] = Field(default_factory=list)


class _FitResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset_px: float
    scale_px_per_unit: float = Field(gt=0.0)
    score: float
    predicted_px: list[float]
    matched_support_px: list[float | None]
    residual_px: list[float | None]
    support_evidence_ids: list[list[str]]


class AxisGridResolver:
    """
    F03-A0 — resuelve primero la retícula arquitectónica.

    Estrategia:
    1. Toma DIMENSION/TRAMO Gemini como una cadena semántica relativa.
    2. NO usa esas cotas para publicar metros ni escala.
    3. Busca una transformación afín de la cadena que sea respaldada por
       múltiples líneas H/V reales de OpenCV/PyMuPDF.
    4. Solo entonces publica ejes en coordenadas px.

    Esto evita el fallo anterior donde AXIS legacy tenía geometry=NONE y, al
    mismo tiempo, evita inventar coordenadas directamente desde Gemini.
    """

    VERSION = "AXIS_GRID_RESOLVER_V2"
    NUMERIC_TOLERANCE = 1e-6

    def resolve(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> AxisGridResolution:
        raw = list(evidence)
        self._validate_inputs(level_view=level_view, evidence=raw)

        diagnostics = AxisGridResolutionDiagnostics(
            dimension_semantic_evidence_count=sum(
                1 for item in raw if self._semantic_category(item) == "DIMENSION"
            ),
            axis_semantic_evidence_count=sum(
                1 for item in raw if self._semantic_category(item) == "AXIS"
            ),
            raw_line_evidence_count=sum(
                1
                for item in raw
                if item.kind in {"RASTER_LINE", "VECTOR_LINE"}
                and item.geometry.geometry_type in {"SEGMENT", "POLYLINE"}
            ),
        )

        axis_semantic_lookup = self._axis_semantic_lookup(raw)
        axes_by_key: dict[tuple[GridAxisOrientation, str], ResolvedGridAxis] = {}
        warnings: list[str] = []

        for semantic_orientation in ("HORIZONTAL", "VERTICAL"):
            edges = self._dimension_edges(
                raw,
                semantic_orientation=semantic_orientation,
                diagnostics=diagnostics,
            )
            chains = self._dimension_chains(edges, diagnostics)
            diagnostics.dimension_chain_count += len(chains)

            axis_orientation: GridAxisOrientation = (
                "vertical" if semantic_orientation == "HORIZONTAL" else "horizontal"
            )
            supports = self._coordinate_supports(
                level_view=level_view,
                evidence=raw,
                axis_orientation=axis_orientation,
            )

            for chain in chains:
                fit_diag = AxisGridFitDiagnostics(
                    semantic_orientation=semantic_orientation,
                    axis_orientation=axis_orientation,
                    labels=list(chain.labels),
                    chain_dimension_evidence_ids=list(chain.edge_evidence_ids),
                    chain_span_value=float(chain.positions[-1]),
                    candidate_coordinate_count=len(supports),
                    axis_count=len(chain.labels),
                )
                general_value, general_consistent = self._general_span_consistency(
                    evidence=raw,
                    semantic_orientation=semantic_orientation,
                    start_label=chain.labels[0],
                    end_label=chain.labels[-1],
                    chain_total=float(chain.positions[-1]),
                )
                fit_diag.general_span_value = general_value
                fit_diag.general_span_consistent = general_consistent
                if general_consistent is False:
                    fit_diag.warnings.append(
                        "La cota GENERAL no coincide con la suma de la cadena TRAMO; "
                        "la cota se conserva como evidencia pero no gobierna geometría."
                    )

                fit, match_tolerance = self._fit_chain(
                    level_view=level_view,
                    chain=chain,
                    axis_orientation=axis_orientation,
                    supports=supports,
                )
                fit_diag.match_tolerance_px = match_tolerance

                if fit is None:
                    fit_diag.state = "UNRESOLVED"
                    fit_diag.warnings.append(
                        "La cadena semántica no obtuvo soporte geométrico global suficiente."
                    )
                    diagnostics.fits.append(fit_diag)
                    continue

                matched_residuals = [
                    value for value in fit.residual_px if value is not None
                ]
                fit_diag.matched_axis_count = len(matched_residuals)
                fit_diag.matched_fraction = (
                    len(matched_residuals) / len(chain.labels)
                    if chain.labels
                    else 0.0
                )
                fit_diag.median_residual_px = (
                    float(statistics.median(matched_residuals))
                    if matched_residuals
                    else None
                )
                fit_diag.affine_offset_px = fit.offset_px
                fit_diag.affine_pixels_per_semantic_unit = fit.scale_px_per_unit
                fit_diag.score = fit.score

                if not self._fit_is_acceptable(
                    axis_count=len(chain.labels),
                    residuals=fit.residual_px,
                    match_tolerance_px=match_tolerance,
                ):
                    fit_diag.state = "UNRESOLVED"
                    fit_diag.warnings.append(
                        "La transformación candidata fue descartada por baja cobertura "
                        "o residuales geométricos altos."
                    )
                    diagnostics.fits.append(fit_diag)
                    continue

                fit_diag.state = "RESOLVED"
                diagnostics.resolved_chain_count += 1
                diagnostics.fits.append(fit_diag)

                edge_ids_by_label = self._edge_ids_by_label(chain, edges)
                chain_confidence = self._fit_confidence(
                    residuals=fit.residual_px,
                    match_tolerance_px=match_tolerance,
                )

                for index, (label, semantic_position) in enumerate(
                    zip(chain.labels, chain.positions)
                ):
                    coordinate = float(fit.predicted_px[index])
                    support_coordinate = fit.matched_support_px[index]
                    residual = fit.residual_px[index]
                    geometric_ids = fit.support_evidence_ids[index]
                    semantic_axis_ids = axis_semantic_lookup.get(label, [])
                    source_dimension_ids = edge_ids_by_label.get(label, [])

                    start, end = self._axis_segment(
                        level_view=level_view,
                        orientation=axis_orientation,
                        coordinate_px=coordinate,
                    )
                    axis = ResolvedGridAxis(
                        id=self._stable_id(
                            "GRID_AXIS",
                            level_view.id,
                            {
                                "orientation": axis_orientation,
                                "label": label,
                                "coordinate": round(coordinate, 6),
                            },
                        ),
                        level_view_id=level_view.id,
                        label=label,
                        orientation=axis_orientation,
                        coordinate_px=coordinate,
                        start=start,
                        end=end,
                        semantic_chain_position=float(semantic_position),
                        semantic_chain_orientation=semantic_orientation,
                        source_dimension_evidence_ids=sorted(
                            set(source_dimension_ids)
                        ),
                        source_axis_evidence_ids=sorted(set(semantic_axis_ids)),
                        source_geometric_evidence_ids=sorted(set(geometric_ids)),
                        support_coordinate_px=support_coordinate,
                        residual_px=residual,
                        confidence=chain_confidence,
                        confirmed=False,
                    )
                    key = (axis_orientation, label)
                    previous = axes_by_key.get(key)
                    if previous is None:
                        axes_by_key[key] = axis
                    elif abs(previous.coordinate_px - axis.coordinate_px) <= match_tolerance:
                        axes_by_key[key] = self._merge_axes(previous, axis)
                    else:
                        diagnostics.conflicting_axis_labels.append(label)

        # Respaldo para proveedores futuros que sí entreguen AXIS con geometría.
        # Se toma una fotografía previa de los ejes inferidos por cadena para
        # poder contrastarlos después contra rótulos explícitos de eje.
        affine_axes_by_key = dict(axes_by_key)

        for observed in self._observed_semantic_axes(level_view, raw):
            key = (observed.orientation, observed.label)
            if key not in axes_by_key:
                axes_by_key[key] = observed

        # Los rótulos de eje VECTOR_TEXT/OCR_TEXT son una evidencia de identidad
        # más directa que un ajuste afín sobre líneas sin etiqueta. Antes este
        # mecanismo solo se usaba como fallback cuando faltaban ejes; por ello un
        # patrón paralelo (muro, marquesina, escalera, etc.) podía satisfacer toda
        # una cadena de cotas y desplazar A..I / 1..N a líneas incorrectas.
        #
        # V2 los usa SIEMPRE como cross-check de los ejes ya inferidos. Si dos o
        # más rótulos contradicen de forma material al mismo ajuste afín, ese fit
        # deja de ser fuente métrica RESOLVED. Las coordenadas de los rótulos
        # explícitos prevalecen sobre el fit no etiquetado, pero no sobre un AXIS
        # semántico que ya venga con geometría propia.
        label_warnings = self._reconcile_text_labeled_axes(
            level_view=level_view,
            evidence=raw,
            axes_by_key=axes_by_key,
            affine_axes_by_key=affine_axes_by_key,
            diagnostics=diagnostics,
        )
        warnings.extend(label_warnings)

        axes = sorted(
            axes_by_key.values(),
            key=lambda axis: (axis.orientation, axis.coordinate_px, axis.label),
        )
        diagnostics.vertical_axis_count = sum(
            1 for axis in axes if axis.orientation == "vertical"
        )
        diagnostics.horizontal_axis_count = sum(
            1 for axis in axes if axis.orientation == "horizontal"
        )
        diagnostics.rejected_dimension_evidence_ids = sorted(
            set(diagnostics.rejected_dimension_evidence_ids)
        )
        diagnostics.conflicting_axis_labels = sorted(
            set(diagnostics.conflicting_axis_labels)
        )

        if diagnostics.conflicting_axis_labels:
            state: AxisGridState = "CONFLICT"
        elif diagnostics.vertical_axis_count >= 2 and diagnostics.horizontal_axis_count >= 2:
            state = "RESOLVED"
        elif axes:
            state = "PARTIAL"
        else:
            state = "UNRESOLVED"

        if state != "RESOLVED":
            warnings.append(
                "F03 no pudo resolver una retícula H/V completa con soporte geométrico."
            )
        if any(fit.general_span_consistent is False for fit in diagnostics.fits):
            warnings.append(
                "Existen discrepancias entre cotas GENERAL y cadenas TRAMO; "
                "no se publicó escala métrica."
            )

        return AxisGridResolution(
            level_view_id=level_view.id,
            axes=axes,
            state=state,
            diagnostics=diagnostics,
            warnings=list(dict.fromkeys(warnings)),
            confirmed=False,
        )

    def _dimension_edges(
        self,
        evidence: list[RawEvidence],
        *,
        semantic_orientation: DimensionChainOrientation,
        diagnostics: AxisGridResolutionDiagnostics,
    ) -> list[_DimensionEdge]:
        result: list[_DimensionEdge] = []
        for item in evidence:
            if self._semantic_category(item) != "DIMENSION":
                continue
            if str(item.metadata.get("orientation") or "").strip().upper() != semantic_orientation:
                continue
            if str(item.metadata.get("span_type") or "").strip().upper() != "TRAMO":
                continue
            start = self._clean_label(item.metadata.get("reference_start"))
            end = self._clean_label(item.metadata.get("reference_end"))
            value = self._measurement_value(item)
            if not start or not end or start == end or value is None or value <= 0:
                diagnostics.rejected_dimension_evidence_ids.append(item.id)
                continue
            result.append(
                _DimensionEdge(
                    evidence_id=item.id,
                    orientation=semantic_orientation,
                    start_label=start,
                    end_label=end,
                    value=value,
                    confidence=item.confidence,
                )
            )
        return result

    def _dimension_chains(
        self,
        edges: list[_DimensionEdge],
        diagnostics: AxisGridResolutionDiagnostics,
    ) -> list[_DimensionChain]:
        if not edges:
            return []

        label_to_edge_indices: dict[str, set[int]] = defaultdict(set)
        for index, edge in enumerate(edges):
            label_to_edge_indices[edge.start_label].add(index)
            label_to_edge_indices[edge.end_label].add(index)

        remaining = set(range(len(edges)))
        chains: list[_DimensionChain] = []

        while remaining:
            seed = next(iter(remaining))
            component_edges: set[int] = set()
            queue: deque[int] = deque([seed])
            while queue:
                index = queue.popleft()
                if index in component_edges:
                    continue
                component_edges.add(index)
                edge = edges[index]
                for label in (edge.start_label, edge.end_label):
                    for neighbor in label_to_edge_indices[label]:
                        if neighbor in remaining and neighbor not in component_edges:
                            queue.append(neighbor)
            remaining.difference_update(component_edges)

            comp = [edges[index] for index in sorted(component_edges)]

            # Ruta 1: cadena dirigida simple. Se conserva como primera opción
            # porque evita introducir grados de libertad innecesarios.
            simple = self._simple_dimension_chain(comp)
            if simple is not None:
                chains.append(simple)
                continue

            # Ruta 2: red ramificada/cíclica. Resuelve posiciones semánticas
            # relativas por mínimos cuadrados sobre:
            #     position(end) - position(start) = value
            #
            # No convierte metros a px. El grounding geométrico sigue siendo
            # obligatorio en _fit_chain().
            network = self._network_dimension_chain(comp)
            if network is not None:
                chains.append(network)
                continue

            diagnostics.rejected_dimension_evidence_ids.extend(
                edge.evidence_id for edge in comp
            )

        return chains

    def _simple_dimension_chain(
        self,
        edges: list[_DimensionEdge],
    ) -> _DimensionChain | None:
        if not edges:
            return None

        out_by_label: dict[str, list[_DimensionEdge]] = defaultdict(list)
        indegree: dict[str, int] = defaultdict(int)
        labels: set[str] = set()
        for edge in edges:
            out_by_label[edge.start_label].append(edge)
            indegree[edge.end_label] += 1
            labels.update((edge.start_label, edge.end_label))

        starts = [
            label
            for label in labels
            if out_by_label.get(label) and indegree.get(label, 0) == 0
        ]
        ambiguous = (
            len(starts) != 1
            or any(len(values) != 1 for values in out_by_label.values())
            or any(indegree.get(label, 0) > 1 for label in labels)
        )
        if ambiguous:
            return None

        start_label = starts[0]
        ordered_labels = [start_label]
        positions = [0.0]
        edge_ids: list[str] = []
        current = start_label
        visited: set[str] = set()
        cumulative = 0.0

        while current in out_by_label:
            edge = out_by_label[current][0]
            if edge.evidence_id in visited:
                break
            visited.add(edge.evidence_id)
            cumulative += edge.value
            current = edge.end_label
            ordered_labels.append(current)
            positions.append(cumulative)
            edge_ids.append(edge.evidence_id)

        if len(edge_ids) != len(edges) or len(ordered_labels) < 2:
            return None

        return _DimensionChain(
            orientation=edges[0].orientation,
            labels=ordered_labels,
            positions=positions,
            edge_evidence_ids=edge_ids,
        )

    def _network_dimension_chain(
        self,
        edges: list[_DimensionEdge],
    ) -> _DimensionChain | None:
        if not edges:
            return None

        labels = sorted(
            {label for edge in edges for label in (edge.start_label, edge.end_label)},
            key=self._axis_label_sort_key,
        )
        if len(labels) < 2:
            return None

        # Fijar un origen elimina la indeterminación de traslación.
        anchor = labels[0]
        unknown = [label for label in labels if label != anchor]
        index = {label: position for position, label in enumerate(unknown)}
        size = len(unknown)
        if size == 0:
            return None

        ata = [[0.0 for _ in range(size)] for _ in range(size)]
        atb = [0.0 for _ in range(size)]

        for edge in edges:
            row = [0.0 for _ in range(size)]
            if edge.start_label != anchor:
                row[index[edge.start_label]] -= 1.0
            if edge.end_label != anchor:
                row[index[edge.end_label]] += 1.0
            weight = max(0.25, float(edge.confidence or 0.75))
            for i in range(size):
                if abs(row[i]) <= self.NUMERIC_TOLERANCE:
                    continue
                atb[i] += weight * row[i] * edge.value
                for j in range(size):
                    if abs(row[j]) <= self.NUMERIC_TOLERANCE:
                        continue
                    ata[i][j] += weight * row[i] * row[j]

        solution = self._solve_linear_system(ata, atb)
        if solution is None:
            return None

        positions_by_label = {anchor: 0.0}
        positions_by_label.update(
            {label: float(solution[index[label]]) for label in unknown}
        )

        # La red puede contener pequeñas discrepancias de redondeo. Se acepta
        # solo cuando el residual es bajo respecto a las cotas observadas.
        residuals = [
            abs(
                (positions_by_label[edge.end_label] - positions_by_label[edge.start_label])
                - edge.value
            )
            for edge in edges
        ]
        median_value = statistics.median(edge.value for edge in edges)
        residual_limit = max(0.03, 0.08 * median_value)
        if residuals and statistics.median(residuals) > residual_limit:
            return None

        minimum = min(positions_by_label.values())
        normalized = {label: value - minimum for label, value in positions_by_label.items()}
        ordered = sorted(
            normalized,
            key=lambda label: (normalized[label], self._axis_label_sort_key(label)),
        )

        # Una retícula necesita posiciones diferenciables.
        ordered_positions = [float(normalized[label]) for label in ordered]
        for first, second in zip(ordered_positions, ordered_positions[1:]):
            if second - first <= self.NUMERIC_TOLERANCE:
                return None

        return _DimensionChain(
            orientation=edges[0].orientation,
            labels=ordered,
            positions=ordered_positions,
            edge_evidence_ids=sorted(edge.evidence_id for edge in edges),
        )

    @staticmethod
    def _solve_linear_system(
        matrix: list[list[float]],
        vector: list[float],
    ) -> list[float] | None:
        n = len(vector)
        if n == 0 or len(matrix) != n or any(len(row) != n for row in matrix):
            return None

        augmented = [list(matrix[i]) + [float(vector[i])] for i in range(n)]
        eps = 1e-10

        for column in range(n):
            pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
            if abs(augmented[pivot][column]) <= eps:
                return None
            if pivot != column:
                augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

            pivot_value = augmented[column][column]
            for j in range(column, n + 1):
                augmented[column][j] /= pivot_value

            for row in range(n):
                if row == column:
                    continue
                factor = augmented[row][column]
                if abs(factor) <= eps:
                    continue
                for j in range(column, n + 1):
                    augmented[row][j] -= factor * augmented[column][j]

        return [augmented[i][n] for i in range(n)]

    def _coordinate_supports(
        self,
        *,
        level_view: LevelView,
        evidence: list[RawEvidence],
        axis_orientation: GridAxisOrientation,
    ) -> list[_CoordinateSupport]:
        stage_priority = ("raw_hough", "continuity_merged")
        selected: list[RawEvidence] = []
        for stage in stage_priority:
            selected = [
                item
                for item in evidence
                if item.source == "OPENCV"
                and item.kind == "RASTER_LINE"
                and str(item.metadata.get("stage") or "").strip().lower() == stage
                and self._line_matches_orientation(item, axis_orientation)
            ]
            if selected:
                break

        # PyMuPDF se agrega como soporte adicional, no como sustituto del Hough.
        selected.extend(
            item
            for item in evidence
            if item.source == "PYMUPDF"
            and item.kind == "VECTOR_LINE"
            and self._line_matches_orientation(item, axis_orientation)
        )
        if not selected:
            return []

        coordinate_weight: dict[int, float] = defaultdict(float)
        coordinate_ids: dict[int, set[str]] = defaultdict(set)
        for item in selected:
            points = item.geometry.points
            if len(points) < 2:
                continue
            start, end = points[0], points[-1]
            if axis_orientation == "vertical":
                coordinate = int(round((float(start.x) + float(end.x)) / 2.0))
            else:
                coordinate = int(round((float(start.y) + float(end.y)) / 2.0))
            length = math.hypot(float(end.x - start.x), float(end.y - start.y))
            if length <= self.NUMERIC_TOLERANCE:
                continue
            coordinate_weight[coordinate] += length
            coordinate_ids[coordinate].add(item.id)

        min_extent = float(
            min(level_view.raster_width_px, level_view.raster_height_px)
        )
        orth_extent = float(
            level_view.raster_height_px
            if axis_orientation == "vertical"
            else level_view.raster_width_px
        )
        smoothing_radius = max(1, int(round(0.005 * min_extent)))
        min_support_length = max(4.0, 0.008 * orth_extent)

        candidates: list[_CoordinateSupport] = []
        for coordinate in sorted(coordinate_weight):
            local_weight = sum(
                coordinate_weight.get(value, 0.0)
                for value in range(
                    coordinate - smoothing_radius,
                    coordinate + smoothing_radius + 1,
                )
            )
            if local_weight < min_support_length:
                continue
            ids: set[str] = set()
            for value in range(
                coordinate - smoothing_radius,
                coordinate + smoothing_radius + 1,
            ):
                ids.update(coordinate_ids.get(value, set()))
            candidates.append(
                _CoordinateSupport(
                    coordinate_px=float(coordinate),
                    weight=math.log1p(local_weight),
                    evidence_ids=sorted(ids),
                )
            )

        # NMS muy local evita probar varias veces el mismo trazo grueso sin
        # borrar caras distintas de un muro.
        suppression_radius = max(1.0, 0.003 * min_extent)
        ranked = sorted(candidates, key=lambda item: item.weight, reverse=True)
        kept: list[_CoordinateSupport] = []
        for item in ranked:
            if any(
                abs(item.coordinate_px - other.coordinate_px) < suppression_radius
                for other in kept
            ):
                continue
            kept.append(item)
            if len(kept) >= 180:
                break
        return sorted(kept, key=lambda item: item.coordinate_px)

    def _fit_chain(
        self,
        *,
        level_view: LevelView,
        chain: _DimensionChain,
        axis_orientation: GridAxisOrientation,
        supports: list[_CoordinateSupport],
    ) -> tuple[_FitResult | None, float]:
        min_extent = float(
            min(level_view.raster_width_px, level_view.raster_height_px)
        )
        match_tolerance = max(4.0, 0.025 * min_extent)
        if len(chain.labels) < 2 or len(supports) < 2:
            return None, match_tolerance

        total = float(chain.positions[-1])
        if total <= self.NUMERIC_TOLERANCE:
            return None, match_tolerance

        image_extent = float(
            level_view.raster_width_px
            if axis_orientation == "vertical"
            else level_view.raster_height_px
        )
        expected_scale = image_extent / total
        min_scale = max(self.NUMERIC_TOLERANCE, 0.15 * expected_scale)
        max_scale = 4.0 * expected_scale
        min_semantic_pair_span = max(self.NUMERIC_TOLERANCE, 0.20 * total)

        best: _FitResult | None = None
        positions = chain.positions

        semantic_pairs = [
            (i, j)
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
            if positions[j] - positions[i] >= min_semantic_pair_span
        ]
        support_pairs = [
            (first, second)
            for i, first in enumerate(supports)
            for second in supports[i + 1 :]
            if second.coordinate_px > first.coordinate_px
        ]

        sigma = max(1.0, 0.40 * match_tolerance)
        margin = max(match_tolerance * 2.0, 0.05 * image_extent)

        for i, j in semantic_pairs:
            semantic_delta = positions[j] - positions[i]
            for first, second in support_pairs:
                scale = (second.coordinate_px - first.coordinate_px) / semantic_delta
                if scale < min_scale or scale > max_scale:
                    continue
                offset = first.coordinate_px - scale * positions[i]
                predicted = [offset + scale * value for value in positions]
                if min(predicted) < -margin or max(predicted) > image_extent + margin:
                    continue

                matched_support: list[float | None] = []
                residuals: list[float | None] = []
                support_ids: list[list[str]] = []
                score = 0.0

                for coordinate in predicted:
                    nearby = [
                        support
                        for support in supports
                        if abs(support.coordinate_px - coordinate) <= match_tolerance
                    ]
                    if not nearby:
                        matched_support.append(None)
                        residuals.append(None)
                        support_ids.append([])
                        score -= 2.0
                        continue
                    chosen = min(
                        nearby,
                        key=lambda support: (
                            abs(support.coordinate_px - coordinate),
                            -support.weight,
                        ),
                    )
                    residual = abs(chosen.coordinate_px - coordinate)
                    score += chosen.weight * math.exp(-((residual / sigma) ** 2))
                    matched_support.append(chosen.coordinate_px)
                    residuals.append(residual)
                    support_ids.append(list(chosen.evidence_ids))

                score += 0.15 * (first.weight + second.weight)
                candidate = _FitResult(
                    offset_px=offset,
                    scale_px_per_unit=scale,
                    score=score,
                    predicted_px=predicted,
                    matched_support_px=matched_support,
                    residual_px=residuals,
                    support_evidence_ids=support_ids,
                )
                if best is None or candidate.score > best.score:
                    best = candidate

        return best, match_tolerance

    @staticmethod
    def _fit_is_acceptable(
        *,
        axis_count: int,
        residuals: list[float | None],
        match_tolerance_px: float,
    ) -> bool:
        matched = [value for value in residuals if value is not None]
        if not matched:
            return False
        minimum_matches = max(2, math.ceil(0.60 * axis_count))
        if len(matched) < minimum_matches:
            return False
        if statistics.median(matched) > 0.75 * match_tolerance_px:
            return False
        return True

    @staticmethod
    def _fit_confidence(
        *,
        residuals: list[float | None],
        match_tolerance_px: float,
    ) -> float:
        matched = [value for value in residuals if value is not None]
        if not residuals or not matched:
            return 0.0
        coverage = len(matched) / len(residuals)
        median_residual = statistics.median(matched)
        residual_quality = max(
            0.0,
            1.0 - median_residual / max(match_tolerance_px, 1e-6),
        )
        return max(0.0, min(1.0, 0.65 * coverage + 0.35 * residual_quality))

    def _reconcile_text_labeled_axes(
        self,
        *,
        level_view: LevelView,
        evidence: list[RawEvidence],
        axes_by_key: dict[tuple[GridAxisOrientation, str], ResolvedGridAxis],
        affine_axes_by_key: dict[tuple[GridAxisOrientation, str], ResolvedGridAxis],
        diagnostics: AxisGridResolutionDiagnostics,
    ) -> list[str]:
        """Cruza ejes inferidos con rótulos explícitos del plano.

        La cadena DIMENSION/TRAMO solo aporta identidad/orden relativo. El ajuste
        afín puede coincidir accidentalmente con familias paralelas que no son la
        retícula (caras de muro, marquesinas, peldaños, etc.). Un texto de eje
        localizado y asociado a una línea geométrica identifica directamente qué
        soporte pertenece a esa etiqueta.

        Reglas:
        - si la orientación ya tiene >=2 ejes, solo se aceptan rótulos de labels
          ya referenciados por la cadena; no se agregan números/letras ajenos;
        - si faltan ejes, conserva el comportamiento de fallback anterior;
        - un rótulo no reemplaza un AXIS semántico con geometría explícita;
        - >=2 contradicciones de rótulo invalidan el fit afín como fuente métrica,
          pero la retícula puede seguir RESOLVED mediante los propios rótulos.
        """
        warnings: list[str] = []
        evidence_by_id = {item.id: item for item in evidence}
        anchor_count: dict[GridAxisOrientation, int] = defaultdict(int)
        conflict_count: dict[GridAxisOrientation, int] = defaultdict(int)
        conflict_labels: dict[GridAxisOrientation, set[str]] = defaultdict(set)

        min_extent = float(min(level_view.raster_width_px, level_view.raster_height_px))
        conflict_tolerance = max(6.0, 0.030 * min_extent)

        for axis_orientation in ("vertical", "horizontal"):
            current_count = sum(
                1 for orientation, _label in axes_by_key if orientation == axis_orientation
            )
            allow_fallback_additions = current_count < 2
            supports = self._coordinate_supports(
                level_view=level_view,
                evidence=evidence,
                axis_orientation=axis_orientation,
            )
            text_axes = self._text_label_axes(
                level_view=level_view,
                evidence=evidence,
                axis_orientation=axis_orientation,
                supports=supports,
            )

            for observed in text_axes:
                key = (observed.orientation, observed.label)
                previous = axes_by_key.get(key)

                # Cuando ya existe retícula semántica suficiente, el texto solo
                # puede confirmar/corregir labels que pertenecen a ella.
                if previous is None and not allow_fallback_additions:
                    continue

                affine = affine_axes_by_key.get(key)
                if affine is not None:
                    anchor_count[axis_orientation] += 1
                    if abs(float(affine.coordinate_px) - float(observed.coordinate_px)) > conflict_tolerance:
                        conflict_count[axis_orientation] += 1
                        conflict_labels[axis_orientation].add(observed.label)

                if previous is None:
                    axes_by_key[key] = observed
                    continue

                distance = abs(float(previous.coordinate_px) - float(observed.coordinate_px))
                if distance <= conflict_tolerance:
                    axes_by_key[key] = self._merge_axes(previous, observed)
                    continue

                if self._axis_has_direct_semantic_geometry(previous, evidence_by_id):
                    diagnostics.conflicting_axis_labels.append(observed.label)
                    warnings.append(
                        f"Eje {observed.label}: rótulo textual y AXIS geométrico explícito discrepan "
                        f"{distance:.1f}px; se conserva CONFLICT para revisión."
                    )
                    continue

                axes_by_key[key] = self._prefer_text_labeled_axis(previous, observed)
                warnings.append(
                    f"Eje {observed.label}: rótulo explícito corrigió {distance:.1f}px un soporte "
                    "afín no etiquetado."
                )

        invalidated_orientations: set[GridAxisOrientation] = set()
        for axis_orientation in ("vertical", "horizontal"):
            anchors = anchor_count.get(axis_orientation, 0)
            conflicts = conflict_count.get(axis_orientation, 0)
            if anchors < 2 or conflicts < 2:
                continue
            if conflicts / anchors < 0.50:
                continue
            invalidated_orientations.add(axis_orientation)

        if invalidated_orientations:
            for fit in diagnostics.fits:
                if fit.state != "RESOLVED" or fit.axis_orientation not in invalidated_orientations:
                    continue
                fit.state = "CONFLICT"
                labels = ",".join(sorted(conflict_labels.get(fit.axis_orientation, set())))
                fit.warnings.append(
                    "El ajuste afín fue contradicho por rótulos explícitos de eje "
                    f"({labels or 'múltiples labels'}); no se usa como fuente métrica."
                )
                diagnostics.resolved_chain_count = max(0, diagnostics.resolved_chain_count - 1)

            names = ", ".join(sorted(invalidated_orientations))
            warnings.append(
                "Ajuste(s) afín(es) invalidado(s) por identidad de ejes observada: " + names + "."
            )

        return list(dict.fromkeys(warnings))

    @staticmethod
    def _axis_has_direct_semantic_geometry(
        axis: ResolvedGridAxis,
        evidence_by_id: dict[str, RawEvidence],
    ) -> bool:
        for evidence_id in axis.source_axis_evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is None:
                continue
            if AxisGridResolver._semantic_category(item) != "AXIS":
                continue
            if item.geometry.geometry_type in {"SEGMENT", "POLYLINE"} and len(item.geometry.points) >= 2:
                return True
        return False

    @staticmethod
    def _prefer_text_labeled_axis(
        previous: ResolvedGridAxis,
        observed: ResolvedGridAxis,
    ) -> ResolvedGridAxis:
        previous_conf = float(previous.confidence or 0.0)
        observed_conf = float(observed.confidence or 0.0)
        confidence = max(observed_conf, min(previous_conf, 0.90))
        return observed.model_copy(
            update={
                "semantic_chain_position": previous.semantic_chain_position,
                "semantic_chain_orientation": previous.semantic_chain_orientation,
                "source_dimension_evidence_ids": sorted(
                    set(previous.source_dimension_evidence_ids)
                    | set(observed.source_dimension_evidence_ids)
                ),
                "source_axis_evidence_ids": sorted(
                    set(previous.source_axis_evidence_ids)
                    | set(observed.source_axis_evidence_ids)
                ),
                "source_geometric_evidence_ids": sorted(
                    set(previous.source_geometric_evidence_ids)
                    | set(observed.source_geometric_evidence_ids)
                ),
                "confidence": confidence,
            },
            deep=False,
        )

    def _observed_semantic_axes(
        self,
        level_view: LevelView,
        evidence: list[RawEvidence],
    ) -> list[ResolvedGridAxis]:
        result: list[ResolvedGridAxis] = []
        for item in evidence:
            if self._semantic_category(item) != "AXIS":
                continue
            label = self._clean_label(
                item.metadata.get("semantic_name") or item.text
            )
            if not label:
                continue
            points = item.geometry.points
            if item.geometry.geometry_type not in {"SEGMENT", "POLYLINE"} or len(points) < 2:
                continue
            start, end = points[0], points[-1]
            dx = abs(float(end.x - start.x))
            dy = abs(float(end.y - start.y))
            if dx <= self.NUMERIC_TOLERANCE and dy > self.NUMERIC_TOLERANCE:
                orientation: GridAxisOrientation = "vertical"
                coordinate = (float(start.x) + float(end.x)) / 2.0
            elif dy <= self.NUMERIC_TOLERANCE and dx > self.NUMERIC_TOLERANCE:
                orientation = "horizontal"
                coordinate = (float(start.y) + float(end.y)) / 2.0
            else:
                continue
            full_start, full_end = self._axis_segment(
                level_view=level_view,
                orientation=orientation,
                coordinate_px=coordinate,
            )
            result.append(
                ResolvedGridAxis(
                    id=self._stable_id(
                        "GRID_AXIS_OBSERVED",
                        level_view.id,
                        {"label": label, "orientation": orientation, "evidence": item.id},
                    ),
                    level_view_id=level_view.id,
                    label=label,
                    orientation=orientation,
                    coordinate_px=coordinate,
                    start=full_start,
                    end=full_end,
                    semantic_chain_position=0.0,
                    semantic_chain_orientation=(
                        "HORIZONTAL" if orientation == "vertical" else "VERTICAL"
                    ),
                    source_dimension_evidence_ids=[],
                    source_axis_evidence_ids=[item.id],
                    source_geometric_evidence_ids=[item.id],
                    support_coordinate_px=coordinate,
                    residual_px=0.0,
                    confidence=item.confidence,
                    confirmed=False,
                )
            )
        return result

    def _text_label_axes(
        self,
        *,
        level_view: LevelView,
        evidence: list[RawEvidence],
        axis_orientation: GridAxisOrientation,
        supports: list[_CoordinateSupport],
    ) -> list[ResolvedGridAxis]:
        if len(supports) < 2:
            return []

        width = float(level_view.raster_width_px)
        height = float(level_view.raster_height_px)
        min_extent = min(width, height)
        match_tolerance = max(6.0, 0.030 * min_extent)
        outer_fraction = 0.32

        candidates: dict[str, list[tuple[RawEvidence, _CoordinateSupport, float]]] = defaultdict(list)

        for item in evidence:
            if item.kind not in {"VECTOR_TEXT", "OCR_TEXT"}:
                continue
            label = self._axis_label_from_text(item.text)
            if label is None:
                continue

            expected_orientation: GridAxisOrientation = (
                "vertical" if label.isdigit() else "horizontal"
            )
            if expected_orientation != axis_orientation:
                continue

            bbox = item.geometry.bbox_px
            if bbox is None:
                continue
            center_x = (float(bbox.x_min) + float(bbox.x_max)) / 2.0
            center_y = (float(bbox.y_min) + float(bbox.y_max)) / 2.0

            # Los rótulos de ejes se esperan fuera del cuerpo principal del
            # dibujo: números arriba/abajo para ejes verticales y letras a
            # izquierda/derecha para ejes horizontales. Este filtro evita
            # convertir numeración interior, notas o cotas en ejes.
            if axis_orientation == "vertical":
                if not (
                    center_y <= outer_fraction * height
                    or center_y >= (1.0 - outer_fraction) * height
                ):
                    continue
                text_coordinate = center_x
            else:
                if not (
                    center_x <= outer_fraction * width
                    or center_x >= (1.0 - outer_fraction) * width
                ):
                    continue
                text_coordinate = center_y

            nearby = [
                support
                for support in supports
                if abs(support.coordinate_px - text_coordinate) <= match_tolerance
            ]
            if not nearby:
                continue
            chosen = min(
                nearby,
                key=lambda support: (
                    abs(support.coordinate_px - text_coordinate),
                    -support.weight,
                ),
            )
            residual = abs(chosen.coordinate_px - text_coordinate)
            candidates[label].append((item, chosen, residual))

        if len(candidates) < 2:
            return []

        selected: list[tuple[str, float, list[str], list[str], float]] = []
        for label, values in candidates.items():
            # Agrupa por coordenada geométrica. Si el mismo texto aparece en
            # ambas caras del plano, ambas observaciones convergen al mismo
            # soporte y refuerzan la decisión.
            clusters: list[list[tuple[RawEvidence, _CoordinateSupport, float]]] = []
            for value in sorted(values, key=lambda item: item[1].coordinate_px):
                placed = False
                for cluster in clusters:
                    center = statistics.median(
                        member[1].coordinate_px for member in cluster
                    )
                    if abs(value[1].coordinate_px - center) <= match_tolerance:
                        cluster.append(value)
                        placed = True
                        break
                if not placed:
                    clusters.append([value])

            best_cluster = max(
                clusters,
                key=lambda cluster: (
                    len(cluster),
                    sum(member[1].weight for member in cluster),
                    -statistics.median(member[2] for member in cluster),
                ),
            )
            coordinate = float(statistics.median(
                member[1].coordinate_px for member in best_cluster
            ))
            text_ids = sorted({member[0].id for member in best_cluster})
            geometry_ids = sorted({
                evidence_id
                for member in best_cluster
                for evidence_id in member[1].evidence_ids
            })
            median_residual = float(statistics.median(
                member[2] for member in best_cluster
            ))
            selected.append(
                (label, coordinate, text_ids, geometry_ids, median_residual)
            )

        if len(selected) < 2:
            return []

        selected.sort(key=lambda item: (item[1], self._axis_label_sort_key(item[0])))
        origin = selected[0][1]
        result: list[ResolvedGridAxis] = []

        for label, coordinate, text_ids, geometry_ids, residual in selected:
            start, end = self._axis_segment(
                level_view=level_view,
                orientation=axis_orientation,
                coordinate_px=coordinate,
            )
            confidence = max(
                0.45,
                min(0.90, 0.80 - 0.35 * residual / max(match_tolerance, 1e-6)),
            )
            result.append(
                ResolvedGridAxis(
                    id=self._stable_id(
                        "GRID_AXIS_TEXT_FALLBACK",
                        level_view.id,
                        {
                            "orientation": axis_orientation,
                            "label": label,
                            "coordinate": round(coordinate, 6),
                        },
                    ),
                    level_view_id=level_view.id,
                    label=label,
                    orientation=axis_orientation,
                    coordinate_px=coordinate,
                    start=start,
                    end=end,
                    semantic_chain_position=max(0.0, coordinate - origin),
                    semantic_chain_orientation=(
                        "HORIZONTAL" if axis_orientation == "vertical" else "VERTICAL"
                    ),
                    source_dimension_evidence_ids=[],
                    source_axis_evidence_ids=text_ids,
                    source_geometric_evidence_ids=geometry_ids,
                    support_coordinate_px=coordinate,
                    residual_px=residual,
                    confidence=confidence,
                    confirmed=False,
                )
            )

        return result

    @staticmethod
    def _axis_label_from_text(value: object) -> str | None:
        text = str(value or "").strip().upper()
        if not text:
            return None
        compact = re.sub(r"\s+", " ", text)
        match = re.fullmatch(r"(?:EJE\s*)?([A-Z]|[0-9]{1,2})", compact)
        return match.group(1) if match else None

    @staticmethod
    def _axis_label_sort_key(value: str) -> tuple[int, int | str]:
        label = str(value or "").strip().upper()
        if label.isdigit():
            return (0, int(label))
        return (1, label)

    def _general_span_consistency(
        self,
        *,
        evidence: list[RawEvidence],
        semantic_orientation: DimensionChainOrientation,
        start_label: str,
        end_label: str,
        chain_total: float,
    ) -> tuple[float | None, bool | None]:
        for item in evidence:
            if self._semantic_category(item) != "DIMENSION":
                continue
            if str(item.metadata.get("orientation") or "").strip().upper() != semantic_orientation:
                continue
            if str(item.metadata.get("span_type") or "").strip().upper() != "GENERAL":
                continue
            start = self._clean_label(item.metadata.get("reference_start"))
            end = self._clean_label(item.metadata.get("reference_end"))
            if (start, end) != (start_label, end_label):
                continue
            value = self._measurement_value(item)
            if value is None:
                return None, None
            tolerance = max(0.01, 0.02 * value)
            return value, abs(value - chain_total) <= tolerance
        return None, None

    @staticmethod
    def _edge_ids_by_label(
        chain: _DimensionChain,
        edges: list[_DimensionEdge],
    ) -> dict[str, list[str]]:
        allowed = set(chain.edge_evidence_ids)
        result: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.evidence_id not in allowed:
                continue
            result[edge.start_label].append(edge.evidence_id)
            result[edge.end_label].append(edge.evidence_id)
        return {key: sorted(set(value)) for key, value in result.items()}

    @staticmethod
    def _axis_semantic_lookup(evidence: list[RawEvidence]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for item in evidence:
            if AxisGridResolver._semantic_category(item) != "AXIS":
                continue
            label = AxisGridResolver._clean_label(
                item.metadata.get("semantic_name") or item.text
            )
            if label:
                result[label].append(item.id)
        return {key: sorted(set(value)) for key, value in result.items()}

    @staticmethod
    def _measurement_value(item: RawEvidence) -> float | None:
        measurements = item.metadata.get("measurements")
        if isinstance(measurements, list):
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    continue
                try:
                    value = float(measurement.get("value"))
                except (TypeError, ValueError):
                    continue
                if value <= 0:
                    continue
                unit = str(measurement.get("unit") or "").strip().upper()
                if unit == "M" or not unit:
                    return value
                if unit == "CM":
                    return value / 100.0
                if unit == "MM":
                    return value / 1000.0
        try:
            value = float(str(item.text or "").strip())
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _line_matches_orientation(
        item: RawEvidence,
        orientation: GridAxisOrientation,
    ) -> bool:
        points = item.geometry.points
        if len(points) < 2:
            return False
        start, end = points[0], points[-1]
        dx = abs(float(end.x - start.x))
        dy = abs(float(end.y - start.y))
        if orientation == "vertical":
            return dx <= max(1.5, 0.01 * max(dy, 1.0)) and dy > 0
        return dy <= max(1.5, 0.01 * max(dx, 1.0)) and dx > 0

    @staticmethod
    def _axis_segment(
        *,
        level_view: LevelView,
        orientation: GridAxisOrientation,
        coordinate_px: float,
    ) -> tuple[PixelPoint, PixelPoint]:
        if orientation == "vertical":
            x = int(round(coordinate_px))
            return (
                PixelPoint(x=x, y=0),
                PixelPoint(x=x, y=max(1, level_view.raster_height_px - 1)),
            )
        y = int(round(coordinate_px))
        return (
            PixelPoint(x=0, y=y),
            PixelPoint(x=max(1, level_view.raster_width_px - 1), y=y),
        )

    @staticmethod
    def _merge_axes(
        first: ResolvedGridAxis,
        second: ResolvedGridAxis,
    ) -> ResolvedGridAxis:
        if first.orientation != second.orientation or first.label != second.label:
            return first
        first_conf = first.confidence or 0.0
        second_conf = second.confidence or 0.0
        chosen = first if first_conf >= second_conf else second
        return chosen.model_copy(
            update={
                "source_dimension_evidence_ids": sorted(
                    set(first.source_dimension_evidence_ids)
                    | set(second.source_dimension_evidence_ids)
                ),
                "source_axis_evidence_ids": sorted(
                    set(first.source_axis_evidence_ids)
                    | set(second.source_axis_evidence_ids)
                ),
                "source_geometric_evidence_ids": sorted(
                    set(first.source_geometric_evidence_ids)
                    | set(second.source_geometric_evidence_ids)
                ),
            },
            deep=True,
        )

    @staticmethod
    def _semantic_category(item: RawEvidence) -> str:
        if item.source != "GEMINI" or item.kind != "GEMINI_OBSERVATION":
            return ""
        return str(item.metadata.get("semantic_category") or "").strip().upper()

    @staticmethod
    def _clean_label(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _stable_id(prefix: str, level_view_id: str, payload: dict) -> str:
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"{level_view_id}__F03_{prefix}__{digest}"

    @staticmethod
    def _validate_inputs(*, level_view: LevelView, evidence: list[RawEvidence]) -> None:
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(f"RawEvidence {item.id} pertenece a otro LevelView.")
