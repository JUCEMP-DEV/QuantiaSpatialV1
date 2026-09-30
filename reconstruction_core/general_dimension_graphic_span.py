from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import EditablePerimeterModel


DimensionOrientation = Literal["HORIZONTAL", "VERTICAL"]
DimensionGraphicSide = Literal["TOP", "BOTTOM", "LEFT", "RIGHT"]


@dataclass(frozen=True, slots=True)
class GraphicDimensionSpan:
    dimension_evidence_id: str
    semantic_orientation: DimensionOrientation
    side: DimensionGraphicSide
    value_m: float
    span_px: float
    m_per_px: float
    line_coordinate_px: float
    line_start_px: float
    line_end_px: float
    score: float
    confidence: float
    geometric_evidence_ids: tuple[str, ...]
    source: str
    axis_referenced: bool = False


@dataclass(frozen=True, slots=True)
class _ParallelSegment:
    evidence_id: str
    source: str
    orientation: Literal["HORIZONTAL", "VERTICAL"]
    coordinate_px: float
    start_px: float
    end_px: float
    length_px: float


class GeneralDimensionGraphicSpanResolver:
    """Mide cotas GENERAL directamente sobre su línea gráfica exterior.

    Este resolver existe para separar dos problemas que antes estaban acoplados:

    - localizar ejes arquitectónicos;
    - calcular la relación métrica px<->m.

    Una cota GENERAL visible ya contiene una distancia métrica y una línea gráfica
    que materializa ese mismo span. Si ambas piezas pueden reconciliarse, la escala
    puede resolverse sin exigir que el texto ``1:N`` exista y sin usar como proxy
    la distancia entre ejes inferidos.

    Reglas de seguridad:
    - solo consume DIMENSION + span_type=GENERAL;
    - solo mide líneas H/V observadas por PyMuPDF/OpenCV;
    - busca fuera o en una banda estrecha del perímetro F02;
    - nunca inventa una línea ni usa el tamaño total del perímetro como cota;
    - cuando Gemini indicó lado (superior/inferior/izquierda/derecha), se respeta;
    - sin lado explícito, exige alineación fuerte con los extremos del perímetro
      para evitar confundir linderos/página con líneas de dimensión.
    """

    VERSION = "GENERAL_DIMENSION_GRAPHIC_SPAN_V2"
    ANGLE_TOLERANCE_DEG = 4.0
    AXIS_REFERENCED_CONFIDENCE_CAP = 0.70
    MIN_SPAN_RATIO = 0.45
    MAX_SPAN_RATIO = 1.25

    def resolve(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
        perimeter: EditablePerimeterModel | None,
    ) -> list[GraphicDimensionSpan]:
        if perimeter is None:
            return []
        if perimeter.level_view_id != level_view.id:
            raise ValueError("GraphicSpanResolver recibió perímetro de otro LevelView.")

        raw = list(evidence)
        for item in raw:
            if item.level_view_id != level_view.id:
                raise ValueError("GraphicSpanResolver recibió evidencia de otro LevelView.")

        bbox = self._perimeter_bbox(perimeter)
        segments = self._parallel_segments(raw)
        if not segments:
            return []

        results: list[GraphicDimensionSpan] = []
        for dimension in raw:
            if self._semantic_category(dimension) != "DIMENSION":
                continue
            if str(dimension.metadata.get("span_type") or "").strip().upper() != "GENERAL":
                continue

            orientation = str(dimension.metadata.get("orientation") or "").strip().upper()
            if orientation not in {"HORIZONTAL", "VERTICAL"}:
                continue
            value_m = self._measurement_value_m(dimension)
            if value_m is None or value_m <= 0.0:
                continue

            # Una cota entre ejes (p.ej. A→I o 1→8) NO mide necesariamente
            # la envolvente F02. Volados, marquesinas, aleros o retranqueos
            # pueden extender el perímetro físico más allá de la retícula.
            #
            # La línea gráfica localizada respecto del bbox F02 se conserva
            # como evidencia diagnóstica, pero no puede cerrar por sí sola la
            # escala cuando la propia semántica declara dos ejes de referencia.
            reference_start = str(dimension.metadata.get("reference_start") or "").strip()
            reference_end = str(dimension.metadata.get("reference_end") or "").strip()
            axis_referenced = bool(reference_start and reference_end and reference_start != reference_end)

            side_hint = self._side_hint(
                orientation=orientation,
                raw=dimension.metadata.get("dimension_side"),
            )
            selected = self._select_span(
                level_view=level_view,
                bbox=bbox,
                segments=segments,
                orientation=orientation,
                side_hint=side_hint,
            )
            if selected is None:
                continue

            segment, side, score = selected
            m_per_px = float(value_m) / float(segment.length_px)
            if not math.isfinite(m_per_px) or m_per_px <= 0.0:
                continue

            # La confianza deriva de geometría observada, no de la confianza LLM.
            # Una cota GENERAL con lado explícito y línea exterior larga llega a
            # ~0.97-0.99. Sin lado explícito se exige endpoint alignment y se
            # conserva un techo ligeramente menor.
            geometry_conf = 0.78 + 0.21 * max(0.0, min(1.0, score))
            if side_hint is None:
                geometry_conf = min(0.975, geometry_conf)
            else:
                geometry_conf = min(0.995, geometry_conf + 0.01)
            semantic_conf = float(
                dimension.confidence if dimension.confidence is not None else 0.90
            )
            confidence = min(0.995, 0.80 * geometry_conf + 0.20 * semantic_conf)
            if axis_referenced:
                confidence = min(confidence, self.AXIS_REFERENCED_CONFIDENCE_CAP)

            results.append(
                GraphicDimensionSpan(
                    dimension_evidence_id=dimension.id,
                    semantic_orientation=orientation,  # type: ignore[arg-type]
                    side=side,
                    value_m=float(value_m),
                    span_px=float(segment.length_px),
                    m_per_px=m_per_px,
                    line_coordinate_px=float(segment.coordinate_px),
                    line_start_px=float(segment.start_px),
                    line_end_px=float(segment.end_px),
                    score=float(score),
                    confidence=float(confidence),
                    geometric_evidence_ids=(segment.evidence_id,),
                    source=segment.source,
                    axis_referenced=axis_referenced,
                )
            )

        return results

    @staticmethod
    def _perimeter_bbox(perimeter: EditablePerimeterModel) -> tuple[float, float, float, float]:
        xs = [float(vertex.point_px.x) for vertex in perimeter.vertices]
        ys = [float(vertex.point_px.y) for vertex in perimeter.vertices]
        if not xs or not ys:
            raise ValueError("EditablePerimeterModel sin vértices.")
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("BBox perimetral degenerado.")
        return x_min, y_min, x_max, y_max

    def _parallel_segments(self, evidence: Sequence[RawEvidence]) -> list[_ParallelSegment]:
        result: list[_ParallelSegment] = []
        for item in evidence:
            if item.kind not in {"VECTOR_LINE", "RASTER_LINE"}:
                continue
            if item.geometry.geometry_type not in {"SEGMENT", "POLYLINE"}:
                continue
            points = list(item.geometry.points)
            if len(points) < 2:
                continue
            for first, second in zip(points, points[1:]):
                x1, y1 = float(first.x), float(first.y)
                x2, y2 = float(second.x), float(second.y)
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length <= 0.0:
                    continue
                angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
                horizontal_error = min(angle, 180.0 - angle)
                vertical_error = abs(angle - 90.0)
                if horizontal_error <= self.ANGLE_TOLERANCE_DEG:
                    result.append(
                        _ParallelSegment(
                            evidence_id=item.id,
                            source=item.source,
                            orientation="HORIZONTAL",
                            coordinate_px=(y1 + y2) / 2.0,
                            start_px=min(x1, x2),
                            end_px=max(x1, x2),
                            length_px=abs(x2 - x1),
                        )
                    )
                elif vertical_error <= self.ANGLE_TOLERANCE_DEG:
                    result.append(
                        _ParallelSegment(
                            evidence_id=item.id,
                            source=item.source,
                            orientation="VERTICAL",
                            coordinate_px=(x1 + x2) / 2.0,
                            start_px=min(y1, y2),
                            end_px=max(y1, y2),
                            length_px=abs(y2 - y1),
                        )
                    )
        return result

    def _select_span(
        self,
        *,
        level_view: LevelView,
        bbox: tuple[float, float, float, float],
        segments: Sequence[_ParallelSegment],
        orientation: str,
        side_hint: DimensionGraphicSide | None,
    ) -> tuple[_ParallelSegment, DimensionGraphicSide, float] | None:
        x_min, y_min, x_max, y_max = bbox
        min_dim = float(min(level_view.raster_width_px, level_view.raster_height_px))
        if orientation == "HORIZONTAL":
            major_start, major_end = x_min, x_max
            sides: tuple[DimensionGraphicSide, ...] = ("TOP", "BOTTOM")
        else:
            major_start, major_end = y_min, y_max
            sides = ("LEFT", "RIGHT")

        if side_hint is not None:
            sides = (side_hint,)

        major_span = major_end - major_start
        if major_span <= 0.0:
            return None

        outer_search = max(24.0, 0.13 * min_dim)
        inner_slack = max(4.0, 0.008 * min_dim)
        scored: list[tuple[float, _ParallelSegment, DimensionGraphicSide]] = []

        for segment in segments:
            if segment.orientation != orientation:
                continue
            if segment.length_px < self.MIN_SPAN_RATIO * major_span:
                continue
            if segment.length_px > self.MAX_SPAN_RATIO * major_span:
                continue

            for side in sides:
                distance = self._outward_distance(
                    side=side,
                    coordinate=segment.coordinate_px,
                    bbox=bbox,
                )
                if distance < -inner_slack or distance > outer_search:
                    continue

                span_ratio = segment.length_px / major_span
                ratio_score = max(0.0, 1.0 - abs(span_ratio - 1.0) / 0.35)
                outside_score = max(0.0, min(1.0, max(0.0, distance) / outer_search))
                source_score = 1.0 if segment.source == "PYMUPDF" else 0.85

                endpoint_error = (
                    abs(segment.start_px - major_start) + abs(segment.end_px - major_end)
                ) / major_span
                endpoint_score = max(0.0, 1.0 - endpoint_error / 0.06)

                if side_hint is None:
                    # Sin indicación de lado, el endpoint alignment es la defensa
                    # principal frente a linderos o bordes de página paralelos.
                    score = (
                        0.40 * ratio_score
                        + 0.30 * endpoint_score
                        + 0.20 * outside_score
                        + 0.10 * source_score
                    )
                else:
                    # Con lado explícito, una cota puede cubrir solo un escalón de
                    # la planta (p.ej. vertical derecha 7.70 vs izquierda 8.50),
                    # por lo que no se fuerza a coincidir con el bbox completo.
                    score = (
                        0.58 * ratio_score
                        + 0.27 * outside_score
                        + 0.15 * source_score
                    )

                scored.append((score, segment, side))

        if not scored:
            return None

        # Preferencia determinista: score, separación exterior, longitud y fuente.
        scored.sort(
            key=lambda row: (
                row[0],
                max(
                    0.0,
                    self._outward_distance(
                        side=row[2],
                        coordinate=row[1].coordinate_px,
                        bbox=bbox,
                    ),
                ),
                row[1].length_px,
                1 if row[1].source == "PYMUPDF" else 0,
                row[1].evidence_id,
            ),
            reverse=True,
        )
        score, segment, side = scored[0]
        return segment, side, float(score)

    @staticmethod
    def _outward_distance(
        *,
        side: DimensionGraphicSide,
        coordinate: float,
        bbox: tuple[float, float, float, float],
    ) -> float:
        x_min, y_min, x_max, y_max = bbox
        if side == "TOP":
            return y_min - coordinate
        if side == "BOTTOM":
            return coordinate - y_max
        if side == "LEFT":
            return x_min - coordinate
        return coordinate - x_max

    @staticmethod
    def _side_hint(
        *,
        orientation: str,
        raw: object,
    ) -> DimensionGraphicSide | None:
        value = str(raw or "").strip().upper()
        if orientation == "HORIZONTAL" and value in {"TOP", "BOTTOM"}:
            return value  # type: ignore[return-value]
        if orientation == "VERTICAL" and value in {"LEFT", "RIGHT"}:
            return value  # type: ignore[return-value]
        return None

    @staticmethod
    def _semantic_category(item: RawEvidence) -> str:
        return str(item.metadata.get("semantic_category") or "").strip().upper()

    @staticmethod
    def _measurement_value_m(item: RawEvidence) -> float | None:
        measurements = item.metadata.get("measurements")
        if not isinstance(measurements, list):
            return None
        for measurement in measurements:
            if not isinstance(measurement, dict):
                continue
            try:
                value = float(measurement.get("value"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value <= 0.0:
                continue
            unit = str(measurement.get("unit") or "").strip().upper()
            if unit in {"", "M"}:
                return value
            if unit == "CM":
                return value / 100.0
            if unit == "MM":
                return value / 1000.0
        return None
