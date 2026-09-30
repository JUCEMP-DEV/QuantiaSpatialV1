from __future__ import annotations

import math
import re
from statistics import median
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import EditablePerimeterModel
from app.quantia_spatialV1.reconstruction_core.general_dimension_graphic_span import (
    GeneralDimensionGraphicSpanResolver,
)
from app.quantia_spatialV1.reconstruction_core.reference_axis_resolver import AxisGridResolver


ScaleEvidenceState = Literal["RESOLVED", "REVIEW", "CONFLICT", "UNRESOLVED"]
ScaleEvidenceMethod = Literal[
    "F02_METRIC",
    "DIMENSION_GENERAL_GRAPHIC_SPAN",
    "DIMENSION_AXIS_FIT",
    "DIMENSION_RESOLVED_AXIS_SPAN",
    "DECLARED_SCALE_PDF",
    "DECLARED_SCALE_TEXT",
    "F01_GEMINI_SCALE_HINT",
]


class ScaleEvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    level_view_id: str = Field(min_length=1)
    method: ScaleEvidenceMethod
    m_per_px: float | None = Field(default=None, gt=0.0)
    declared_denominator: float | None = Field(default=None, gt=0.0)
    current_render_scale: float | None = Field(default=None, gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    semantic_orientation: Literal["HORIZONTAL", "VERTICAL"] | None = None
    graphic_side: Literal["TOP", "BOTTOM", "LEFT", "RIGHT"] | None = None
    axis_referenced: bool = False
    notes: list[str] = Field(default_factory=list)


class LevelScaleEvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    state: ScaleEvidenceState
    selected_m_per_px: float | None = Field(default=None, gt=0.0)
    selected_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    selected_method: ScaleEvidenceMethod | None = None
    current_render_scale: float | None = Field(default=None, gt=0.0)
    declared_scale_denominators: list[float] = Field(default_factory=list)
    candidates: list[ScaleEvidenceCandidate] = Field(default_factory=list)
    axis_fit_state: str = Field(default="UNRESOLVED", min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.state == "RESOLVED" and self.selected_m_per_px is not None


class ScaleEvidenceResolver:
    """Resuelve escala métrica sin depender de una única fuente.

    Prioridad conceptual:
    1. F02 metric_scale_m_per_px si ya está fundamentada.
    2. Si no existe 1:N: cota GENERAL + su línea gráfica exterior observada.
    3. Cadena de cotas + ajuste geométrico de ejes (px por metro).
    4. Cota métrica + dos ejes ya resueltos geométricamente.
    5. Escala declarada 1:N + transformación PDF->raster.
    6. Texto de escala sin transformación: solo REVIEW.

    Gemini puede aportar una pista de escala en F01, pero nunca se usa como
    verdad métrica aislada. El cierre se hace determinísticamente con la
    transformación PDF/raster o con cotas geométricamente reconciliadas.
    """

    VERSION = "SCALE_EVIDENCE_RESOLVER_V1_7"
    STRONG_CONSISTENCY_TOL = 0.05
    HARD_CONFLICT_TOL = 0.12

    _SCALE_RE = re.compile(
        r"(?i)(?:\besc(?:ala)?\.?\s*)?(?:\be\.?\s*)?1\s*[:/]\s*(\d{1,4}(?:[.,]\d+)?)"
    )

    def resolve(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
        perimeter: EditablePerimeterModel | None = None,
    ) -> LevelScaleEvidenceResult:
        raw = list(evidence)
        self._validate(level_view=level_view, evidence=raw, perimeter=perimeter)

        warnings: list[str] = []
        candidates: list[ScaleEvidenceCandidate] = []
        current_render_scale = self._observed_render_scale(raw)

        if perimeter is not None:
            metric_scale = perimeter.metric_summary.metric_scale_m_per_px
            if metric_scale is not None:
                candidates.append(
                    ScaleEvidenceCandidate(
                        id=f"{level_view.id}__SCALE__F02",
                        level_view_id=level_view.id,
                        method="F02_METRIC",
                        m_per_px=float(metric_scale),
                        current_render_scale=current_render_scale,
                        confidence=1.0,
                        evidence_ids=list(perimeter.metric_summary.dimension_evidence_ids),
                        notes=["Escala métrica ya fundamentada por F02."],
                    )
                )

        declared = self._declared_scale_observations(level_view=level_view, evidence=raw)
        declared_denominators = sorted({round(item[0], 8) for item in declared})

        for index, (denominator, confidence, source, evidence_ids) in enumerate(declared, start=1):
            method: ScaleEvidenceMethod = (
                "F01_GEMINI_SCALE_HINT" if source == "F01_GEMINI" else "DECLARED_SCALE_TEXT"
            )
            m_per_px: float | None = None
            notes = [f"Escala declarada detectada: 1:{denominator:g}."]
            if current_render_scale is not None:
                # PDF: 1 punto = 1/72 in. El raster usa px_por_pt ~= render_scale.
                # En 1:N, 1 pt dibujado representa N/72 pulgadas reales.
                m_per_px = denominator * 0.0254 / (72.0 * current_render_scale)
                method = "DECLARED_SCALE_PDF"
                notes.append("Convertida mediante transformación PDF->raster observada.")
            else:
                notes.append("Sin transformación PDF->raster: se conserva solo como pista.")

            candidates.append(
                ScaleEvidenceCandidate(
                    id=f"{level_view.id}__SCALE__DECLARED_{index:02d}",
                    level_view_id=level_view.id,
                    method=method,
                    m_per_px=m_per_px,
                    declared_denominator=denominator,
                    current_render_scale=current_render_scale,
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                    notes=notes,
                )
            )

        # Fallback gráfico dirigido al caso objetivo: cuando no existe 1:N
        # observable por F01/PyMuPDF/OCR, una cota GENERAL puede cerrar escala
        # midiendo su propia línea exterior. Si sí existe escala declarada, se
        # conserva el comportamiento V1.2 para no alterar casos ya estables.
        graphic_spans = (
            GeneralDimensionGraphicSpanResolver().resolve(
                level_view=level_view,
                evidence=raw,
                perimeter=perimeter,
            )
            if not declared_denominators
            else []
        )
        for index, span in enumerate(graphic_spans, start=1):
            candidates.append(
                ScaleEvidenceCandidate(
                    id=f"{level_view.id}__SCALE__GRAPHIC_{index:02d}",
                    level_view_id=level_view.id,
                    method="DIMENSION_GENERAL_GRAPHIC_SPAN",
                    m_per_px=span.m_per_px,
                    current_render_scale=current_render_scale,
                    confidence=span.confidence,
                    evidence_ids=[
                        span.dimension_evidence_id,
                        *list(span.geometric_evidence_ids),
                    ],
                    semantic_orientation=span.semantic_orientation,
                    graphic_side=span.side,
                    axis_referenced=span.axis_referenced,
                    notes=[
                        "Cota GENERAL medida directamente sobre línea gráfica exterior.",
                        f"orientation={span.semantic_orientation}",
                        f"side={span.side}",
                        f"value_m={span.value_m:.6g}",
                        f"span_px={span.span_px:.3f}",
                        f"line_coordinate_px={span.line_coordinate_px:.3f}",
                        f"graphic_score={span.score:.3f}",
                        f"graphic_source={span.source}",
                        f"axis_referenced={span.axis_referenced}",
                        *(
                            [
                                "Cota entre ejes: la asociación gráfica relativa a F02 es solo diagnóstica; "
                                "la escala debe cerrarse con las coordenadas de los ejes referenciados."
                            ]
                            if span.axis_referenced
                            else []
                        ),
                    ],
                )
            )

        axis_resolution = AxisGridResolver().resolve(level_view=level_view, evidence=raw)
        for index, fit in enumerate(axis_resolution.diagnostics.fits, start=1):
            if fit.state != "RESOLVED" or fit.affine_pixels_per_semantic_unit is None:
                continue
            px_per_m = float(fit.affine_pixels_per_semantic_unit)
            if not math.isfinite(px_per_m) or px_per_m <= 0.0:
                continue
            m_per_px = 1.0 / px_per_m
            confidence = self._fit_confidence(fit)
            notes = [
                f"Cadena {fit.semantic_orientation} reconciliada con geometría.",
                f"matched_fraction={fit.matched_fraction:.3f}",
            ]
            if fit.general_span_consistent is False:
                notes.append("La cota general discrepa de la suma de tramos.")
            candidates.append(
                ScaleEvidenceCandidate(
                    id=f"{level_view.id}__SCALE__AXISFIT_{index:02d}",
                    level_view_id=level_view.id,
                    method="DIMENSION_AXIS_FIT",
                    m_per_px=m_per_px,
                    current_render_scale=current_render_scale,
                    confidence=confidence,
                    evidence_ids=list(fit.chain_dimension_evidence_ids),
                    semantic_orientation=fit.semantic_orientation,
                    notes=notes,
                )
            )

        candidates.extend(
            self._resolved_axis_span_candidates(
                level_view=level_view,
                evidence=raw,
                axis_resolution=axis_resolution,
                current_render_scale=current_render_scale,
            )
        )

        numeric = [item for item in candidates if item.m_per_px is not None]
        if not numeric:
            state: ScaleEvidenceState = "REVIEW" if declared_denominators else "UNRESOLVED"
            return LevelScaleEvidenceResult(
                level_view_id=level_view.id,
                state=state,
                selected_m_per_px=None,
                selected_confidence=0.0,
                selected_method=None,
                current_render_scale=current_render_scale,
                declared_scale_denominators=declared_denominators,
                candidates=candidates,
                axis_fit_state=axis_resolution.state,
                warnings=warnings,
            )

        clusters = self._cluster_candidates(numeric)
        if declared_denominators or any(item.method == "F02_METRIC" for item in numeric):
            # Con 1:N o F02 métrico ya existe una referencia externa al patrón
            # geométrico; se conserva la selección histórica de V1.4.
            best = max(
                clusters,
                key=lambda cluster: (
                    sum(item.confidence for item in cluster),
                    max(self._method_rank(item.method) for item in cluster),
                    len(cluster),
                ),
            )
        else:
            # Sin referencia externa, no se permite que una cadena larga de
            # AXISSPAN correlacionados "vote" muchas veces por la misma hipótesis.
            best = self._select_best_cluster(clusters)
        governing_best = [
            item
            for item in best
            if not (item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN" and item.axis_referenced)
        ] or list(best)
        selected_value = self._weighted_mean(governing_best)
        selected_candidate = max(
            governing_best,
            key=lambda item: (self._method_rank(item.method), item.confidence),
        )
        selected_conf = min(
            1.0,
            max(item.confidence for item in governing_best)
            + 0.05 * max(0, len(governing_best) - 1),
        )

        direct_graphic_orientations = {
            item.semantic_orientation
            for item in best
            if item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
            and item.confidence >= 0.90
            and item.semantic_orientation is not None
        }
        direct_graphic_crosscheck = direct_graphic_orientations == {"HORIZONTAL", "VERTICAL"}

        conflicting_strong = [
            item
            for item in numeric
            if item not in best
            and item.confidence >= 0.75
            and self._relative_delta(float(item.m_per_px), selected_value) > self.HARD_CONFLICT_TOL
        ]

        blocking_conflicts: list[ScaleEvidenceCandidate] = []
        downgraded_conflicts: list[ScaleEvidenceCandidate] = []
        best_consensus_orientation = (
            self._cross_family_consensus_orientation(best)
            if not declared_denominators
            else None
        )
        internally_conflicted_orientations = (
            {
                orientation
                for orientation in ("HORIZONTAL", "VERTICAL")
                if self._orientation_has_internal_geometry_conflict(
                    candidates=numeric,
                    orientation=orientation,
                )
            }
            if not declared_denominators
            else set()
        )

        for item in conflicting_strong:
            derived_conflict_under_hv_graphic_consensus = (
                direct_graphic_crosscheck
                and item.method
                in {
                    "DIMENSION_AXIS_FIT",
                    "DIMENSION_RESOLVED_AXIS_SPAN",
                    "DECLARED_SCALE_PDF",
                }
            )
            opposite_orientation_is_self_inconsistent = (
                best_consensus_orientation is not None
                and item.semantic_orientation is not None
                and item.semantic_orientation != best_consensus_orientation
                and item.semantic_orientation in internally_conflicted_orientations
                and item.method in {
                    "DIMENSION_AXIS_FIT",
                    "DIMENSION_RESOLVED_AXIS_SPAN",
                }
            )

            if derived_conflict_under_hv_graphic_consensus or opposite_orientation_is_self_inconsistent:
                downgraded_conflicts.append(item)
            else:
                blocking_conflicts.append(item)

        if downgraded_conflicts:
            if direct_graphic_crosscheck:
                warnings.append(
                    "La escala gráfica GENERAL H/V converge, pero existe evidencia derivada incompatible "
                    "(ejes o escala declarada). Se conserva como diagnóstico y no bloquea el cierre métrico."
                )
            if best_consensus_orientation is not None and internally_conflicted_orientations:
                conflicted = ", ".join(sorted(internally_conflicted_orientations))
                warnings.append(
                    f"La orientación {conflicted} presenta conflicto interno entre línea gráfica GENERAL "
                    "y geometría de ejes. Se prioriza la orientación con consenso entre familias "
                    f"independientes ({best_consensus_orientation}) y la discrepancia queda en diagnóstico."
                )

        if blocking_conflicts:
            state = "CONFLICT"
            warnings.append(
                "Existen fuentes fuertes de escala incompatibles; no debe publicarse una escala final sin revisión."
            )
        elif selected_conf >= 0.75:
            state = "RESOLVED"
        else:
            state = "REVIEW"

        return LevelScaleEvidenceResult(
            level_view_id=level_view.id,
            state=state,
            selected_m_per_px=selected_value,
            selected_confidence=selected_conf,
            selected_method=selected_candidate.method,
            current_render_scale=current_render_scale,
            declared_scale_denominators=declared_denominators,
            candidates=candidates,
            axis_fit_state=axis_resolution.state,
            warnings=warnings,
        )

    def _resolved_axis_span_candidates(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
        axis_resolution: Any,
        current_render_scale: float | None,
    ) -> list[ScaleEvidenceCandidate]:
        """Fallback cuando existe retícula geométrica pero no ajuste afín de cadena.

        Usa una cota métrica SOLO si sus dos referencias de eje ya tienen
        coordenadas px resueltas por AxisGridResolver. No infiere etiquetas, no
        usa dimensiones del predio y no convierte una cota sin anclaje en escala.
        """
        axes = {
            (str(axis.orientation), self._clean_axis_label(axis.label)): axis
            for axis in axis_resolution.axes
            if self._clean_axis_label(axis.label)
        }
        if len(axes) < 2:
            return []

        min_dim = float(min(level_view.raster_width_px, level_view.raster_height_px))
        min_span_px = max(8.0, 0.004 * min_dim)
        results: list[ScaleEvidenceCandidate] = []

        for item in evidence:
            category = str(item.metadata.get("semantic_category") or "").strip().upper()
            if category != "DIMENSION":
                continue

            semantic_orientation = str(item.metadata.get("orientation") or "").strip().upper()
            if semantic_orientation not in {"HORIZONTAL", "VERTICAL"}:
                continue
            start_label = self._clean_axis_label(item.metadata.get("reference_start"))
            end_label = self._clean_axis_label(item.metadata.get("reference_end"))
            if not start_label or not end_label or start_label == end_label:
                continue

            value_m = self._dimension_value_m(item)
            if value_m is None or value_m <= 0.0:
                continue

            axis_orientation = "vertical" if semantic_orientation == "HORIZONTAL" else "horizontal"
            axis_a = axes.get((axis_orientation, start_label))
            axis_b = axes.get((axis_orientation, end_label))
            if axis_a is None or axis_b is None:
                continue

            span_px = abs(float(axis_b.coordinate_px) - float(axis_a.coordinate_px))
            if not math.isfinite(span_px) or span_px < min_span_px:
                continue

            m_per_px = float(value_m) / span_px
            if not math.isfinite(m_per_px) or m_per_px <= 0.0:
                continue

            span_type = str(item.metadata.get("span_type") or "").strip().upper()
            confidence = self._axis_span_confidence(
                item=item,
                axis_a=axis_a,
                axis_b=axis_b,
                span_px=span_px,
                min_dim=min_dim,
                span_type=span_type,
            )
            evidence_ids = [item.id]
            for axis in (axis_a, axis_b):
                evidence_ids.extend(list(axis.source_axis_evidence_ids))
                evidence_ids.extend(list(axis.source_geometric_evidence_ids))

            results.append(
                ScaleEvidenceCandidate(
                    id=f"{level_view.id}__SCALE__AXISSPAN_{len(results) + 1:02d}",
                    level_view_id=level_view.id,
                    method="DIMENSION_RESOLVED_AXIS_SPAN",
                    m_per_px=m_per_px,
                    current_render_scale=current_render_scale,
                    confidence=confidence,
                    evidence_ids=sorted(set(evidence_ids)),
                    semantic_orientation=semantic_orientation,
                    notes=[
                        f"Cota {semantic_orientation} {start_label}->{end_label} anclada a ejes resueltos.",
                        f"span_type={span_type or 'NO_DECLARADO'}",
                        f"value_m={value_m:.6g}",
                        f"span_px={span_px:.3f}",
                    ],
                )
            )

        return results

    @staticmethod
    def _clean_axis_label(value: object) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _dimension_value_m(item: RawEvidence) -> float | None:
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

    @staticmethod
    def _axis_span_confidence(
        *,
        item: RawEvidence,
        axis_a: Any,
        axis_b: Any,
        span_px: float,
        min_dim: float,
        span_type: str,
    ) -> float:
        dimension_confidence = float(item.confidence if item.confidence is not None else 0.80)
        axis_confidences = [
            float(value)
            for value in (axis_a.confidence, axis_b.confidence)
            if value is not None
        ]
        axis_confidence = min(axis_confidences) if axis_confidences else 0.55
        span_quality = max(0.0, min(1.0, span_px / max(24.0, 0.08 * min_dim)))
        confidence = (
            0.45 * max(0.0, min(1.0, dimension_confidence))
            + 0.40 * max(0.0, min(1.0, axis_confidence))
            + 0.15 * span_quality
        )
        if span_type == "GENERAL":
            confidence += 0.05
            return max(0.0, min(0.99, confidence))
        # Un único tramo corto no debe resolver por sí solo una escala global.
        # Dos o más tramos concordantes sí pueden superar el umbral mediante
        # la consolidación de candidatos del resolver.
        return max(0.0, min(0.72, confidence))

    def _declared_scale_observations(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> list[tuple[float, float, str, list[str]]]:
        found: list[tuple[float, float, str, list[str]]] = []

        for item in level_view.evidence:
            if str(item.reference or "").strip().lower() != "declared_scale_hint":
                continue
            denominator = self._parse_denominator(item.text)
            if denominator is None:
                continue
            found.append(
                (
                    denominator,
                    float(item.confidence if item.confidence is not None else 0.82),
                    "F01_GEMINI",
                    [],
                )
            )

        for item in evidence:
            category = str(item.metadata.get("semantic_category") or "").strip().upper()
            source_conf = {
                "PYMUPDF": 0.97,
                "OCR": 0.80,
                "GEMINI": 0.86,
                "OPENCV": 0.55,
            }.get(item.source, 0.70)
            texts: list[str] = []
            if item.text:
                texts.append(str(item.text))

            properties = item.metadata.get("properties")
            if isinstance(properties, list):
                for prop in properties:
                    if not isinstance(prop, dict):
                        continue
                    name = str(prop.get("name") or "").strip().lower()
                    if name in {"scale_text", "declared_scale", "scale_ratio"}:
                        value = prop.get("value")
                        if value is not None:
                            texts.append(str(value))

            if category not in {"SCALE", "DOCUMENT"} and item.source not in {"PYMUPDF", "OCR"}:
                continue

            for text in texts:
                denominator = self._parse_denominator(text)
                if denominator is None:
                    continue
                confidence = float(item.confidence if item.confidence is not None else source_conf)
                found.append((denominator, min(confidence, source_conf), item.source, [item.id]))

        # Deduplicación por denominador+fuente conservando mayor confianza.
        best: dict[tuple[float, str], tuple[float, float, str, list[str]]] = {}
        for row in found:
            key = (round(row[0], 8), row[2])
            previous = best.get(key)
            if previous is None or row[1] > previous[1]:
                best[key] = row
        return list(best.values())

    @classmethod
    def _parse_denominator(cls, text: object) -> float | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        match = cls._SCALE_RE.search(raw)
        if match is None:
            return None
        try:
            value = float(match.group(1).replace(",", "."))
        except ValueError:
            return None
        if value < 5.0 or value > 1000.0:
            return None
        return value

    @staticmethod
    def _observed_render_scale(evidence: Sequence[RawEvidence]) -> float | None:
        x_values: list[float] = []
        y_values: list[float] = []
        for item in evidence:
            for key, target in (
                ("page_to_raster_scale_x", x_values),
                ("page_to_raster_scale_y", y_values),
            ):
                try:
                    value = float(item.metadata.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    target.append(value)
        if not x_values and not y_values:
            return None
        x = float(median(x_values)) if x_values else None
        y = float(median(y_values)) if y_values else None
        if x is not None and y is not None:
            mean = (x + y) / 2.0
            if mean <= 0.0 or abs(x - y) / mean > 0.03:
                return None
            return mean
        return x if x is not None else y

    @staticmethod
    def _fit_confidence(fit: Any) -> float:
        coverage = max(0.0, min(1.0, float(fit.matched_fraction or 0.0)))
        residual_quality = 0.5
        if fit.median_residual_px is not None and fit.match_tolerance_px > 0:
            residual_quality = max(
                0.0,
                min(1.0, 1.0 - float(fit.median_residual_px) / float(fit.match_tolerance_px)),
            )
        confidence = 0.55 + 0.30 * coverage + 0.15 * residual_quality
        if fit.general_span_consistent is False:
            confidence -= 0.15
        return max(0.0, min(1.0, confidence))

    def _select_best_cluster(
        self,
        clusters: Sequence[Sequence[ScaleEvidenceCandidate]],
    ) -> list[ScaleEvidenceCandidate]:
        """Selecciona soporte independiente sin contar derivados correlacionados por cantidad."""
        if not clusters:
            raise ValueError("No hay clusters de escala para seleccionar.")
        return max(
            (list(cluster) for cluster in clusters),
            key=lambda cluster: (
                self._independent_support_score(cluster),
                max(self._method_rank(item.method) for item in cluster),
                max(item.confidence for item in cluster),
                len(cluster),
            ),
        )

    @staticmethod
    def _independent_support_score(
        cluster: Sequence[ScaleEvidenceCandidate],
    ) -> float:
        buckets: dict[tuple[str, str, str], float] = {}
        for item in cluster:
            orientation = str(item.semantic_orientation or "NONE")
            side = str(item.graphic_side or "NONE")
            if item.method in {"DIMENSION_RESOLVED_AXIS_SPAN", "DIMENSION_AXIS_FIT"}:
                key = ("AXIS_GEOMETRY", orientation, "CHAIN")
            elif item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN":
                key = (item.method, orientation, side)
            else:
                key = (item.method, orientation, side)
            buckets[key] = max(buckets.get(key, 0.0), float(item.confidence))
        return sum(buckets.values())

    @staticmethod
    def _cross_family_consensus_orientation(
        cluster: Sequence[ScaleEvidenceCandidate],
    ) -> Literal["HORIZONTAL", "VERTICAL"] | None:
        resolved: list[Literal["HORIZONTAL", "VERTICAL"]] = []
        for orientation in ("HORIZONTAL", "VERTICAL"):
            has_graphic = any(
                item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
                and item.semantic_orientation == orientation
                and item.confidence >= 0.65
                for item in cluster
            )
            has_axis = any(
                item.method in {"DIMENSION_AXIS_FIT", "DIMENSION_RESOLVED_AXIS_SPAN"}
                and item.semantic_orientation == orientation
                and item.confidence >= 0.85
                for item in cluster
            )
            if has_graphic and has_axis:
                resolved.append(orientation)
        if len(resolved) == 1:
            return resolved[0]
        return None

    def _orientation_has_internal_geometry_conflict(
        self,
        *,
        candidates: Sequence[ScaleEvidenceCandidate],
        orientation: Literal["HORIZONTAL", "VERTICAL"],
    ) -> bool:
        graphics = [
            item
            for item in candidates
            if item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
            and item.semantic_orientation == orientation
            and item.confidence >= 0.65
            and item.m_per_px is not None
        ]
        axis_family = [
            item
            for item in candidates
            if item.method in {"DIMENSION_AXIS_FIT", "DIMENSION_RESOLVED_AXIS_SPAN"}
            and item.semantic_orientation == orientation
            and item.confidence >= 0.85
            and item.m_per_px is not None
        ]
        if not graphics or not axis_family:
            return False
        graphic_value = self._weighted_mean(graphics)
        axis_value = self._weighted_mean(axis_family)
        return self._relative_delta(graphic_value, axis_value) > self.HARD_CONFLICT_TOL

    def _cluster_candidates(
        self,
        candidates: Sequence[ScaleEvidenceCandidate],
    ) -> list[list[ScaleEvidenceCandidate]]:
        ordered = sorted(candidates, key=lambda item: float(item.m_per_px or 0.0))
        clusters: list[list[ScaleEvidenceCandidate]] = []
        for item in ordered:
            value = float(item.m_per_px)
            placed = False
            for cluster in clusters:
                center = self._weighted_mean(cluster)
                if self._relative_delta(value, center) <= self.STRONG_CONSISTENCY_TOL:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])
        return clusters

    @staticmethod
    def _weighted_mean(candidates: Sequence[ScaleEvidenceCandidate]) -> float:
        total_weight = sum(max(0.05, item.confidence) for item in candidates)
        return sum(float(item.m_per_px) * max(0.05, item.confidence) for item in candidates) / total_weight

    @staticmethod
    def _relative_delta(a: float, b: float) -> float:
        denominator = max(abs(a), abs(b), 1e-12)
        return abs(a - b) / denominator

    @staticmethod
    def _method_rank(method: ScaleEvidenceMethod) -> int:
        return {
            "F02_METRIC": 7,
            "DIMENSION_GENERAL_GRAPHIC_SPAN": 6,
            "DIMENSION_AXIS_FIT": 5,
            "DIMENSION_RESOLVED_AXIS_SPAN": 4,
            "DECLARED_SCALE_PDF": 3,
            "F01_GEMINI_SCALE_HINT": 2,
            "DECLARED_SCALE_TEXT": 1,
        }[method]

    @staticmethod
    def _validate(
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
        perimeter: EditablePerimeterModel | None,
    ) -> None:
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError("ScaleEvidenceResolver recibió evidencia de otro LevelView.")
        if perimeter is not None and perimeter.level_view_id != level_view.id:
            raise ValueError("ScaleEvidenceResolver recibió un perímetro de otro LevelView.")
