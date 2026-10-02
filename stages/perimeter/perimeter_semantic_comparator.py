from __future__ import annotations

import math
from collections.abc import Sequence

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView

from .perimeter_models import (
    PerimeterCandidate,
    PerimeterComparisonState,
    PerimeterEvidenceReconciliationResult,
    PerimeterSemanticComparisonResult,
    PerimeterCandidateSemanticComparison,
    PerimeterSemanticDimension,
)


class PerimeterSemanticComparator:
    """
    Auditoría independiente MOTOR F02 ↔ Gemini legacy.

    MOTOR:
        PyMuPDF + OpenCV + OCR reconciliados.

    GEMINI:
        baseline semántico independiente.

    Gemini no selecciona, completa ni corrige el perímetro del motor.
    La salida conserva ambos resultados y las diferencias observables para
    fine tuning.
    """

    GENERAL_SPANS = frozenset({"GENERAL", "TOTAL", "OVERALL"})
    PARTIAL_SPANS = frozenset({"TRAMO", "PARTIAL", "SEGMENT", "PARCIAL"})
    HORIZONTAL_VALUES = frozenset({"HORIZONTAL", "H", "X"})
    VERTICAL_VALUES = frozenset({"VERTICAL", "V", "Y"})
    LENGTH_UNITS = frozenset({"M", "CM", "MM"})

    def compare(
        self,
        *,
        level_view: LevelView,
        candidates: Sequence[PerimeterCandidate],
        evidence: Sequence[RawEvidence],
        selected_candidate_id: str | None,
        reconciliation: PerimeterEvidenceReconciliationResult | None = None,
    ) -> PerimeterSemanticComparisonResult:
        items = list(candidates)
        raw = list(evidence)
        self._validate_input(level_view=level_view, candidates=items, evidence=raw)

        gemini = [
            item
            for item in raw
            if item.source == "GEMINI" and item.kind == "GEMINI_OBSERVATION"
        ]
        if not gemini:
            return PerimeterSemanticComparisonResult(
                level_view_id=level_view.id,
                state="NOT_AVAILABLE",
                gemini_available=False,
                selected_candidate_id=selected_candidate_id,
                notes=[
                    "Gemini no estuvo disponible; F02 conserva el resultado propio del motor sin comparación semántica."
                ],
            )

        all_dimensions = self._dimensions(gemini)
        general_dimensions = [
            item
            for item in all_dimensions
            if str(item.span_type or "").upper() in self.GENERAL_SPANS
        ]

        if not all_dimensions:
            return PerimeterSemanticComparisonResult(
                level_view_id=level_view.id,
                state="UNRESOLVED",
                gemini_available=True,
                semantic_dimensions=[],
                semantic_chain_dimensions=[],
                selected_candidate_id=selected_candidate_id,
                notes=[
                    "Gemini respondió, pero no entregó cotas estructuradas utilizables para comparar el perímetro."
                ],
            )

        reconciliation_by_id = {
            item.candidate_id: item
            for item in (reconciliation.candidates if reconciliation is not None else [])
        }
        comparisons = [
            self._compare_candidate(
                candidate=candidate,
                general_dimensions=general_dimensions,
                all_dimensions=all_dimensions,
                reconciled=reconciliation_by_id.get(candidate.id),
            )
            for candidate in items
        ]

        selected_state: PerimeterComparisonState | None = None
        if selected_candidate_id is not None:
            selected = next(
                (item for item in comparisons if item.candidate_id == selected_candidate_id),
                None,
            )
            selected_state = selected.state if selected is not None else "UNRESOLVED"
            overall = selected_state
        elif not comparisons:
            overall = "UNRESOLVED"
        elif any(item.state == "MATCH" for item in comparisons):
            overall = "PARTIAL"
        elif any(item.state == "PARTIAL" for item in comparisons):
            overall = "PARTIAL"
        elif all(item.state == "CONFLICT" for item in comparisons):
            overall = "CONFLICT"
        else:
            overall = "UNRESOLVED"

        notes = [
            "Gemini se usa únicamente como baseline comparativo; no modifica el resultado del motor.",
            "Se publican además cadenas parciales Gemini y segmentos px del motor para fine tuning por tramo.",
        ]
        if selected_candidate_id is None and comparisons:
            notes.append(
                "El motor no cerró un candidato único; se comparan todos sin usar Gemini para seleccionar."
            )
        if any(item.relative_scale_difference is not None for item in comparisons):
            notes.append(
                "relative_scale_difference es diagnóstico; no implica un umbral arquitectónico de aceptación."
            )

        return PerimeterSemanticComparisonResult(
            level_view_id=level_view.id,
            state=overall,
            gemini_available=True,
            semantic_dimensions=general_dimensions,
            semantic_chain_dimensions=all_dimensions,
            candidate_comparisons=comparisons,
            selected_candidate_id=selected_candidate_id,
            selected_candidate_state=selected_state,
            notes=notes,
        )

    def _compare_candidate(
        self,
        *,
        candidate: PerimeterCandidate,
        general_dimensions: Sequence[PerimeterSemanticDimension],
        all_dimensions: Sequence[PerimeterSemanticDimension],
        reconciled: object | None,
    ) -> PerimeterCandidateSemanticComparison:
        bbox = candidate.geometry.bbox
        span_x = bbox.x_max - bbox.x_min
        span_y = bbox.y_max - bbox.y_min

        horizontal = sorted(
            {item.value_m for item in general_dimensions if item.orientation == "HORIZONTAL"}
        )
        vertical = sorted(
            {item.value_m for item in general_dimensions if item.orientation == "VERTICAL"}
        )

        motor_h: list[float] = []
        motor_v: list[float] = []
        if reconciled is not None and hasattr(reconciled, "segment_support"):
            for segment in reconciled.segment_support:
                if segment.orientation == "HORIZONTAL":
                    motor_h.append(segment.length_px)
                elif segment.orientation == "VERTICAL":
                    motor_v.append(segment.length_px)
        else:
            points = [(point.x, point.y) for point in candidate.geometry.points]
            for index, start in enumerate(points):
                end = points[(index + 1) % len(points)]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.hypot(dx, dy)
                if dy == 0.0 and length > 0.0:
                    motor_h.append(length)
                elif dx == 0.0 and length > 0.0:
                    motor_v.append(length)

        gemini_h_tramos = [
            item.value_m
            for item in all_dimensions
            if item.orientation == "HORIZONTAL"
            and str(item.span_type or "").upper() not in self.GENERAL_SPANS
        ]
        gemini_v_tramos = [
            item.value_m
            for item in all_dimensions
            if item.orientation == "VERTICAL"
            and str(item.span_type or "").upper() not in self.GENERAL_SPANS
        ]

        notes: list[str] = []
        if len(horizontal) > 1 or len(vertical) > 1:
            notes.append(
                "Gemini contiene más de un valor GENERAL para una misma orientación."
            )
            return PerimeterCandidateSemanticComparison(
                candidate_id=candidate.id,
                motor_span_x_px=span_x,
                motor_span_y_px=span_y,
                horizontal_value_m=horizontal[0] if len(horizontal) == 1 else None,
                vertical_value_m=vertical[0] if len(vertical) == 1 else None,
                motor_horizontal_segment_lengths_px=motor_h,
                motor_vertical_segment_lengths_px=motor_v,
                gemini_horizontal_tramos_m=gemini_h_tramos,
                gemini_vertical_tramos_m=gemini_v_tramos,
                state="CONFLICT",
                notes=notes,
            )

        h_value = horizontal[0] if horizontal else None
        v_value = vertical[0] if vertical else None
        h_scale = (h_value / span_x) if h_value is not None and span_x > 0.0 else None
        v_scale = (v_value / span_y) if v_value is not None and span_y > 0.0 else None

        if h_scale is None or v_scale is None:
            state: PerimeterComparisonState = "PARTIAL"
            notes.append(
                "Falta una cota GENERAL H/V Gemini; la comparación total es parcial."
            )
            difference = None
        else:
            denominator = max(h_scale, v_scale)
            difference = abs(h_scale - v_scale) / denominator if denominator > 0 else None
            if math.isclose(h_scale, v_scale, rel_tol=0.0, abs_tol=1e-12):
                state = "MATCH"
                notes.append(
                    "Las cotas GENERALES Gemini producen la misma relación m/px sobre los spans del candidato."
                )
            else:
                state = "CONFLICT"
                notes.append(
                    "Las cotas GENERALES Gemini producen relaciones m/px distintas sobre este candidato."
                )

        return PerimeterCandidateSemanticComparison(
            candidate_id=candidate.id,
            motor_span_x_px=span_x,
            motor_span_y_px=span_y,
            horizontal_value_m=h_value,
            vertical_value_m=v_value,
            horizontal_meters_per_px=h_scale,
            vertical_meters_per_px=v_scale,
            relative_scale_difference=difference,
            motor_horizontal_segment_lengths_px=motor_h,
            motor_vertical_segment_lengths_px=motor_v,
            gemini_horizontal_tramos_m=gemini_h_tramos,
            gemini_vertical_tramos_m=gemini_v_tramos,
            state=state,
            notes=notes,
        )

    def _dimensions(
        self,
        evidence: Sequence[RawEvidence],
    ) -> list[PerimeterSemanticDimension]:
        result: list[PerimeterSemanticDimension] = []
        seen: set[tuple[str, float, str, str | None, str | None]] = set()

        for item in evidence:
            if str(item.metadata.get("semantic_category", "")).upper() != "DIMENSION":
                continue

            default_orientation = self._orientation(item.metadata.get("orientation"))
            default_span = str(item.metadata.get("span_type") or "").strip().upper()
            measurements = item.metadata.get("measurements")
            if not isinstance(measurements, list):
                continue

            for measurement in measurements:
                if not isinstance(measurement, dict):
                    continue
                orientation = self._orientation(
                    measurement.get("orientation") or default_orientation
                )
                span_type = str(
                    measurement.get("span_type") or default_span or ""
                ).strip().upper()
                if orientation is None:
                    continue

                value_m = self._meters(measurement)
                if value_m is None:
                    continue

                reference_start = self._text(
                    measurement.get("reference_start")
                    or item.metadata.get("reference_start")
                )
                reference_end = self._text(
                    measurement.get("reference_end")
                    or item.metadata.get("reference_end")
                )
                key = (orientation, value_m, span_type, reference_start, reference_end)
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    PerimeterSemanticDimension(
                        evidence_id=item.id,
                        orientation=orientation,
                        value_m=value_m,
                        visible_text=self._text(measurement.get("text") or item.text),
                        span_type=span_type or None,
                        reference_start=reference_start,
                        reference_end=reference_end,
                    )
                )

        return result

    def _meters(self, measurement: dict[str, object]) -> float | None:
        unit = str(measurement.get("unit") or "").strip().upper()
        if unit not in self.LENGTH_UNITS:
            return None
        value = measurement.get("value")
        if isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0.0:
            return None
        if unit == "M":
            return parsed
        if unit == "CM":
            return parsed / 100.0
        return parsed / 1000.0

    def _orientation(self, value: object) -> str | None:
        normalized = str(value or "").strip().upper()
        if normalized in self.HORIZONTAL_VALUES:
            return "HORIZONTAL"
        if normalized in self.VERTICAL_VALUES:
            return "VERTICAL"
        return None

    @staticmethod
    def _text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _validate_input(
        *,
        level_view: LevelView,
        candidates: Sequence[PerimeterCandidate],
        evidence: Sequence[RawEvidence],
    ) -> None:
        for candidate in candidates:
            if candidate.level_view_id != level_view.id:
                raise ValueError("PerimeterCandidate pertenece a otro LevelView.")
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError("RawEvidence pertenece a otro LevelView.")
