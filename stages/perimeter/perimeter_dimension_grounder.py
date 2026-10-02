from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView

from .perimeter_models import (
    PerimeterDimensionGroundingResult,
    PerimeterDimensionReference,
    PerimeterMetricScale,
    PerimeterWallLayer,
)


class PerimeterDimensionGrounder:
    """
    Fase 02 — grounding dimensional de muros perimetrales.

    Relaciona cotas estructuradas de F01.5 con geometría px del perímetro.
    No recibe una escala externa y no crea px->m desde una sola cota.

    Reglas actuales de alta certeza:
        - Gemini está excluido del grounding del motor;
        - usa únicamente medidas estructuradas provenientes de PyMuPDF/OpenCV/OCR;
        - consume primero `RawEvidence.parameters` y consulta metadata/text crudo si falta estructura;
        - una cota GENERAL HORIZONTAL se asocia al span X completo;
        - una cota GENERAL VERTICAL se asocia al span Y completo;
        - una cota general puede fundamentar directamente wall_runs que
          coincidan exactamente con ese span;
        - una escala global solo se publica si existen al menos una relación
          HORIZONTAL y una VERTICAL y todas producen la misma relación m/px;
        - igualdad de ratios usa únicamente precisión numérica, no una
          tolerancia arquitectónica inventada;
        - cotas parciales/locales se conservan como no resueltas hasta que
          F02 tenga una asociación geométrica demostrable.
    """

    _GENERAL_SPAN_VALUES = frozenset({"GENERAL", "TOTAL", "OVERALL"})
    _HORIZONTAL_VALUES = frozenset({"HORIZONTAL", "H", "X"})
    _VERTICAL_VALUES = frozenset({"VERTICAL", "V", "Y"})
    _LENGTH_UNITS = frozenset({"M", "CM", "MM"})

    def ground(
        self,
        *,
        level_view: LevelView,
        wall_layer: PerimeterWallLayer,
        evidence: Sequence[RawEvidence],
    ) -> tuple[PerimeterWallLayer, PerimeterDimensionGroundingResult]:
        if wall_layer.level_view_id != level_view.id:
            raise ValueError("PerimeterWallLayer pertenece a otro LevelView.")

        raw = list(evidence)
        self._validate_evidence(level_view=level_view, evidence=raw)
        layer = wall_layer.model_copy(deep=True)

        motor_evidence = [item for item in raw if item.source != "GEMINI"]
        dimension_evidence = [
            item for item in motor_evidence if self._has_structured_dimension(item)
        ]
        declared_scale_evidence_ids = sorted(
            item.id
            for item in motor_evidence
            if self._contains_declared_scale(item)
        )

        references: list[PerimeterDimensionReference] = []
        unresolved_ids: set[str] = set()

        for item in dimension_evidence:
            extracted = self._dimension_references_for_evidence(
                evidence=item,
                wall_layer=layer,
                all_evidence=raw,
            )
            if extracted:
                references.extend(extracted)
            else:
                unresolved_ids.add(item.id)

        conflict_ids: set[str] = set()
        self._apply_direct_general_dimensions(
            wall_layer=layer,
            references=references,
            conflict_evidence_ids=conflict_ids,
        )

        metric_scale, scale_consistency, scale_conflict_ids = self._resolve_global_scale(
            references=references,
        )
        conflict_ids.update(scale_conflict_ids)

        if metric_scale is not None:
            self._apply_validated_scale(
                wall_layer=layer,
                metric_scale=metric_scale,
                conflict_evidence_ids=conflict_ids,
            )

        self._refresh_layer_metrics(layer)

        if conflict_ids:
            status = "CONFLICT"
        elif all(run.length_m is not None for run in layer.wall_runs):
            status = "GROUNDED"
        elif any(run.length_m is not None for run in layer.wall_runs):
            status = "PARTIAL"
        else:
            status = "UNRESOLVED"

        layer.metric_status = status
        layer.metric_scale = metric_scale

        notes: list[str] = []
        if not dimension_evidence:
            notes.append(
                "PyMuPDF/OpenCV/OCR no entregaron todavía cotas estructuradas con "
                "valor+unidad+orientación+span suficientes para grounding; la métrica permanece null."
            )
        if declared_scale_evidence_ids and metric_scale is None:
            notes.append(
                "Existe escala declarada en el plano, pero no se usa directamente para "
                "px->m sin una transformación geométrica fundamentada."
            )
        if scale_consistency == "SINGLE_REFERENCE":
            notes.append(
                "Solo existe una relación geométrica↔m utilizable; no se publica escala global."
            )
        if scale_consistency == "CONFLICT":
            notes.append(
                "Las asociaciones dimensionales no producen una única relación m/px; "
                "no se fuerza una escala global."
            )
        if metric_scale is not None:
            notes.append(
                "Escala global validada por asociaciones independientes horizontal y vertical."
            )

        grounding = PerimeterDimensionGroundingResult(
            level_view_id=level_view.id,
            status=status,
            scale_consistency=scale_consistency,
            dimension_references=references,
            metric_scale=metric_scale,
            declared_scale_evidence_ids=declared_scale_evidence_ids,
            unresolved_dimension_evidence_ids=sorted(unresolved_ids),
            conflict_evidence_ids=sorted(conflict_ids),
            notes=notes,
        )
        return layer, grounding

    def _dimension_references_for_evidence(
        self,
        *,
        evidence: RawEvidence,
        wall_layer: PerimeterWallLayer,
        all_evidence: Sequence[RawEvidence],
    ) -> list[PerimeterDimensionReference]:
        metadata = evidence.metadata
        measurements = self._structured_measurements(evidence)
        if not measurements:
            return []

        observation = metadata.get("raw_observation")
        observation = observation if isinstance(observation, dict) else {}

        orientation_default = self._normalize_orientation(
            metadata.get("orientation") or observation.get("orientation")
        )
        span_default = self._normalize_text(
            metadata.get("span_type") or observation.get("span_type")
        )
        reference_start_default = self._optional_text(
            metadata.get("reference_start") or observation.get("reference_start")
        )
        reference_end_default = self._optional_text(
            metadata.get("reference_end") or observation.get("reference_end")
        )

        bbox = wall_layer.polygon.bbox
        result: list[PerimeterDimensionReference] = []

        for index, measurement in enumerate(measurements):
            if not isinstance(measurement, dict):
                continue

            value_m = self._measurement_to_meters(measurement)
            if value_m is None:
                continue

            orientation = self._normalize_orientation(
                measurement.get("orientation") or orientation_default
            )
            span_type = self._normalize_text(
                measurement.get("span_type") or span_default
            )
            if orientation is None or span_type not in self._GENERAL_SPAN_VALUES:
                continue

            if orientation == "HORIZONTAL":
                geometry_target = "BOUNDING_SPAN_X"
                geometry_length_px = bbox.x_max - bbox.x_min
            else:
                geometry_target = "BOUNDING_SPAN_Y"
                geometry_length_px = bbox.y_max - bbox.y_min

            if geometry_length_px <= 0.0:
                continue

            visible_text = self._optional_text(
                measurement.get("text") or evidence.text
            )
            reference_start = self._optional_text(
                measurement.get("reference_start") or reference_start_default
            )
            reference_end = self._optional_text(
                measurement.get("reference_end") or reference_end_default
            )
            corroborating_ids = self._corroborating_text_evidence(
                visible_text=visible_text,
                source_evidence_id=evidence.id,
                evidence=all_evidence,
            )

            ref_id = self._reference_id(
                evidence_id=evidence.id,
                index=index,
                value_m=value_m,
                orientation=orientation,
                geometry_length_px=geometry_length_px,
            )
            result.append(
                PerimeterDimensionReference(
                    id=ref_id,
                    evidence_id=evidence.id,
                    visible_text=visible_text,
                    value_m=value_m,
                    orientation=orientation,
                    span_type=span_type,
                    reference_start=reference_start,
                    reference_end=reference_end,
                    geometry_target=geometry_target,
                    geometry_length_px=geometry_length_px,
                    meters_per_px=value_m / geometry_length_px,
                    state="GROUNDED",
                    grounded_wall_run_ids=[],
                    corroborating_evidence_ids=corroborating_ids,
                    notes=[
                        "Cota GENERAL asociada al span geométrico completo del perímetro."
                    ],
                )
            )

        return result

    def _structured_measurements(self, evidence: RawEvidence) -> list[dict[str, Any]]:
        """Extrae medidas del motor; nunca lee observaciones Gemini."""
        if evidence.source == "GEMINI":
            return []

        result: list[dict[str, Any]] = []
        raw_measurements = evidence.metadata.get("measurements")
        if isinstance(raw_measurements, list):
            result.extend(
                item.copy() for item in raw_measurements if isinstance(item, dict)
            )

        orientation = self._parameter_value(
            evidence, names={"orientation", "segment_orientation"}
        ) or evidence.metadata.get("orientation")
        span_type = self._parameter_value(
            evidence, names={"span_type"}
        ) or evidence.metadata.get("span_type")
        reference_start = self._parameter_value(
            evidence, names={"reference_start"}
        ) or evidence.metadata.get("reference_start")
        reference_end = self._parameter_value(
            evidence, names={"reference_end"}
        ) or evidence.metadata.get("reference_end")

        for parameter in getattr(evidence, "parameters", []):
            if str(parameter.group).upper() != "MEASUREMENT":
                continue
            if str(parameter.unit or "").upper() not in self._LENGTH_UNITS:
                continue
            result.append(
                {
                    "name": parameter.name,
                    "value": parameter.value,
                    "unit": parameter.unit,
                    "state": parameter.state,
                    "confidence": parameter.confidence,
                    "orientation": orientation,
                    "span_type": span_type,
                    "reference_start": reference_start,
                    "reference_end": reference_end,
                    "text": evidence.text,
                }
            )

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for measurement in result:
            signature = repr(sorted(measurement.items(), key=lambda item: item[0]))
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(measurement)
        return unique

    def _has_structured_dimension(self, evidence: RawEvidence) -> bool:
        return bool(self._structured_measurements(evidence))

    @staticmethod
    def _parameter_value(
        evidence: RawEvidence,
        *,
        names: set[str],
    ) -> object | None:
        for parameter in getattr(evidence, "parameters", []):
            if str(parameter.name) in names:
                return parameter.value
        return None

    def _apply_direct_general_dimensions(
        self,
        *,
        wall_layer: PerimeterWallLayer,
        references: Sequence[PerimeterDimensionReference],
        conflict_evidence_ids: set[str],
    ) -> None:
        bbox = wall_layer.polygon.bbox

        for run in wall_layer.wall_runs:
            matching = [
                ref
                for ref in references
                if self._reference_applies_to_full_span_run(
                    reference=ref,
                    run=run,
                    bbox=(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max),
                )
            ]
            if not matching:
                continue

            values = {ref.value_m for ref in matching}
            if len(values) != 1:
                run.metric_status = "CONFLICT"
                run.length_m = None
                run.dimensional_evidence_ids = sorted(ref.evidence_id for ref in matching)
                conflict_evidence_ids.update(run.dimensional_evidence_ids)
                for ref in matching:
                    ref.state = "CONFLICT"
                    ref.grounded_wall_run_ids.append(run.id)
                    ref.notes.append("Varias cotas GENERAL incompatibles aplican al mismo tramo.")
                continue

            value_m = next(iter(values))
            run.length_m = value_m
            run.metric_status = "GROUNDED"
            run.dimensional_evidence_ids = sorted({ref.evidence_id for ref in matching})
            for ref in matching:
                if run.id not in ref.grounded_wall_run_ids:
                    ref.grounded_wall_run_ids.append(run.id)

    @staticmethod
    def _reference_applies_to_full_span_run(
        *,
        reference: PerimeterDimensionReference,
        run: Any,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        x_min, y_min, x_max, y_max = bbox
        if reference.orientation == "HORIZONTAL":
            if run.drawing_orientation != "HORIZONTAL":
                return False
            return (
                min(run.start_px.x, run.end_px.x) == x_min
                and max(run.start_px.x, run.end_px.x) == x_max
            )

        if run.drawing_orientation != "VERTICAL":
            return False
        return (
            min(run.start_px.y, run.end_px.y) == y_min
            and max(run.start_px.y, run.end_px.y) == y_max
        )

    def _resolve_global_scale(
        self,
        *,
        references: Sequence[PerimeterDimensionReference],
    ) -> tuple[PerimeterMetricScale | None, str, set[str]]:
        grounded = [ref for ref in references if ref.state == "GROUNDED"]
        if not grounded:
            return None, "UNRESOLVED", set()

        orientations = {ref.orientation for ref in grounded}
        if len(grounded) == 1 or len(orientations) < 2:
            return None, "SINGLE_REFERENCE", set()

        baseline = grounded[0].meters_per_px
        all_equal = all(
            math.isclose(
                ref.meters_per_px,
                baseline,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for ref in grounded[1:]
        )

        if not all_equal:
            return None, "CONFLICT", {ref.evidence_id for ref in grounded}

        evidence_ids = sorted({ref.evidence_id for ref in grounded})
        reference_ids = sorted(ref.id for ref in grounded)
        return (
            PerimeterMetricScale(
                meters_per_px=baseline,
                source="DIMENSION_RECONCILIATION",
                evidence_ids=evidence_ids,
                dimension_reference_ids=reference_ids,
                description=(
                    "Relación px->m reconciliada con cotas GENERAL independientes "
                    "horizontal y vertical del propio LevelView."
                ),
            ),
            "CONSISTENT",
            set(),
        )

    @staticmethod
    def _apply_validated_scale(
        *,
        wall_layer: PerimeterWallLayer,
        metric_scale: PerimeterMetricScale,
        conflict_evidence_ids: set[str],
    ) -> None:
        for run in wall_layer.wall_runs:
            expected = run.length_px * metric_scale.meters_per_px
            if run.length_m is not None:
                if not math.isclose(
                    run.length_m,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    run.metric_status = "CONFLICT"
                    conflict_evidence_ids.update(run.dimensional_evidence_ids)
                    continue
            else:
                run.length_m = expected
                run.metric_status = "GROUNDED"
                run.dimensional_evidence_ids = sorted(
                    set(run.dimensional_evidence_ids) | set(metric_scale.evidence_ids)
                )

    @staticmethod
    def _refresh_layer_metrics(wall_layer: PerimeterWallLayer) -> None:
        if all(run.length_m is not None for run in wall_layer.wall_runs):
            wall_layer.total_length_m = sum(
                float(run.length_m) for run in wall_layer.wall_runs if run.length_m is not None
            )
        else:
            wall_layer.total_length_m = None

        for summary in wall_layer.drawing_side_summaries:
            selected = [
                run
                for run in wall_layer.wall_runs
                if run.drawing_side == summary.drawing_side
            ]
            if selected and all(run.length_m is not None for run in selected):
                summary.length_m = sum(
                    float(run.length_m) for run in selected if run.length_m is not None
                )
            else:
                summary.length_m = None

    @staticmethod
    def _measurement_to_meters(measurement: dict[str, Any]) -> float | None:
        unit = str(measurement.get("unit") or "").strip().upper()
        if unit not in PerimeterDimensionGrounder._LENGTH_UNITS:
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

    @staticmethod
    def _normalize_orientation(value: object) -> str | None:
        text = str(value or "").strip().upper()
        if text in PerimeterDimensionGrounder._HORIZONTAL_VALUES:
            return "HORIZONTAL"
        if text in PerimeterDimensionGrounder._VERTICAL_VALUES:
            return "VERTICAL"
        return None

    @staticmethod
    def _normalize_text(value: object) -> str:
        return " ".join(str(value or "").strip().upper().split())

    @staticmethod
    def _optional_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _contains_declared_scale(evidence: RawEvidence) -> bool:
        raw = evidence.metadata.get("raw_observation")
        texts = [evidence.text or ""]
        if isinstance(raw, dict):
            texts.extend(
                str(raw.get(key) or "")
                for key in ("description", "visible_text", "name")
            )
            for prop in raw.get("properties", []) if isinstance(raw.get("properties"), list) else []:
                if isinstance(prop, dict):
                    texts.append(str(prop.get("value_text") or ""))
        joined = " ".join(texts)
        return bool(re.search(r"\b1\s*:\s*\d+(?:\.\d+)?\b", joined))

    @staticmethod
    def _corroborating_text_evidence(
        *,
        visible_text: str | None,
        source_evidence_id: str,
        evidence: Sequence[RawEvidence],
    ) -> list[str]:
        if not visible_text:
            return []
        target = " ".join(visible_text.strip().upper().split())
        if not target:
            return []

        result: list[str] = []
        for item in evidence:
            if item.id == source_evidence_id or item.source == "GEMINI":
                continue
            if item.kind not in {"OCR_TEXT", "VECTOR_TEXT"} or not item.text:
                continue
            candidate = " ".join(item.text.strip().upper().split())
            if candidate == target:
                result.append(item.id)
        return sorted(result)

    @staticmethod
    def _reference_id(
        *,
        evidence_id: str,
        index: int,
        value_m: float,
        orientation: str,
        geometry_length_px: float,
    ) -> str:
        payload = (
            f"{evidence_id}|{index}|{value_m}|{orientation}|{geometry_length_px}"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"PERIMETER_DIMENSION_REF__{digest}"

    @staticmethod
    def _validate_evidence(
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> None:
        ids: set[str] = set()
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(f"RawEvidence {item.id} pertenece a otro LevelView.")
            if item.id in ids:
                raise ValueError(f"RawEvidence.id duplicado: {item.id}.")
            ids.add(item.id)
            if item.confirmed:
                raise ValueError(f"RawEvidence {item.id} llegó confirmed=True.")
