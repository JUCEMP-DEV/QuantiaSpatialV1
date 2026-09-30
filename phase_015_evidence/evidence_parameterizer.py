from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from app.quantia_spatialV1.models.evidence import (
    EvidenceParameter,
    RawEvidence,
)


class EvidenceParameterizer:
    """
    Fase 01.5 — parametrización general de RawEvidence.

    Responsabilidad:
        - recibe evidencia ya extraída por PyMuPDF/OpenCV/OCR/Gemini;
        - NO modifica ni elimina geometry/text/metadata;
        - construye `parameters[]` con nombres comunes y trazables;
        - no decide objetos arquitectónicos finales;
        - no convierte px -> m;
        - no usa Gemini para reinterpretar las demás fuentes.

    F02 consume primero `parameters`. Si necesita mayor detalle puede volver
    a geometry/text/metadata mediante `source_path`.
    """

    def parameterize_many(
        self,
        evidence: Iterable[RawEvidence],
    ) -> list[RawEvidence]:
        return [self.parameterize(item) for item in evidence]

    def parameterize(self, evidence: RawEvidence) -> RawEvidence:
        parameters: list[EvidenceParameter] = []

        self._append_common(evidence=evidence, target=parameters)

        if evidence.source == "PYMUPDF":
            self._append_pymupdf(evidence=evidence, target=parameters)
        elif evidence.source == "OPENCV":
            self._append_opencv(evidence=evidence, target=parameters)
        elif evidence.source == "OCR":
            self._append_ocr(evidence=evidence, target=parameters)
        elif evidence.source == "GEMINI":
            self._append_gemini(evidence=evidence, target=parameters)

        # RawEvidence conserva geometry/text/metadata como evidencia cruda e inmutable
        # durante F01.5. Solo sustituimos `parameters`; no existe razón para
        # deep-copy de toda la metadata/geométrica por cada elemento. En planos
        # densos (p. ej. CAD con miles de primitivas) deep=True multiplica el
        # costo de memoria/CPU y puede bloquear la parametrización.
        return evidence.model_copy(
            update={"parameters": self._deduplicate(parameters)},
            deep=False,
        )

    # ========================================================
    # COMÚN
    # ========================================================

    def _append_common(
        self,
        *,
        evidence: RawEvidence,
        target: list[EvidenceParameter],
    ) -> None:
        self._add(
            target,
            group="EVIDENCE",
            name="source",
            value=evidence.source,
            source_path="source",
        )
        self._add(
            target,
            group="EVIDENCE",
            name="kind",
            value=evidence.kind,
            source_path="kind",
        )
        self._add(
            target,
            group="GEOMETRY",
            name="geometry_type",
            value=evidence.geometry.geometry_type,
            source_path="geometry.geometry_type",
        )

        if evidence.text is not None:
            self._add(
                target,
                group="TEXT",
                name="visible_text",
                value=evidence.text,
                confidence=evidence.confidence,
                source_path="text",
            )

        if evidence.confidence is not None:
            self._add(
                target,
                group="QUALITY",
                name="confidence",
                value=evidence.confidence,
                source_path="confidence",
            )

        points = evidence.geometry.points
        if points:
            self._add(
                target,
                group="GEOMETRY",
                name="points_px",
                value=[{"x": point.x, "y": point.y} for point in points],
                unit="PX",
                source_path="geometry.points",
            )

        bbox = evidence.geometry.bbox_px
        if bbox is not None:
            self._add(
                target,
                group="GEOMETRY",
                name="bbox_px",
                value={
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                },
                unit="PX",
                source_path="geometry.bbox_px",
            )
            self._add(
                target,
                group="GEOMETRY",
                name="bbox_width_px",
                value=bbox.x_max - bbox.x_min,
                unit="PX",
                source_path="geometry.bbox_px",
            )
            self._add(
                target,
                group="GEOMETRY",
                name="bbox_height_px",
                value=bbox.y_max - bbox.y_min,
                unit="PX",
                source_path="geometry.bbox_px",
            )

        if evidence.geometry.geometry_type == "SEGMENT" and len(points) == 2:
            p0, p1 = points
            dx = float(p1.x - p0.x)
            dy = float(p1.y - p0.y)
            length_px = math.hypot(dx, dy)

            if dy == 0.0 and dx != 0.0:
                orientation = "HORIZONTAL"
            elif dx == 0.0 and dy != 0.0:
                orientation = "VERTICAL"
            else:
                orientation = "OTHER"

            self._add(
                target,
                group="GEOMETRY",
                name="segment_length_px",
                value=length_px,
                unit="PX",
                source_path="geometry.points",
            )
            self._add(
                target,
                group="GEOMETRY",
                name="segment_orientation",
                value=orientation,
                source_path="geometry.points",
            )

    # ========================================================
    # PYMUPDF
    # ========================================================

    def _append_pymupdf(
        self,
        *,
        evidence: RawEvidence,
        target: list[EvidenceParameter],
    ) -> None:
        metadata = evidence.metadata

        self._add_metadata(
            target,
            metadata,
            group="VECTOR_SOURCE",
            mappings=(
                ("vector_command", "vector_command", None),
                ("primitive_type", "primitive_type", None),
                ("drawing_index", "drawing_index", None),
                ("item_index", "item_index", None),
                ("segment_index", "segment_index", None),
            ),
        )

        drawing = metadata.get("drawing")
        if isinstance(drawing, dict):
            for raw_name, parameter_name in (
                ("drawing_type", "drawing_type"),
                ("stroke_color", "stroke_color"),
                ("fill_color", "fill_color"),
                ("stroke_width", "stroke_width"),
                ("dashes", "dashes"),
                ("line_cap", "line_cap"),
                ("line_join", "line_join"),
                ("close_path", "close_path"),
            ):
                self._add(
                    target,
                    group="VECTOR_STYLE",
                    name=parameter_name,
                    value=drawing.get(raw_name),
                    source_path=f"metadata.drawing.{raw_name}",
                )

        self._add_metadata(
            target,
            metadata,
            group="TRANSFORM",
            mappings=(
                ("page_to_raster_scale_x", "page_to_raster_scale_x", None),
                ("page_to_raster_scale_y", "page_to_raster_scale_y", None),
                ("clipped_to_level", "clipped_to_level", None),
            ),
        )

        if evidence.kind == "VECTOR_TEXT":
            for raw_name, parameter_name in (
                ("font", "font"),
                ("font_size", "font_size"),
                ("font_flags", "font_flags"),
                ("char_flags", "char_flags"),
                ("origin", "origin"),
                ("ascender", "ascender"),
                ("descender", "descender"),
                ("line_direction", "line_direction"),
                ("line_wmode", "line_wmode"),
            ):
                self._add(
                    target,
                    group="TEXT_STYLE",
                    name=parameter_name,
                    value=metadata.get(raw_name),
                    source_path=f"metadata.{raw_name}",
                )

    # ========================================================
    # OPENCV
    # ========================================================

    def _append_opencv(
        self,
        *,
        evidence: RawEvidence,
        target: list[EvidenceParameter],
    ) -> None:
        metadata = evidence.metadata

        if evidence.kind == "RASTER_LINE":
            self._add_metadata(
                target,
                metadata,
                group="RASTER_GEOMETRY",
                mappings=(
                    ("stage", "stage", None),
                    ("source_reference", "source_reference", None),
                    ("detector", "detector", None),
                    ("orientation", "orientation", None),
                    ("angle_deg", "angle", "DEG"),
                    ("length_px", "length", "PX"),
                    ("merged_from", "merged_from", None),
                ),
            )

        elif evidence.kind == "RASTER_INTERSECTION":
            self._add_metadata(
                target,
                metadata,
                group="TOPOLOGY",
                mappings=(
                    ("source_reference", "source_reference", None),
                    ("horizontal_segment_id", "horizontal_segment_id", None),
                    ("vertical_segment_id", "vertical_segment_id", None),
                ),
            )

        elif evidence.kind == "RASTER_CONTOUR":
            self._add_metadata(
                target,
                metadata,
                group="RASTER_GEOMETRY",
                mappings=(
                    ("source_reference", "source_reference", None),
                    ("area_px2", "area", "PX2"),
                    ("perimeter_px", "perimeter", "PX"),
                    ("closed", "closed", None),
                    ("contour_retrieval_mode", "contour_retrieval_mode", None),
                    ("contour_chain_mode", "contour_chain_mode", None),
                ),
            )

            hierarchy = metadata.get("hierarchy")
            if isinstance(hierarchy, dict):
                for name in ("next", "previous", "child", "parent"):
                    self._add(
                        target,
                        group="TOPOLOGY",
                        name=f"contour_{name}",
                        value=hierarchy.get(name),
                        source_path=f"metadata.hierarchy.{name}",
                    )

            raw_bbox = metadata.get("raw_bbox_px")
            if isinstance(raw_bbox, dict):
                for name in ("x", "y", "width", "height"):
                    self._add(
                        target,
                        group="RASTER_GEOMETRY",
                        name=f"raw_bbox_{name}_px",
                        value=raw_bbox.get(name),
                        unit="PX",
                        source_path=f"metadata.raw_bbox_px.{name}",
                    )

    # ========================================================
    # OCR
    # ========================================================

    def _append_ocr(
        self,
        *,
        evidence: RawEvidence,
        target: list[EvidenceParameter],
    ) -> None:
        metadata = evidence.metadata

        self._add_metadata(
            target,
            metadata,
            group="OCR",
            mappings=(
                ("ocr_engine", "engine", None),
                ("language", "language", None),
                ("granularity", "granularity", None),
                ("confidence_raw", "confidence_raw", None),
            ),
        )

        self._add_metadata(
            target,
            metadata,
            group="TEXT_LAYOUT",
            mappings=(
                ("tesseract_index", "tesseract_index", None),
                ("block_num", "block_num", None),
                ("paragraph_num", "paragraph_num", None),
                ("line_num", "line_num", None),
                ("word_num", "word_num", None),
                ("token_ids", "token_ids", None),
                ("token_count", "token_count", "COUNT"),
            ),
        )

        raw_bbox = metadata.get("raw_bbox_px")
        if isinstance(raw_bbox, dict):
            for name in ("x", "y", "width", "height"):
                self._add(
                    target,
                    group="OCR",
                    name=f"raw_bbox_{name}_px",
                    value=raw_bbox.get(name),
                    unit="PX",
                    source_path=f"metadata.raw_bbox_px.{name}",
                )

    # ========================================================
    # GEMINI
    # ========================================================

    def _append_gemini(
        self,
        *,
        evidence: RawEvidence,
        target: list[EvidenceParameter],
    ) -> None:
        metadata = evidence.metadata

        self._add_metadata(
            target,
            metadata,
            group="SEMANTIC",
            mappings=(
                ("semantic_contract_version", "contract_version", None),
                ("semantic_source_mode", "source_mode", None),
                ("semantic_call_id", "call_id", None),
                ("semantic_category", "category", None),
                ("semantic_state", "state", None),
                ("semantic_subtype", "subtype", None),
                ("semantic_name", "name", None),
                ("location_text", "location", None),
                ("orientation", "orientation", None),
                ("span_type", "span_type", None),
            ),
        )

        self._add_metadata(
            target,
            metadata,
            group="REFERENCE",
            mappings=(
                ("reference_start", "reference_start", None),
                ("reference_end", "reference_end", None),
            ),
        )

        measurements = metadata.get("measurements")
        if isinstance(measurements, list):
            for index, measurement in enumerate(measurements):
                if not isinstance(measurement, dict):
                    continue
                self._add(
                    target,
                    group="MEASUREMENT",
                    name=str(measurement.get("name") or f"measurement_{index + 1}"),
                    value=measurement.get("value"),
                    unit=self._text_or_none(measurement.get("unit")),
                    state=self._text_or_none(measurement.get("state")),
                    confidence=self._confidence_or_none(
                        measurement.get("confidence")
                    ),
                    source_path=f"metadata.measurements[{index}]",
                )

        properties = metadata.get("properties")
        if isinstance(properties, list):
            for index, prop in enumerate(properties):
                if not isinstance(prop, dict):
                    continue
                value = (
                    prop.get("value_number")
                    if prop.get("value_number") is not None
                    else prop.get("value_text")
                )
                self._add(
                    target,
                    group="SEMANTIC_PROPERTY",
                    name=str(prop.get("name") or f"property_{index + 1}"),
                    value=value,
                    state=self._text_or_none(prop.get("state")),
                    confidence=self._confidence_or_none(prop.get("confidence")),
                    source_path=f"metadata.properties[{index}]",
                )

        relations = metadata.get("relations")
        if isinstance(relations, list):
            for index, relation in enumerate(relations):
                if not isinstance(relation, dict):
                    continue
                relation_type = str(
                    relation.get("type") or f"relation_{index + 1}"
                )
                self._add(
                    target,
                    group="RELATION",
                    name=relation_type,
                    value={
                        "target": relation.get("target"),
                        "target_category": relation.get("target_category"),
                    },
                    state=self._text_or_none(relation.get("state")),
                    confidence=self._confidence_or_none(
                        relation.get("confidence")
                    ),
                    source_path=f"metadata.relations[{index}]",
                )

        semantic_evidence = metadata.get("semantic_evidence")
        if isinstance(semantic_evidence, list):
            for index, text in enumerate(semantic_evidence):
                self._add(
                    target,
                    group="SEMANTIC_EVIDENCE",
                    name="evidence",
                    value=text,
                    source_path=f"metadata.semantic_evidence[{index}]",
                )

    # ========================================================
    # HELPERS
    # ========================================================

    def _add_metadata(
        self,
        target: list[EvidenceParameter],
        metadata: dict[str, Any],
        *,
        group: str,
        mappings: tuple[tuple[str, str, str | None], ...],
    ) -> None:
        for raw_name, parameter_name, unit in mappings:
            self._add(
                target,
                group=group,
                name=parameter_name,
                value=metadata.get(raw_name),
                unit=unit,
                source_path=f"metadata.{raw_name}",
            )

    @staticmethod
    def _add(
        target: list[EvidenceParameter],
        *,
        group: str,
        name: str,
        value: Any,
        unit: str | None = None,
        state: str | None = None,
        confidence: float | None = None,
        source_path: str | None = None,
    ) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        if isinstance(value, (list, dict, tuple, set)) and not value:
            return

        target.append(
            EvidenceParameter(
                group=group,
                name=name,
                value=value,
                unit=unit,
                state=state,
                confidence=confidence,
                source_path=source_path,
            )
        )

    @staticmethod
    def _deduplicate(
        parameters: list[EvidenceParameter],
    ) -> list[EvidenceParameter]:
        result: list[EvidenceParameter] = []
        seen: set[str] = set()

        for parameter in parameters:
            signature = parameter.model_dump_json(
                exclude_none=True,
                by_alias=False,
            )
            if signature in seen:
                continue
            seen.add(signature)
            result.append(parameter)

        return result

    @staticmethod
    def _text_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _confidence_or_none(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if 0.0 <= parsed <= 1.0 else None
