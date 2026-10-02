from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from shapely.geometry import LineString, Polygon, box

from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView

from .perimeter_models import (
    PerimeterBBoxPx,
    PerimeterCandidate,
    PerimeterCandidateEvidenceReconciliation,
    PerimeterEvidenceReconciliationResult,
    PerimeterMotorSegmentSupport,
    PerimeterPointPx,
    PerimeterSourceEvidenceSummary,
    PerimeterTextEvidenceObservation,
)


class PerimeterEvidenceReconciler:
    """
    Fase 02 — reconciliación multifuente previa a resolver el perímetro.

    Usa únicamente la reconstrucción propia del motor:
        PyMuPDF + OpenCV + OCR.

    Gemini se excluye explícitamente y se mantiene como baseline independiente
    para `PerimeterSemanticComparator`.

    Principios:
        - consume primero `RawEvidence.parameters`;
        - si un parámetro necesario no existe, consulta geometry/text/metadata;
        - no hace snapping ni usa tolerancias arquitectónicas inventadas;
        - no completa huecos con Gemini;
        - evidencia incompleta se conserva como PARTIAL/UNRESOLVED;
        - cada asociación conserva ids y fuente.
    """

    _TEXT_KINDS = frozenset({"OCR_TEXT", "VECTOR_TEXT"})
    _GEOMETRY_KINDS = frozenset({"VECTOR_LINE", "RASTER_LINE", "RASTER_CONTOUR"})
    _NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])([0-9]+(?:[.,][0-9]+)?)(?![A-Za-z0-9])")
    _EXPLICIT_UNIT_RE = re.compile(
        r"\b(?:M|MT|MTS|METRO|METROS|CM|MM)\b", re.IGNORECASE
    )
    _AXIS_TOKEN_RE = re.compile(r"^[A-Za-z]{1,2}$|^[0-9]{1,3}$")

    def reconcile(
        self,
        *,
        level_view: LevelView,
        candidates: Sequence[PerimeterCandidate],
        evidence: Sequence[RawEvidence],
    ) -> PerimeterEvidenceReconciliationResult:
        raw = list(evidence)
        self._validate_input(level_view=level_view, candidates=candidates, evidence=raw)

        motor_evidence = [item for item in raw if item.source != "GEMINI"]
        gemini_count = sum(1 for item in raw if item.source == "GEMINI")
        text_observations = self._text_observations(motor_evidence)

        reconciled = [
            self._reconcile_candidate(
                candidate=candidate,
                evidence=motor_evidence,
                text_observations=text_observations,
            )
            for candidate in candidates
        ]

        notes = [
            "La reconciliación F02 usa PyMuPDF/OpenCV/OCR; Gemini queda fuera del motor.",
            "Las asociaciones geométricas son exactas sobre las coordenadas observadas; no se aplica snapping ni tolerancia inventada.",
        ]
        if not motor_evidence:
            notes.append("No existe evidencia no-Gemini para reconstruir el perímetro.")

        return PerimeterEvidenceReconciliationResult(
            level_view_id=level_view.id,
            non_gemini_evidence_count=len(motor_evidence),
            gemini_evidence_count=gemini_count,
            candidates=reconciled,
            unassigned_text_evidence=text_observations,
            notes=notes,
        )

    def _reconcile_candidate(
        self,
        *,
        candidate: PerimeterCandidate,
        evidence: Sequence[RawEvidence],
        text_observations: Sequence[PerimeterTextEvidenceObservation],
    ) -> PerimeterCandidateEvidenceReconciliation:
        points = [(point.x, point.y) for point in candidate.geometry.points]
        segments = [
            (points[index], points[(index + 1) % len(points)])
            for index in range(len(points))
            if points[index] != points[(index + 1) % len(points)]
        ]

        # No limitar la reconciliación a la evidencia que creó el candidato.
        # Un candidato RASTER puede ser corroborado por una VECTOR_LINE de
        # PyMuPDF que coincide exactamente con uno de sus tramos, aunque esa
        # línea no haya participado en el polygonize que originó el candidato.
        candidate_geometry_evidence = [
            item for item in evidence if item.kind in self._GEOMETRY_KINDS
        ]

        segment_support: list[PerimeterMotorSegmentSupport] = []
        all_support_ids: set[str] = set()
        parameterized_ids: set[str] = set()
        raw_fallback_ids: set[str] = set()
        sources: set[str] = set()

        for index, (start, end) in enumerate(segments):
            target = LineString([start, end])
            support_ids: set[str] = set()
            support_sources: set[str] = set()
            param_ids: set[str] = set()
            fallback_ids: set[str] = set()

            for item in candidate_geometry_evidence:
                geometry, used_parameters = self._line_geometry(item)
                if geometry is None:
                    continue
                shared = self._support_overlap_length(
                    target=target,
                    geometry=geometry,
                    evidence=item,
                )
                if shared <= 0.0:
                    continue

                support_ids.add(item.id)
                support_sources.add(str(item.source))
                if used_parameters:
                    param_ids.add(item.id)
                else:
                    fallback_ids.add(item.id)

            all_support_ids.update(support_ids)
            parameterized_ids.update(param_ids)
            raw_fallback_ids.update(fallback_ids)
            sources.update(support_sources)

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            orientation = (
                "HORIZONTAL" if dy == 0.0 else "VERTICAL" if dx == 0.0 else "DIAGONAL"
            )
            segment_support.append(
                PerimeterMotorSegmentSupport(
                    sequence_index=index,
                    start_px=PerimeterPointPx(x=start[0], y=start[1]),
                    end_px=PerimeterPointPx(x=end[0], y=end[1]),
                    orientation=orientation,
                    length_px=float(target.length),
                    evidence_ids=sorted(support_ids),
                    sources=sorted(support_sources),
                    parameterized_evidence_ids=sorted(param_ids),
                    raw_fallback_evidence_ids=sorted(fallback_ids),
                )
            )

        supported_count = sum(1 for item in segment_support if item.evidence_ids)
        cross_source_count = sum(
            1 for item in segment_support if len(set(item.sources)) >= 2
        )
        independent_sources = sorted(
            {source for item in segment_support for source in item.sources}
        )

        # SUPPORTED exige una cadena completa y corroboración real entre
        # fuentes independientes. La misma evidencia que generó el candidato
        # no puede convertirlo por sí sola en candidato corroborado.
        if (
            segment_support
            and supported_count == len(segment_support)
            and len(independent_sources) >= 2
            and cross_source_count > 0
        ):
            state = "SUPPORTED"
        elif supported_count > 0:
            state = "PARTIAL"
        else:
            state = "UNRESOLVED"

        source_summaries: list[PerimeterSourceEvidenceSummary] = []
        for source in ("PYMUPDF", "OPENCV", "OCR"):
            source_items = [item for item in evidence if item.source == source]
            relevant_ids = {
                evidence_id
                for segment in segment_support
                for evidence_id in segment.evidence_ids
                if any(item.id == evidence_id for item in source_items)
            }
            source_summaries.append(
                PerimeterSourceEvidenceSummary(
                    source=source,
                    evidence_count=len(source_items),
                    perimeter_support_count=len(relevant_ids),
                    parameterized_support_count=sum(
                        1 for evidence_id in relevant_ids if evidence_id in parameterized_ids
                    ),
                    raw_fallback_support_count=sum(
                        1 for evidence_id in relevant_ids if evidence_id in raw_fallback_ids
                    ),
                    kind_counts=dict(
                        sorted(Counter(str(item.kind) for item in source_items).items())
                    ),
                )
            )

        candidate_bbox = box(
            candidate.geometry.bbox.x_min,
            candidate.geometry.bbox.y_min,
            candidate.geometry.bbox.x_max,
            candidate.geometry.bbox.y_max,
        )
        localized_text = []
        for observation in text_observations:
            if observation.bbox_px is None:
                continue
            b = observation.bbox_px
            try:
                if candidate_bbox.intersects(box(b.x_min, b.y_min, b.x_max, b.y_max)):
                    localized_text.append(observation)
            except Exception:
                continue

        notes: list[str] = []
        if supported_count < len(segment_support):
            notes.append(
                "La cadena perimetral tiene tramos sin soporte geométrico verificable; se conserva para fine tuning sin completar huecos."
            )
        if len(independent_sources) < 2:
            notes.append(
                "El candidato solo tiene autosoporte de una fuente geométrica; no se considera corroborado."
            )
        elif cross_source_count == 0:
            notes.append(
                "Hay evidencia de varias fuentes en el candidato, pero ningún tramo tiene corroboración espacial verificable entre fuentes."
            )
        else:
            notes.append(
                f"Tramos con corroboración verificable entre fuentes: {cross_source_count}/{len(segment_support)}."
            )

        return PerimeterCandidateEvidenceReconciliation(
            candidate_id=candidate.id,
            state=state,
            independent_geometry_sources=independent_sources,
            segment_support=segment_support,
            source_summaries=source_summaries,
            localized_text_evidence=localized_text,
            supporting_evidence_ids=sorted(all_support_ids),
            parameterized_evidence_ids=sorted(parameterized_ids),
            raw_fallback_evidence_ids=sorted(raw_fallback_ids),
            notes=notes,
        )

    def _support_overlap_length(
        self,
        *,
        target: LineString,
        geometry: LineString,
        evidence: RawEvidence,
    ) -> float:
        """
        Devuelve longitud de soporte del tramo sin introducir tolerancias fijas.

        1. Primero exige intersección geométrica real.
        2. Para VECTOR_LINE permite corroboración contra una cara raster
           desplazada respecto al eje vectorial únicamente dentro de la mitad
           del grosor de trazo observado y transformado a px.

        Ese margen no es arquitectónico ni inventado: procede del propio
        `stroke_width` del PDF y de la transformación page->raster conservada
        por PyMuPDF.
        """
        try:
            exact = float(target.intersection(geometry).length)
        except Exception:
            exact = 0.0
        if exact > 0.0:
            return exact

        if evidence.kind != "VECTOR_LINE" or str(evidence.source) != "PYMUPDF":
            return 0.0

        target_orientation = self._axis_orientation(target)
        geometry_orientation = self._axis_orientation(geometry)
        if (
            target_orientation not in {"HORIZONTAL", "VERTICAL"}
            or target_orientation != geometry_orientation
        ):
            return 0.0

        half_width_px = self._vector_half_stroke_width_px(
            evidence=evidence,
            orientation=geometry_orientation,
        )
        if half_width_px is None or half_width_px <= 0.0:
            return 0.0

        t0 = target.coords[0]
        t1 = target.coords[-1]
        g0 = geometry.coords[0]
        g1 = geometry.coords[-1]

        if target_orientation == "HORIZONTAL":
            target_axis = float(t0[1])
            geometry_axis = float(g0[1])
            if abs(target_axis - geometry_axis) > half_width_px:
                return 0.0
            return self._interval_overlap(
                float(t0[0]), float(t1[0]), float(g0[0]), float(g1[0])
            )

        target_axis = float(t0[0])
        geometry_axis = float(g0[0])
        if abs(target_axis - geometry_axis) > half_width_px:
            return 0.0
        return self._interval_overlap(
            float(t0[1]), float(t1[1]), float(g0[1]), float(g1[1])
        )

    def _vector_half_stroke_width_px(
        self,
        *,
        evidence: RawEvidence,
        orientation: str,
    ) -> float | None:
        stroke_width = self._parameter_value(
            evidence, group="VECTOR_STYLE", name="stroke_width"
        )
        if stroke_width is None:
            drawing = evidence.metadata.get("drawing")
            if isinstance(drawing, dict):
                stroke_width = drawing.get("stroke_width")

        scale_name = (
            "page_to_raster_scale_y"
            if orientation == "HORIZONTAL"
            else "page_to_raster_scale_x"
        )
        scale = self._parameter_value(
            evidence, group="TRANSFORM", name=scale_name
        )
        if scale is None:
            scale = evidence.metadata.get(scale_name)

        try:
            width = float(stroke_width)
            raster_scale = float(scale)
        except (TypeError, ValueError):
            return None
        if width <= 0.0 or raster_scale <= 0.0:
            return None
        return (width * raster_scale) / 2.0

    @staticmethod
    def _axis_orientation(line: LineString) -> str:
        if len(line.coords) < 2:
            return "DIAGONAL"
        p0 = line.coords[0]
        p1 = line.coords[-1]
        if float(p0[1]) == float(p1[1]):
            return "HORIZONTAL"
        if float(p0[0]) == float(p1[0]):
            return "VERTICAL"
        return "DIAGONAL"

    @staticmethod
    def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        a_min, a_max = sorted((a0, a1))
        b_min, b_max = sorted((b0, b1))
        return max(0.0, min(a_max, b_max) - max(a_min, b_min))

    def _line_geometry(self, evidence: RawEvidence) -> tuple[LineString | None, bool]:
        parameter_points = self._parameter_value(
            evidence,
            group="GEOMETRY",
            name="points_px",
        )
        if isinstance(parameter_points, list) and len(parameter_points) >= 2:
            try:
                coordinates = [
                    (float(point["x"]), float(point["y"]))
                    for point in parameter_points
                    if isinstance(point, dict) and "x" in point and "y" in point
                ]
                if len(coordinates) >= 2:
                    return LineString(coordinates), True
            except (TypeError, ValueError, KeyError):
                pass

        geometry = evidence.geometry
        if geometry.geometry_type in {"SEGMENT", "POLYLINE"} and len(geometry.points) >= 2:
            try:
                return LineString(
                    [(float(point.x), float(point.y)) for point in geometry.points]
                ), False
            except Exception:
                return None, False
        return None, False

    def _text_observations(
        self,
        evidence: Sequence[RawEvidence],
    ) -> list[PerimeterTextEvidenceObservation]:
        result: list[PerimeterTextEvidenceObservation] = []
        for item in evidence:
            if item.kind not in self._TEXT_KINDS:
                continue

            text = self._parameter_value(item, group="TEXT", name="visible_text")
            used_parameters = isinstance(text, str) and bool(text.strip())
            if not used_parameters:
                text = item.text
            normalized = str(text or "").strip()
            if not normalized:
                continue

            values = []
            for token in self._NUMBER_RE.findall(normalized):
                try:
                    values.append(float(token.replace(",", ".")))
                except ValueError:
                    continue

            explicit_unit = None
            unit_match = self._EXPLICIT_UNIT_RE.search(normalized)
            if unit_match:
                explicit_unit = unit_match.group(0).upper()

            compact = normalized.strip()
            if explicit_unit and values:
                role = "DIMENSION_TEXT_CANDIDATE"
            elif self._AXIS_TOKEN_RE.fullmatch(compact):
                role = "AXIS_TOKEN_CANDIDATE"
            elif values:
                role = "NUMERIC_TOKEN_CANDIDATE"
            else:
                role = "OTHER_TEXT"

            result.append(
                PerimeterTextEvidenceObservation(
                    evidence_id=item.id,
                    source=str(item.source),
                    text=normalized,
                    role=role,
                    parsed_values=values,
                    explicit_unit=explicit_unit,
                    bbox_px=(
                        PerimeterBBoxPx(
                            x_min=item.geometry.bbox_px.x_min,
                            y_min=item.geometry.bbox_px.y_min,
                            x_max=item.geometry.bbox_px.x_max,
                            y_max=item.geometry.bbox_px.y_max,
                        )
                        if item.geometry.bbox_px is not None
                        else None
                    ),
                    parameterized=used_parameters,
                )
            )
        return result

    @staticmethod
    def _parameter_value(
        evidence: RawEvidence,
        *,
        group: str,
        name: str,
    ) -> object | None:
        for parameter in getattr(evidence, "parameters", []):
            if str(parameter.group).upper() == group and str(parameter.name) == name:
                return parameter.value
        return None

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
        seen: set[str] = set()
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(f"RawEvidence {item.id} pertenece a otro LevelView.")
            if item.id in seen:
                raise ValueError(f"RawEvidence.id duplicado: {item.id}.")
            seen.add(item.id)
            if item.confirmed:
                raise ValueError(f"RawEvidence {item.id} llegó confirmed=True.")
