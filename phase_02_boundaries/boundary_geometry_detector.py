from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
)
from shapely.ops import polygonize_full, unary_union

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView

from .perimeter_models import (
    PerimeterCandidate,
    PerimeterEvidenceRef,
    PerimeterGeometrySource,
    PerimeterPointPx,
    PerimeterPolygonPx,
)


@dataclass(frozen=True, slots=True)
class _EvidenceLine:
    evidence: RawEvidence
    geometry: LineString


class BoundaryGeometryDetector:
    """
    Subproceso geométrico interno de Fase 02.

    Produce candidatos de contorno exterior a partir de evidencia geométrica
    real de F01.5. No decide todavía cuál candidato representa los muros
    perimetrales arquitectónicos.

    Consume:
        VECTOR_LINE
        RASTER_LINE
        RASTER_CONTOUR

    No:
        - usa el bbox del LevelView como perímetro;
        - genera convex hull;
        - hace snapping;
        - cierra líneas abiertas artificialmente;
        - selecciona por área;
        - convierte px -> m.
    """

    def detect(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> list[PerimeterCandidate]:
        items = list(evidence)
        self._validate_input(level_view=level_view, evidence=items)

        vector_lines = self._build_lines(
            level_view=level_view,
            evidence=[item for item in items if item.kind == "VECTOR_LINE"],
        )
        vector_candidates = self._detect_line_candidates(
            level_view=level_view,
            lines=vector_lines,
            geometry_source="VECTOR",
        )

        raster_lines = self._build_lines(
            level_view=level_view,
            evidence=self._select_raster_lines(items),
        )
        raster_line_candidates = self._detect_line_candidates(
            level_view=level_view,
            lines=raster_lines,
            geometry_source="RASTER",
        )

        # Reconstrucción híbrida sin snapping ni tolerancias inventadas.
        # Solo puede cerrar polígonos cuando segmentos VECTOR y RASTER
        # coinciden realmente en el mismo sistema px del LevelView.
        hybrid_line_candidates = self._detect_line_candidates(
            level_view=level_view,
            lines=[*vector_lines, *raster_lines],
            geometry_source="HYBRID",
        )
        hybrid_line_candidates = [
            candidate
            for candidate in hybrid_line_candidates
            if {ref.source for ref in candidate.evidence} >= {"PYMUPDF", "OPENCV"}
        ]

        raster_contour_candidates = self._detect_contour_candidates(
            level_view=level_view,
            evidence=[item for item in items if item.kind == "RASTER_CONTOUR"],
        )

        return self._merge_exact_candidates(
            level_view=level_view,
            candidates=[
                *vector_candidates,
                *raster_line_candidates,
                *hybrid_line_candidates,
                *raster_contour_candidates,
            ],
        )

    @staticmethod
    def _select_raster_lines(
        evidence: Sequence[RawEvidence],
    ) -> list[RawEvidence]:
        raster = [item for item in evidence if item.kind == "RASTER_LINE"]
        if not raster:
            return []

        merged = [
            item
            for item in raster
            if str(BoundaryGeometryDetector._parameter_or_metadata(
                item, group="RASTER_GEOMETRY", name="stage", metadata_name="stage"
            ) or "") == "continuity_merged"
        ]
        if not merged:
            return raster

        raw_other = [
            item
            for item in raster
            if str(BoundaryGeometryDetector._parameter_or_metadata(
                item, group="RASTER_GEOMETRY", name="stage", metadata_name="stage"
            ) or "") == "raw_hough"
            and str(BoundaryGeometryDetector._parameter_or_metadata(
                item, group="RASTER_GEOMETRY", name="orientation", metadata_name="orientation"
            ) or "") == "other"
        ]
        return [*merged, *raw_other]

    def _build_lines(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> list[_EvidenceLine]:
        bounds = box(
            0.0,
            0.0,
            float(level_view.raster_width_px),
            float(level_view.raster_height_px),
        )
        result: list[_EvidenceLine] = []

        for item in evidence:
            line = self._segment_from_parameters(item)
            if line is None:
                geometry = item.geometry
                if geometry.geometry_type != "SEGMENT" or len(geometry.points) != 2:
                    continue
                p0, p1 = geometry.points
                line = LineString(
                    [(float(p0.x), float(p0.y)), (float(p1.x), float(p1.y))]
                )
            if line.is_empty or line.length <= 0.0:
                continue

            try:
                clipped = line.intersection(bounds)
            except Exception:
                continue

            for part in self._extract_line_strings(clipped):
                if part.is_empty or part.length <= 0.0:
                    continue
                result.append(_EvidenceLine(evidence=item, geometry=part))

        return result

    @staticmethod
    def _segment_from_parameters(evidence: RawEvidence) -> LineString | None:
        for parameter in getattr(evidence, "parameters", []):
            if str(parameter.group).upper() != "GEOMETRY" or str(parameter.name) != "points_px":
                continue
            value = parameter.value
            if not isinstance(value, list) or len(value) != 2:
                continue
            try:
                coordinates = [
                    (float(point["x"]), float(point["y"]))
                    for point in value
                    if isinstance(point, dict) and "x" in point and "y" in point
                ]
            except (TypeError, ValueError, KeyError):
                continue
            if len(coordinates) == 2 and coordinates[0] != coordinates[1]:
                return LineString(coordinates)
        return None

    @staticmethod
    def _parameter_or_metadata(
        evidence: RawEvidence,
        *,
        group: str,
        name: str,
        metadata_name: str,
    ) -> object | None:
        for parameter in getattr(evidence, "parameters", []):
            if str(parameter.group).upper() == group and str(parameter.name) == name:
                return parameter.value
        return evidence.metadata.get(metadata_name)

    def _detect_line_candidates(
        self,
        *,
        level_view: LevelView,
        lines: Sequence[_EvidenceLine],
        geometry_source: PerimeterGeometrySource,
    ) -> list[PerimeterCandidate]:
        if not lines:
            return []

        faces = self._polygonize_faces(lines)
        if not faces:
            return []

        components = self._build_components(faces)
        if not components:
            return []

        return self._build_candidates_from_polygons(
            level_view=level_view,
            polygons=components,
            lines=lines,
            geometry_source=geometry_source,
        )

    def _detect_contour_candidates(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> list[PerimeterCandidate]:
        result: list[PerimeterCandidate] = []

        for item in evidence:
            geometry = item.geometry
            if geometry.geometry_type != "POLYLINE" or len(geometry.points) < 3:
                continue
            if item.metadata.get("closed") is not True:
                continue

            coordinates = [(float(point.x), float(point.y)) for point in geometry.points]
            if coordinates[0] == coordinates[-1]:
                coordinates = coordinates[:-1]
            if len(set(coordinates)) < 3:
                continue

            polygon = Polygon(coordinates)
            if polygon.is_empty or polygon.area <= 0.0 or not polygon.is_valid:
                continue

            candidate = self._candidate_from_polygon(
                level_view=level_view,
                polygon=polygon,
                geometry_source="RASTER",
                support=[self._evidence_ref(item)],
            )
            if candidate is not None:
                result.append(candidate)

        return self._deduplicate_candidates(result)

    @staticmethod
    def _polygonize_faces(lines: Sequence[_EvidenceLine]) -> list[Polygon]:
        geometries = [
            item.geometry
            for item in lines
            if not item.geometry.is_empty and item.geometry.length > 0.0
        ]
        if not geometries:
            return []

        try:
            merged = unary_union(geometries)
            polygons_geometry, _cuts, _dangles, _invalid = polygonize_full(merged)
        except Exception:
            return []

        return [
            geometry
            for geometry in BoundaryGeometryDetector._geometry_items(polygons_geometry)
            if isinstance(geometry, Polygon)
            and not geometry.is_empty
            and geometry.is_valid
            and geometry.area > 0.0
        ]

    @staticmethod
    def _build_components(faces: Sequence[Polygon]) -> list[Polygon]:
        if not faces:
            return []
        try:
            dissolved = unary_union(list(faces))
        except Exception:
            return []

        return [
            polygon
            for polygon in BoundaryGeometryDetector._extract_polygons(dissolved)
            if not polygon.is_empty and polygon.is_valid and polygon.area > 0.0
        ]

    def _build_candidates_from_polygons(
        self,
        *,
        level_view: LevelView,
        polygons: Sequence[Polygon],
        lines: Sequence[_EvidenceLine],
        geometry_source: PerimeterGeometrySource,
    ) -> list[PerimeterCandidate]:
        result: list[PerimeterCandidate] = []

        for polygon in polygons:
            support = self._source_evidence_for_polygon(polygon=polygon, lines=lines)
            if not support:
                continue
            candidate = self._candidate_from_polygon(
                level_view=level_view,
                polygon=polygon,
                geometry_source=geometry_source,
                support=support,
            )
            if candidate is not None:
                result.append(candidate)

        return self._deduplicate_candidates(result)

    def _candidate_from_polygon(
        self,
        *,
        level_view: LevelView,
        polygon: Polygon,
        geometry_source: PerimeterGeometrySource,
        support: Sequence[PerimeterEvidenceRef],
    ) -> PerimeterCandidate | None:
        coordinates = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        if len(coordinates) < 3:
            return None

        try:
            geometry = PerimeterPolygonPx(
                points=[PerimeterPointPx(x=x, y=y) for x, y in coordinates]
            )
        except ValueError:
            return None

        signature = self._polygon_signature(polygon)
        return PerimeterCandidate(
            id=self._candidate_id(
                level_view_id=level_view.id,
                geometry_source=geometry_source,
                signature=signature,
            ),
            level_view_id=level_view.id,
            geometry=geometry,
            geometry_source=geometry_source,
            evidence=[item.model_copy(deep=True) for item in support],
            semantic_evidence_ids=[],
            semantic_localized_evidence_ids=[],
            interior_ring_count=len(polygon.interiors),
            confirmed=False,
        )

    @staticmethod
    def _source_evidence_for_polygon(
        *,
        polygon: Polygon,
        lines: Sequence[_EvidenceLine],
    ) -> list[PerimeterEvidenceRef]:
        boundary = polygon.boundary
        supported: dict[str, PerimeterEvidenceRef] = {}

        for item in lines:
            try:
                shared_length = float(boundary.intersection(item.geometry).length)
            except Exception:
                continue
            if shared_length <= 0.0:
                continue
            supported[item.evidence.id] = BoundaryGeometryDetector._evidence_ref(
                item.evidence
            )

        return [supported[key] for key in sorted(supported)]

    @staticmethod
    def _evidence_ref(evidence: RawEvidence) -> PerimeterEvidenceRef:
        description = str(evidence.metadata.get("description", "") or "").strip() or None
        semantic_category = str(
            evidence.metadata.get("semantic_category", "") or ""
        ).strip() or None
        return PerimeterEvidenceRef(
            evidence_id=evidence.id,
            source=str(evidence.source),
            kind=str(evidence.kind),
            semantic_category=semantic_category,
            text=evidence.text,
            description=description,
            confidence=evidence.confidence,
        )

    def _merge_exact_candidates(
        self,
        *,
        level_view: LevelView,
        candidates: Sequence[PerimeterCandidate],
    ) -> list[PerimeterCandidate]:
        """
        Fusiona candidatos que representan exactamente la misma geometría
        topológica, aunque una fuente haya partido un lado en más vértices.

        No usa buffer, snapping ni distancia/tolerancia. Shapely.equals() debe
        confirmar igualdad geométrica real.
        """
        groups: list[list[PerimeterCandidate]] = []

        for candidate in candidates:
            polygon = Polygon(
                [(point.x, point.y) for point in candidate.geometry.points]
            )
            matched_group: list[PerimeterCandidate] | None = None

            for group in groups:
                reference = group[0]
                reference_polygon = Polygon(
                    [(point.x, point.y) for point in reference.geometry.points]
                )
                try:
                    equal = polygon.equals(reference_polygon)
                except Exception:
                    equal = False
                if equal:
                    matched_group = group
                    break

            if matched_group is None:
                groups.append([candidate])
            else:
                matched_group.append(candidate)

        result: list[PerimeterCandidate] = []
        for group in groups:
            polygon = Polygon(
                [(point.x, point.y) for point in group[0].geometry.points]
            )
            signature = self._polygon_signature(polygon)
            sources = {item.geometry_source for item in group}
            evidence_sources = {
                ref.source
                for item in group
                for ref in item.evidence
            }

            geometry_source: PerimeterGeometrySource = (
                "HYBRID"
                if (
                    "HYBRID" in sources
                    or {"PYMUPDF", "OPENCV"}.issubset(evidence_sources)
                    or ("VECTOR" in sources and "RASTER" in sources)
                )
                else group[0].geometry_source
            )

            evidence_by_id: dict[str, PerimeterEvidenceRef] = {}
            semantic_ids: set[str] = set()
            semantic_localized_ids: set[str] = set()
            interior_ring_count = 0

            for item in group:
                for ref in item.evidence:
                    evidence_by_id[ref.evidence_id] = ref.model_copy(deep=True)
                semantic_ids.update(item.semantic_evidence_ids)
                semantic_localized_ids.update(item.semantic_localized_evidence_ids)
                interior_ring_count = max(interior_ring_count, item.interior_ring_count)

            result.append(
                PerimeterCandidate(
                    id=self._candidate_id(
                        level_view_id=level_view.id,
                        geometry_source=geometry_source,
                        signature=signature,
                    ),
                    level_view_id=level_view.id,
                    geometry=group[0].geometry.model_copy(deep=True),
                    geometry_source=geometry_source,
                    evidence=[evidence_by_id[key] for key in sorted(evidence_by_id)],
                    semantic_evidence_ids=sorted(semantic_ids),
                    semantic_localized_evidence_ids=sorted(semantic_localized_ids),
                    interior_ring_count=interior_ring_count,
                    confirmed=False,
                )
            )

        return sorted(result, key=lambda item: item.id)

    def _deduplicate_candidates(
        self,
        candidates: Sequence[PerimeterCandidate],
    ) -> list[PerimeterCandidate]:
        unique: dict[tuple[tuple[float, float], ...], PerimeterCandidate] = {}
        for candidate in candidates:
            polygon = Polygon(
                [(point.x, point.y) for point in candidate.geometry.points]
            )
            signature = self._polygon_signature(polygon)
            if signature and signature not in unique:
                unique[signature] = candidate
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _extract_line_strings(geometry: object) -> list[LineString]:
        if isinstance(geometry, LineString):
            return [geometry]
        if isinstance(geometry, MultiLineString):
            return [item for item in geometry.geoms if isinstance(item, LineString)]
        if isinstance(geometry, GeometryCollection):
            result: list[LineString] = []
            for item in geometry.geoms:
                result.extend(BoundaryGeometryDetector._extract_line_strings(item))
            return result
        return []

    @staticmethod
    def _extract_polygons(geometry: object) -> list[Polygon]:
        if isinstance(geometry, Polygon):
            return [geometry]
        if isinstance(geometry, MultiPolygon):
            return [item for item in geometry.geoms if isinstance(item, Polygon)]
        if isinstance(geometry, GeometryCollection):
            result: list[Polygon] = []
            for item in geometry.geoms:
                result.extend(BoundaryGeometryDetector._extract_polygons(item))
            return result
        return []

    @staticmethod
    def _geometry_items(geometry: object) -> list[object]:
        if geometry is None:
            return []
        if hasattr(geometry, "geoms"):
            try:
                return list(geometry.geoms)
            except Exception:
                return []
        return [geometry]

    @staticmethod
    def _polygon_signature(
        polygon: Polygon,
    ) -> tuple[tuple[float, float], ...]:
        coordinates = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        if not coordinates:
            return ()

        rotations = [
            tuple(coordinates[index:] + coordinates[:index])
            for index in range(len(coordinates))
        ]
        reversed_coordinates = list(reversed(coordinates))
        rotations.extend(
            tuple(reversed_coordinates[index:] + reversed_coordinates[:index])
            for index in range(len(reversed_coordinates))
        )
        return min(rotations)

    @staticmethod
    def _candidate_id(
        *,
        level_view_id: str,
        geometry_source: PerimeterGeometrySource,
        signature: tuple[tuple[float, float], ...],
    ) -> str:
        payload = f"{level_view_id}|{geometry_source}|{repr(signature)}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return f"{level_view_id}__PERIMETER_CANDIDATE__{geometry_source}__{digest}"

    @staticmethod
    def _validate_input(
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
