from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry import LineString

from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel

from .candidate_models import WallCandidate, WallEvidenceVector
from .drawing_model import DrawingLine, DrawingModel, DrawingPoint
from .evidence_fusion import WallEvidenceFusion
from .perimeter_adapter import constrain_line_to_perimeter, perimeter_polygon
from .reference_constraints import ReferenceConstraintResult
from .wall_track_consolidator import WallTrackConsolidator


@dataclass(frozen=True)
class _ProjectedLine:
    item: DrawingLine
    ux: float
    uy: float
    nx: float
    ny: float
    t0: float
    t1: float
    offset: float


class WallCandidateGenerator:
    """Genera hipótesis; no decide muros definitivos."""

    ANGLE_TOLERANCE_DEG = 3.0

    def __init__(self) -> None:
        self.fusion = WallEvidenceFusion()
        self.track_consolidator = WallTrackConsolidator()

    def generate(
        self,
        *,
        drawing: DrawingModel,
        perimeter: EditablePerimeterModel,
        constraints: ReferenceConstraintResult | None = None,
    ) -> list[WallCandidate]:
        """Proposal C A1.2 Candidate Discovery, sin clasificación semántica.

        V4 conserva las reglas geométricas de generación y mueve cualquier
        aislamiento contextual fuera de este archivo. De este modo Candidate
        Discovery no puede excluir una hipótesis porque "parezca" escalera,
        retícula u otro símbolo.

        Importante: el límite operativo de candidatos ya no se aplica aquí.
        Context Gate debe ejecutarse primero para impedir que ruido repetitivo
        ocupe el cupo del CandidateGraph y desplace hipótesis válidas.
        """
        polygon = perimeter_polygon(perimeter)
        mask = self._mask(drawing)
        face_lines = [item for item in drawing.lines if item.kind == "LINE"]
        region_lines = [item for item in drawing.lines if item.kind == "REGION_CENTERLINE"]
        representatives = self._representative_lines(face_lines, drawing=drawing)
        profiles = self._thickness_profiles(representatives, drawing=drawing)

        candidates = self._double_face_candidates(
            drawing=drawing,
            lines=representatives,
            thickness_profiles=profiles,
            perimeter_polygon_geom=polygon,
            wall_region_mask=mask,
            constraints=constraints,
        )
        candidates.extend(
            self._region_candidates(
                drawing=drawing,
                region_lines=region_lines,
                face_lines=representatives,
                thickness_profiles=profiles,
                perimeter_polygon_geom=polygon,
                wall_region_mask=mask,
                constraints=constraints,
            )
        )
        candidates = self._dedupe_candidates(candidates)
        candidates = self.track_consolidator.consolidate(
            candidates=candidates,
            width_px=drawing.width_px,
            height_px=drawing.height_px,
        )
        return self._constrain_candidates_to_perimeter(
            candidates=candidates,
            polygon=polygon,
        )

    def _constrain_candidates_to_perimeter(
        self,
        *,
        candidates: Sequence[WallCandidate],
        polygon,
    ) -> list[WallCandidate]:
        constrained: list[WallCandidate] = []
        for candidate in candidates:
            original = LineString([
                (candidate.start.x, candidate.start.y),
                (candidate.end.x, candidate.end.y),
            ])
            max_rebase = max(2.0, 1.50 * candidate.thickness_px)
            clipped = constrain_line_to_perimeter(
                line=original,
                polygon=polygon,
                max_rebase_px=max_rebase,
                clip_tolerance_px=1.0,
            )
            if clipped is None or clipped.length <= 1e-6:
                continue

            coords = list(clipped.coords)
            x1, y1 = coords[0]
            x2, y2 = coords[-1]
            length = float(clipped.length)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            retained_ratio = max(0.0, min(1.0, length / max(original.length, 1e-6)))

            metadata = dict(candidate.metadata)
            if retained_ratio < 0.999999:
                metadata.update({
                    "f02_rebased": True,
                    "f02_original_start": [candidate.start.x, candidate.start.y],
                    "f02_original_end": [candidate.end.x, candidate.end.y],
                    "f02_retained_ratio": retained_ratio,
                })

            constrained.append(
                candidate.model_copy(
                    update={
                        "start": DrawingPoint(x=float(x1), y=float(y1)),
                        "end": DrawingPoint(x=float(x2), y=float(y2)),
                        "angle_deg": float(angle),
                        "length_px": length,
                        "prior_score": max(0.0, min(1.0, candidate.prior_score * retained_ratio)),
                        "metadata": metadata,
                    },
                    deep=True,
                )
            )
        return constrained

    def _representative_lines(
        self,
        lines: Sequence[DrawingLine],
        *,
        drawing: DrawingModel,
    ) -> list[DrawingLine]:
        diag = math.hypot(drawing.width_px, drawing.height_px)
        distance_tol = max(1.5, diag * 0.0011)
        minimum_length = max(8.0, min(drawing.width_px, drawing.height_px) * 0.008)
        groups: list[list[DrawingLine]] = []
        for line in sorted(lines, key=lambda item: -item.length_px):
            if line.length_px < minimum_length:
                continue
            target = None
            for group in groups:
                ref = group[0]
                if self._angle_diff(line.angle_deg, ref.angle_deg) > self.ANGLE_TOLERANCE_DEG:
                    continue
                if self._line_distance(line, ref) > distance_tol:
                    continue
                if self._overlap_ratio(line, ref) < 0.55:
                    continue
                target = group
                break
            if target is None:
                groups.append([line])
            else:
                target.append(line)

        priority = {"PYMUPDF": 5, "OPENCV": 4, "LSD_LOCAL": 3, "OCR": 1, "GEMINI": 1}
        output: list[DrawingLine] = []
        for group in groups:
            group.sort(
                key=lambda item: (
                    max([priority.get(source, 0) for source in item.sources] or [0]),
                    item.length_px,
                ),
                reverse=True,
            )
            ref = group[0].model_copy(deep=True)
            ref.sources = sorted({source for item in group for source in item.sources})
            ref.evidence_ids = sorted({eid for item in group for eid in item.evidence_ids})
            ref.metadata["consensus_sources"] = list(ref.sources)
            ref.metadata["merged_line_ids"] = [item.id for item in group]
            # Element Context Isolation V4: preserva equivalencias de provenance
            # entre la línea representante usada por Candidate Discovery y las
            # primitivas originales que después forman ContextRegion.member_line_ids.
            context_ids: set[str] = set()
            for item in group:
                context_ids.update(self._line_lineage_ids(item))
            ref.metadata["context_equivalent_line_ids"] = sorted(context_ids)
            ref.dashed = all(item.dashed for item in group)
            output.append(ref)
        return output

    def _thickness_profiles(
        self,
        lines: Sequence[DrawingLine],
        *,
        drawing: DrawingModel,
    ) -> list[tuple[float, float]]:
        """Estima familias de espesor sin permitir votación por repetición.

        La implementación anterior contaba cada par paralelo como un voto. En una
        escalera, baldosa o hatch, decenas de segmentos cortos podían dominar la
        familia de espesor aunque todos provinieran de la misma modalidad raster.

        Selection Contracts V1 cambia la unidad de soporte: cada observación aporta
        longitud realmente solapada y se pondera por modalidades geométricas
        independientes (VECTOR / RASTER). OPENCV, LSD_LOCAL y REGION_LOCAL son la
        misma modalidad de observación y no votan por separado.
        """
        min_dim = min(drawing.width_px, drawing.height_px)
        min_t = max(1.5, min_dim * 0.0012)
        max_t = max(10.0, min_dim * 0.035)
        observations: list[tuple[float, float]] = []
        for index, first in enumerate(lines):
            for second in lines[index + 1 :]:
                if self._angle_diff(first.angle_deg, second.angle_deg) > self.ANGLE_TOLERANCE_DEG:
                    continue
                overlap_ratio = self._overlap_ratio(first, second)
                if overlap_ratio < 0.45:
                    continue
                distance = self._line_distance(first, second)
                if not (min_t <= distance <= max_t):
                    continue

                overlap_length = min(first.length_px, second.length_px) * overlap_ratio
                source_groups = self._independent_source_groups([*first.sources, *second.sources])
                modality_weight = 1.0 if len(source_groups) >= 2 else 0.62
                support_weight = max(1.0, overlap_length) * modality_weight
                observations.append((distance, support_weight))

        if not observations:
            return [(max(3.0, min_dim * 0.007), 1.0)]

        clusters: list[list[tuple[float, float]]] = []
        for value, weight in sorted(observations, key=lambda item: item[0]):
            placed = False
            for cluster in clusters:
                total_weight = sum(item[1] for item in cluster)
                center = sum(item[0] * item[1] for item in cluster) / max(total_weight, 1e-6)
                if abs(value - center) <= max(1.5, 0.20 * center):
                    cluster.append((value, weight))
                    placed = True
                    break
            if not placed:
                clusters.append([(value, weight)])

        profiles: list[tuple[float, float]] = []
        for cluster in clusters:
            total_weight = sum(item[1] for item in cluster)
            if total_weight <= 0.0:
                continue
            center = sum(item[0] * item[1] for item in cluster) / total_weight
            profiles.append((float(center), float(total_weight)))
        profiles.sort(key=lambda item: (-item[1], item[0]))
        return profiles[:8]

    def _double_face_candidates(
        self,
        *,
        drawing: DrawingModel,
        lines: Sequence[DrawingLine],
        thickness_profiles: Sequence[tuple[float, float]],
        perimeter_polygon_geom,
        wall_region_mask: np.ndarray | None,
        constraints: ReferenceConstraintResult | None,
    ) -> list[WallCandidate]:
        min_dim = min(drawing.width_px, drawing.height_px)
        minimum_length = max(12.0, min_dim * 0.022)
        max_thickness = max(12.0, min_dim * 0.040)
        minimum_thickness = max(1.5, min_dim * 0.0012)
        output: list[WallCandidate] = []
        for i, first in enumerate(lines):
            for second in lines[i + 1 :]:
                if self._angle_diff(first.angle_deg, second.angle_deg) > self.ANGLE_TOLERANCE_DEG:
                    continue
                overlap = self._overlap_ratio(first, second)
                if overlap < 0.48:
                    continue
                thickness = self._line_distance(first, second)
                if thickness < minimum_thickness or thickness > max_thickness:
                    continue
                center = self._center_overlap_segment(first, second)
                if center is None:
                    continue
                start, end, length, angle = center
                if length < minimum_length:
                    continue
                line = LineString([(start.x, start.y), (end.x, end.y)])
                containment = self._containment(line, perimeter_polygon_geom)
                if containment < 0.55:
                    continue
                # F02 ya resolvió el perímetro. Una hipótesis cuyo centerline
                # coincide mayoritariamente con la frontera no puede reaparecer
                # como DIVIDER dentro del Reconstruction Core.
                if self._perimeter_duplicate_ratio(
                    line=line,
                    polygon=perimeter_polygon_geom,
                    tolerance=max(2.0, 1.35 * thickness, min_dim * 0.024),
                ) >= 0.68:
                    continue

                thickness_support = self._thickness_support(thickness, thickness_profiles)
                region_support = self._region_support(
                    start=start,
                    end=end,
                    thickness=thickness,
                    mask=wall_region_mask,
                )
                sources = sorted({*first.sources, *second.sources})
                vector_support = float(any(source == "PYMUPDF" for source in sources))
                raster_support = float(any(source in {"OPENCV", "LSD_LOCAL"} for source in sources))
                source_consensus = self._source_consensus(sources)
                dashed_penalty = 1.0 if first.dashed and second.dashed else (0.5 if first.dashed or second.dashed else 0.0)
                semantic_support = 0.0
                axis_support = self._axis_support(
                    start=start,
                    end=end,
                    thickness=thickness,
                    constraints=constraints,
                )
                evidence = WallEvidenceVector(
                    pair_overlap=overlap,
                    thickness_support=thickness_support,
                    vector_support=vector_support,
                    raster_line_support=raster_support,
                    region_support=region_support,
                    source_consensus=source_consensus,
                    perimeter_containment=containment,
                    semantic_support=semantic_support,
                    axis_support=axis_support,
                    dashed_penalty=dashed_penalty,
                )
                independent_sources = len(self._independent_source_groups(sources))
                length_target = max(4.0 * thickness, min_dim * 0.055)
                length_support = max(0.0, min(1.0, length / max(length_target, 1.0)))
                prior = self.fusion.prior(
                    evidence,
                    generator="DOUBLE_FACE",
                    independent_source_count=independent_sources,
                    length_support=length_support,
                )
                ident = hashlib.sha1(
                    f"{drawing.level_view_id}|PAIR|{first.id}|{second.id}|{start.x:.2f}|{start.y:.2f}|{end.x:.2f}|{end.y:.2f}".encode()
                ).hexdigest()[:16]
                output.append(
                    WallCandidate(
                        id=f"{drawing.level_view_id}__WC_PAIR__{ident}",
                        generator="DOUBLE_FACE",
                        start=start,
                        end=end,
                        angle_deg=angle,
                        length_px=length,
                        thickness_px=thickness,
                        face_ids=[first.id, second.id],
                        evidence_ids=sorted({*first.evidence_ids, *second.evidence_ids}),
                        source_names=sources,
                        evidence=evidence,
                        prior_score=prior,
                        metadata={
                            "face_a": first.id,
                            "face_b": second.id,
                            "face_lineage_groups": [
                                sorted(self._line_lineage_ids(first)),
                                sorted(self._line_lineage_ids(second)),
                            ],
                            "lineage_line_ids": sorted(
                                self._line_lineage_ids(first) | self._line_lineage_ids(second)
                            ),
                        },
                    )
                )
        return output

    def _region_candidates(
        self,
        *,
        drawing: DrawingModel,
        region_lines: Sequence[DrawingLine],
        face_lines: Sequence[DrawingLine],
        thickness_profiles: Sequence[tuple[float, float]],
        perimeter_polygon_geom,
        wall_region_mask: np.ndarray | None,
        constraints: ReferenceConstraintResult | None,
    ) -> list[WallCandidate]:
        min_dim = min(drawing.width_px, drawing.height_px)
        minimum_length = max(14.0, min_dim * 0.018)
        output: list[WallCandidate] = []
        for region in region_lines:
            if region.length_px < minimum_length:
                continue
            line_geom = LineString([(region.start.x, region.start.y), (region.end.x, region.end.y)])
            containment = self._containment(line_geom, perimeter_polygon_geom)
            if containment < 0.70:
                continue
            estimated = float(region.metadata.get("estimated_thickness_px") or 0.0)
            if self._perimeter_duplicate_ratio(
                line=line_geom,
                polygon=perimeter_polygon_geom,
                tolerance=max(2.0, 1.35 * max(estimated, 2.0), min_dim * 0.024),
            ) >= 0.68:
                continue
            if estimated <= 0.0:
                estimated = thickness_profiles[0][0]
            aligned_faces = [
                line for line in face_lines
                if self._angle_diff(line.angle_deg, region.angle_deg) <= self.ANGLE_TOLERANCE_DEG
                and self._line_distance(line, region) <= max(estimated * 1.5, 5.0)
                and self._overlap_ratio(line, region) >= 0.35
            ]
            sources = sorted({source for item in aligned_faces for source in item.sources} | {"REGION_LOCAL"})
            vector_support = float(any("PYMUPDF" in item.sources for item in aligned_faces))
            raster_support = float(any(any(src in {"OPENCV", "LSD_LOCAL"} for src in item.sources) for item in aligned_faces))
            source_consensus = self._source_consensus(sources)
            region_support = self._region_support(
                start=region.start,
                end=region.end,
                thickness=max(estimated, 2.0),
                mask=wall_region_mask,
            )
            thickness_support = self._thickness_support(estimated, thickness_profiles)
            axis_support = self._axis_support(
                start=region.start,
                end=region.end,
                thickness=max(estimated, 2.0),
                constraints=constraints,
            )
            evidence = WallEvidenceVector(
                pair_overlap=0.0,
                thickness_support=thickness_support,
                vector_support=vector_support,
                raster_line_support=raster_support,
                region_support=region_support,
                source_consensus=source_consensus,
                perimeter_containment=containment,
                semantic_support=0.0,
                axis_support=axis_support,
                dashed_penalty=0.0,
            )
            independent_sources = len(self._independent_source_groups(sources))
            length_target = max(5.0 * estimated, min_dim * 0.060)
            length_support = max(0.0, min(1.0, region.length_px / max(length_target, 1.0)))
            prior = self.fusion.prior(
                evidence,
                generator="REGION_CENTERLINE",
                independent_source_count=independent_sources,
                length_support=length_support,
            )
            ident = hashlib.sha1(
                f"{drawing.level_view_id}|REGION|{region.id}|{region.start.x:.2f}|{region.start.y:.2f}|{region.end.x:.2f}|{region.end.y:.2f}".encode()
            ).hexdigest()[:16]
            output.append(
                WallCandidate(
                    id=f"{drawing.level_view_id}__WC_REGION__{ident}",
                    generator="REGION_CENTERLINE",
                    start=region.start,
                    end=region.end,
                    angle_deg=region.angle_deg,
                    length_px=region.length_px,
                    thickness_px=max(estimated, 1.0),
                    face_ids=[item.id for item in aligned_faces[:8]],
                    evidence_ids=sorted({eid for item in aligned_faces for eid in item.evidence_ids}),
                    source_names=sources,
                    evidence=evidence,
                    prior_score=prior,
                    metadata={
                        "region_line_id": region.id,
                        "face_lineage_groups": [
                            sorted(self._line_lineage_ids(item))
                            for item in aligned_faces[:8]
                        ],
                        "lineage_line_ids": sorted({
                            lineage_id
                            for item in aligned_faces[:8]
                            for lineage_id in self._line_lineage_ids(item)
                        }),
                    },
                )
            )
        return output

    def _dedupe_candidates(self, candidates: Sequence[WallCandidate]) -> list[WallCandidate]:
        ordered = sorted(candidates, key=lambda item: (-item.prior_score, -item.length_px))
        kept: list[WallCandidate] = []
        for candidate in ordered:
            duplicate = False
            for existing in kept:
                if self._angle_diff(candidate.angle_deg, existing.angle_deg) > 2.5:
                    continue
                if self._candidate_distance(candidate, existing) > max(2.0, 0.35 * min(candidate.thickness_px, existing.thickness_px)):
                    continue
                if self._candidate_overlap(candidate, existing) < 0.72:
                    continue
                duplicate = True
                break
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def _line_lineage_ids(line: DrawingLine) -> set[str]:
        """Expande la identidad de una línea sin mezclar modalidades geométricas.

        Context detectors trabajan sobre trazos físicos normalizados mientras
        Candidate Discovery usa líneas representativas. Esta función conserva el
        puente explícito entre ambos espacios de identidad.
        """
        ids: set[str] = {str(line.id)}
        ids.update(str(item) for item in line.evidence_ids if item)
        for key in (
            "merged_line_ids",
            "physical_stroke_member_line_ids",
            "context_equivalent_line_ids",
        ):
            raw = line.metadata.get(key)
            if isinstance(raw, (list, tuple, set)):
                ids.update(str(item) for item in raw if item)
        return ids

    @staticmethod
    def _mask(drawing: DrawingModel) -> np.ndarray | None:
        for layer in drawing.raster_layers:
            if layer.kind == "WALL_REGION_MASK":
                data = np.frombuffer(layer.png_bytes, dtype=np.uint8)
                return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return None

    @staticmethod
    def _independent_source_groups(sources: Sequence[str]) -> set[str]:
        """Agrupa procedencias por modalidad realmente independiente."""
        groups: set[str] = set()
        for source in sources:
            if source == "PYMUPDF":
                groups.add("VECTOR")
            elif source in {"OPENCV", "LSD_LOCAL", "REGION_LOCAL"}:
                groups.add("RASTER")
        return groups

    @classmethod
    def _source_consensus(cls, sources: Sequence[str]) -> float:
        # Mantiene la escala histórica del score (0.50 una modalidad, 0.75 dos)
        # pero elimina la falsa independencia OPENCV/LSD/REGION_LOCAL.
        count = len(cls._independent_source_groups(sources))
        if count <= 0:
            return 0.0
        if count == 1:
            return 0.50
        return 0.75

    @staticmethod
    def _axis_support(
        *,
        start: DrawingPoint,
        end: DrawingPoint,
        thickness: float,
        constraints: ReferenceConstraintResult | None,
    ) -> float:
        if constraints is None or not constraints.axes:
            return 0.0
        dx = end.x - start.x
        dy = end.y - start.y
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        horizontal = min(abs(angle), abs(180.0 - angle)) <= 5.0
        vertical = abs(angle - 90.0) <= 5.0
        best = 0.0
        tolerance = max(3.0, 1.6 * thickness)
        for axis in constraints.axes:
            if horizontal and axis.orientation != "horizontal":
                continue
            if vertical and axis.orientation != "vertical":
                continue
            if not horizontal and not vertical:
                continue
            coordinate = (start.y + end.y) / 2.0 if horizontal else (start.x + end.x) / 2.0
            distance = abs(coordinate - axis.coordinate_px)
            if distance > tolerance:
                continue
            local = max(0.0, 1.0 - distance / tolerance)
            if axis.confidence is not None:
                local *= 0.7 + 0.3 * axis.confidence
            best = max(best, local)
        return max(0.0, min(1.0, best))

    @staticmethod
    def _perimeter_duplicate_ratio(*, line: LineString, polygon, tolerance: float) -> float:
        if line.length <= 1e-6:
            return 0.0
        near_boundary = polygon.boundary.buffer(
            tolerance,
            cap_style=2,
            join_style=2,
        ).intersection(line).length
        return max(0.0, min(1.0, float(near_boundary / line.length)))

    @staticmethod
    def _containment(line: LineString, polygon) -> float:
        if line.length <= 1e-6:
            return 0.0
        inside = polygon.buffer(1.0, cap_style=2, join_style=2).intersection(line).length
        return max(0.0, min(1.0, float(inside / line.length)))

    @staticmethod
    def _region_support(
        *,
        start: DrawingPoint,
        end: DrawingPoint,
        thickness: float,
        mask: np.ndarray | None,
    ) -> float:
        if mask is None:
            return 0.0
        sample = np.zeros(mask.shape[:2], dtype=np.uint8)
        cv2.line(
            sample,
            (int(round(start.x)), int(round(start.y))),
            (int(round(end.x)), int(round(end.y))),
            255,
            max(1, int(round(max(2.0, thickness * 1.35)))),
            cv2.LINE_AA,
        )
        area = int(np.count_nonzero(sample))
        if area == 0:
            return 0.0
        overlap = int(np.count_nonzero(cv2.bitwise_and(sample, mask)))
        return max(0.0, min(1.0, overlap / area))

    @staticmethod
    def _thickness_support(value: float, profiles: Sequence[tuple[float, float]]) -> float:
        if value <= 0 or not profiles:
            return 0.0
        best = 0.0
        max_support = max(count for _, count in profiles)
        for center, count in profiles:
            relative = abs(value - center) / max(center, 1.0)
            proximity = max(0.0, 1.0 - relative / 0.35)
            family_support = count / max_support
            best = max(best, 0.72 * proximity + 0.28 * family_support)
        return max(0.0, min(1.0, best))

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    def _line_distance(self, first: DrawingLine, second: DrawingLine) -> float:
        return LineString([(first.start.x, first.start.y), (first.end.x, first.end.y)]).distance(
            LineString([(second.start.x, second.start.y), (second.end.x, second.end.y)])
        )

    def _overlap_ratio(self, first: DrawingLine, second: DrawingLine) -> float:
        ux, uy = self._unit(first)
        values_a = [first.start.x * ux + first.start.y * uy, first.end.x * ux + first.end.y * uy]
        values_b = [second.start.x * ux + second.start.y * uy, second.end.x * ux + second.end.y * uy]
        a0, a1 = min(values_a), max(values_a)
        b0, b1 = min(values_b), max(values_b)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1e-6, min(a1 - a0, b1 - b0))
        return max(0.0, min(1.0, overlap / denom))

    def _center_overlap_segment(self, first: DrawingLine, second: DrawingLine):
        ux, uy = self._unit(first)
        nx, ny = -uy, ux
        endpoints = [
            first.start, first.end, second.start, second.end,
        ]
        t_first = [first.start.x * ux + first.start.y * uy, first.end.x * ux + first.end.y * uy]
        t_second = [second.start.x * ux + second.start.y * uy, second.end.x * ux + second.end.y * uy]
        lo = max(min(t_first), min(t_second))
        hi = min(max(t_first), max(t_second))
        if hi <= lo:
            return None
        offsets = [point.x * nx + point.y * ny for point in endpoints]
        offset = sum(offsets) / len(offsets)
        start = DrawingPoint(x=lo * ux + offset * nx, y=lo * uy + offset * ny)
        end = DrawingPoint(x=hi * ux + offset * nx, y=hi * uy + offset * ny)
        length = math.hypot(end.x - start.x, end.y - start.y)
        angle = math.degrees(math.atan2(end.y - start.y, end.x - start.x)) % 180.0
        return start, end, length, angle

    @staticmethod
    def _unit(line: DrawingLine) -> tuple[float, float]:
        dx = line.end.x - line.start.x
        dy = line.end.y - line.start.y
        length = math.hypot(dx, dy)
        return dx / length, dy / length

    def _candidate_distance(self, first: WallCandidate, second: WallCandidate) -> float:
        return LineString([(first.start.x, first.start.y), (first.end.x, first.end.y)]).distance(
            LineString([(second.start.x, second.start.y), (second.end.x, second.end.y)])
        )

    def _candidate_overlap(self, first: WallCandidate, second: WallCandidate) -> float:
        a = DrawingLine(
            id="A", start=first.start, end=first.end, length_px=first.length_px,
            angle_deg=first.angle_deg, sources=[], evidence_ids=[]
        )
        b = DrawingLine(
            id="B", start=second.start, end=second.end, length_px=second.length_px,
            angle_deg=second.angle_deg, sources=[], evidence_ids=[]
        )
        return self._overlap_ratio(a, b)
