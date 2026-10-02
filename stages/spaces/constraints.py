from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize_full, snap, unary_union

from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.stages.walls.adaptive.contracts import (
    MultimodalWallReview,
    SingleLineWallGraph,
)

from .contracts import (
    Call2RemovalAudit,
    ReferenceFootprintCandidate,
    SemanticSpaceAssignment,
    SemanticSpaceObservation,
    SpaceClosureResult,
    SpaceConstraintMetrics,
    SpaceConstraintResult,
    SpaceFaceAudit,
)


@dataclass(frozen=True)
class SpaceConstraintConfig:
    footprint_snap_m: float = 0.20
    min_face_area_m2: float = 0.50
    semantic_match_min_overlap: float = 0.10
    area_conservation_warn: float = 0.95
    semantic_separation_warn: float = 0.80


class SpaceConstraintValidator:
    VERSION = "SPACE_CONSTRAINT_VALIDATOR_V1"

    def __init__(
        self,
        config: SpaceConstraintConfig | None = None,
    ) -> None:
        self.config = config or SpaceConstraintConfig()

    def validate(
        self,
        *,
        closure: SpaceClosureResult,
        evidence: Sequence[RawEvidence],
        final_graph: SingleLineWallGraph,
        input_graph: SingleLineWallGraph | None = None,
        call2_review: MultimodalWallReview | None = None,
    ) -> SpaceConstraintResult:
        footprint, footprint_poly = self._reference_footprint(
            evidence,
            closure.px_per_m,
        )
        semantic_spaces = self._semantic_spaces(evidence)
        removals = self._call2_removals(
            input_graph=input_graph,
            review=call2_review,
            final_graph=final_graph,
            px_per_m=closure.px_per_m,
        )

        if footprint is None or footprint_poly is None:
            return SpaceConstraintResult(
                level_view_id=closure.level_view_id,
                state="UNRESOLVED",
                footprint=None,
                semantic_spaces=semantic_spaces,
                call2_removals=removals,
                warnings=["REFERENCE_FOOTPRINT_UNRESOLVED"],
            )

        faces, dangle_length_m = self._envelope_faces(
            closure=closure,
            footprint=footprint_poly,
        )
        assignments, face_report = self._semantic_partition(
            semantic_spaces=semantic_spaces,
            faces=faces,
            px_per_m=closure.px_per_m,
        )

        metrics = self._score(
            footprint_area=float(footprint_poly.area),
            face_area=sum(float(face.area) for face in faces),
            semantic_spaces=semantic_spaces,
            face_report=face_report,
            dangle_length_m=dangle_length_m,
        )

        warnings: list[str] = []
        if (
            metrics.area_conservation_ratio
            < self.config.area_conservation_warn
        ):
            warnings.append("SPACE_AREA_CONSERVATION_LOW")

        if (
            semantic_spaces
            and metrics.semantic_separation_ratio
            < self.config.semantic_separation_warn
        ):
            warnings.append("SPACE_SEMANTIC_SEPARATION_LOW")

        if any(
            face.state == "UNDER_SEGMENTED_MULTIPLE_SPACES"
            for face in face_report
        ):
            warnings.append("SPACE_UNDER_SEGMENTED")

        if any(
            item.requires_area_topology_review
            for item in removals
        ):
            warnings.append(
                "CALL2_REMOVED_TOPOLOGICALLY_RELEVANT_WALL"
            )

        return SpaceConstraintResult(
            level_view_id=closure.level_view_id,
            state="VALID" if not warnings else "REVIEW",
            footprint=footprint,
            semantic_spaces=semantic_spaces,
            assignments=assignments,
            faces=face_report,
            call2_removals=removals,
            metrics=metrics,
            warnings=warnings,
        )

    @staticmethod
    def _point(value: object) -> tuple[float, float]:
        if isinstance(value, dict):
            return float(value["x"]), float(value["y"])
        return float(value[0]), float(value[1])

    def _reference_footprint(
        self,
        evidence: Sequence[RawEvidence],
        px_per_m: float,
    ) -> tuple[
        ReferenceFootprintCandidate | None,
        Polygon | None,
    ]:
        candidates: list[
            tuple[float, Polygon, RawEvidence]
        ] = []

        for item in evidence:
            if (
                item.kind != "RASTER_CONTOUR"
                or not item.metadata.get("closed")
            ):
                continue

            points = (
                item.metadata.get("raw_contour_points_px")
                or []
            )
            area_px2 = float(
                item.metadata.get("area_px2")
                or 0.0
            )
            if area_px2 <= 0.0 or len(points) < 4:
                continue

            try:
                polygon = Polygon(
                    [self._point(point) for point in points]
                ).buffer(0)
            except Exception:
                continue

            if polygon.is_empty:
                continue

            if polygon.geom_type == "MultiPolygon":
                polygon = max(
                    polygon.geoms,
                    key=lambda geometry: geometry.area,
                )

            if (
                not isinstance(polygon, Polygon)
                or polygon.area <= 0
            ):
                continue

            candidates.append(
                (area_px2, polygon, item)
            )

        if not candidates:
            return None, None

        _, polygon, item = max(
            candidates,
            key=lambda row: row[0],
        )

        return (
            ReferenceFootprintCandidate(
                evidence_id=item.id,
                vertices_px=[
                    (float(x), float(y))
                    for x, y
                    in list(polygon.exterior.coords)[:-1]
                ],
                area_m2=(
                    float(polygon.area)
                    / (px_per_m * px_per_m)
                ),
                perimeter_m=(
                    float(polygon.length)
                    / px_per_m
                ),
            ),
            polygon,
        )

    @staticmethod
    def _semantic_spaces(
        evidence: Sequence[RawEvidence],
    ) -> list[SemanticSpaceObservation]:
        result: list[SemanticSpaceObservation] = []

        for item in evidence:
            if (
                item.metadata.get("semantic_category")
                != "SPACE"
                or item.geometry.bbox_px is None
            ):
                continue

            properties = (
                item.metadata.get("properties")
                or []
            )
            semantic_id = next(
                (
                    prop.get("value_text")
                    for prop in properties
                    if prop.get("name") == "id_proposed"
                ),
                None,
            )
            bbox = item.geometry.bbox_px

            result.append(
                SemanticSpaceObservation(
                    evidence_id=item.id,
                    semantic_id=semantic_id,
                    name=item.metadata.get(
                        "semantic_name"
                    ),
                    description=item.metadata.get(
                        "description"
                    ),
                    confidence=float(
                        item.confidence
                        or 0.0
                    ),
                    bbox_px=(
                        float(bbox.x_min),
                        float(bbox.y_min),
                        float(bbox.x_max),
                        float(bbox.y_max),
                    ),
                    has_explicit_dimensions=bool(
                        item.metadata.get(
                            "measurements"
                        )
                        or []
                    ),
                )
            )

        return result

    def _envelope_faces(
        self,
        *,
        closure: SpaceClosureResult,
        footprint: Polygon,
    ) -> tuple[list[Polygon], float]:
        physical = [
            LineString(
                [wall.start_px, wall.end_px]
            )
            for wall in closure.derived_walls
        ]
        logical = [
            LineString(
                [item.start_px, item.end_px]
            )
            for item in closure.logical_closures
        ]

        network = unary_union(
            [*physical, *logical]
        )
        tolerance_px = (
            self.config.footprint_snap_m
            * closure.px_per_m
        )
        network = snap(
            network,
            footprint.boundary,
            tolerance_px,
        )
        boundary = snap(
            footprint.boundary,
            network,
            tolerance_px,
        )
        polygons, _, dangles, _ = polygonize_full(
            unary_union([network, boundary])
        )

        faces: list[Polygon] = []
        for polygon in polygons.geoms:
            if (
                polygon.area
                / (
                    closure.px_per_m
                    * closure.px_per_m
                )
                < self.config.min_face_area_m2
            ):
                continue

            if (
                polygon.intersection(
                    footprint
                ).area
                / max(polygon.area, 1e-9)
                < 0.95
            ):
                continue

            faces.append(polygon)

        dangle_length_m = (
            float(
                sum(
                    geometry.length
                    for geometry
                    in dangles.geoms
                )
            )
            / closure.px_per_m
        )
        return faces, dangle_length_m

    def _semantic_partition(
        self,
        *,
        semantic_spaces: list[
            SemanticSpaceObservation
        ],
        faces: list[Polygon],
        px_per_m: float,
    ) -> tuple[
        list[SemanticSpaceAssignment],
        list[SpaceFaceAudit],
    ]:
        assignments: list[
            SemanticSpaceAssignment
        ] = []
        face_semantics: dict[
            int,
            list[str],
        ] = {
            index: []
            for index in range(len(faces))
        }

        for semantic in semantic_spaces:
            semantic_bbox = box(
                *semantic.bbox_px
            )
            best_index: int | None = None
            best_overlap = 0.0
            center_matches: list[int] = []

            for index, face in enumerate(faces):
                overlap = (
                    face.intersection(
                        semantic_bbox
                    ).area
                    / max(
                        semantic_bbox.area,
                        1e-9,
                    )
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = index

                if face.covers(
                    semantic_bbox.centroid
                ):
                    center_matches.append(index)

            assigned = (
                center_matches[0]
                if len(center_matches) == 1
                else best_index
            )
            matched = (
                assigned is not None
                and (
                    bool(center_matches)
                    or best_overlap
                    >= self.config.semantic_match_min_overlap
                )
            )

            if matched:
                face_semantics[assigned].append(
                    semantic.semantic_id
                    or semantic.name
                    or semantic.evidence_id
                )

            assignments.append(
                SemanticSpaceAssignment(
                    evidence_id=semantic.evidence_id,
                    semantic_id=semantic.semantic_id,
                    name=semantic.name,
                    assigned_face_index=(
                        assigned
                        if matched
                        else None
                    ),
                    center_match_count=len(
                        center_matches
                    ),
                    bbox_overlap_ratio=min(
                        1.0,
                        max(
                            0.0,
                            float(best_overlap),
                        ),
                    ),
                    status=(
                        "MATCHED"
                        if matched
                        else "UNMATCHED"
                    ),
                )
            )

        report: list[SpaceFaceAudit] = []
        for index, face in enumerate(faces):
            semantic_ids = face_semantics[
                index
            ]

            if not semantic_ids:
                state = "UNASSIGNED_FACE"
            elif len(semantic_ids) == 1:
                state = "SEMANTICALLY_SEPARATED"
            else:
                state = (
                    "UNDER_SEGMENTED_MULTIPLE_SPACES"
                )

            report.append(
                SpaceFaceAudit(
                    face_index=index,
                    area_m2=(
                        float(face.area)
                        / (
                            px_per_m
                            * px_per_m
                        )
                    ),
                    semantic_space_ids=semantic_ids,
                    state=state,
                )
            )

        return assignments, report

    def _call2_removals(
        self,
        *,
        input_graph: SingleLineWallGraph | None,
        review: MultimodalWallReview | None,
        final_graph: SingleLineWallGraph,
        px_per_m: float,
    ) -> list[Call2RemovalAudit]:
        if (
            input_graph is None
            or review is None
        ):
            return []

        source = {
            wall.id: wall
            for wall in input_graph.walls
        }
        final_ids = {
            wall.id
            for wall in final_graph.walls
        }
        result: list[Call2RemovalAudit] = []

        for delta in review.deltas:
            if (
                delta.action != "REMOVE_WALL"
                or not delta.wall_id
                or delta.wall_id not in source
            ):
                continue

            wall = source[delta.wall_id]
            length_m = (
                self._line_length(
                    wall.start_px,
                    wall.end_px,
                )
                / px_per_m
            )

            result.append(
                Call2RemovalAudit(
                    wall_id=wall.id,
                    role_before_call2=wall.role,
                    confidence_before_call2=(
                        wall.confidence
                    ),
                    f03_seed_protected=(
                        wall.f03_seed_protected
                    ),
                    length_m=length_m,
                    call2_reason=delta.reason,
                    still_in_final_graph=(
                        wall.id in final_ids
                    ),
                    requires_area_topology_review=(
                        wall.role == "PERIMETER"
                        or wall.f03_seed_protected
                    ),
                )
            )

        return result

    @staticmethod
    def _line_length(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (second[0] - first[0]) ** 2
            + (second[1] - first[1]) ** 2
        ) ** 0.5

    @staticmethod
    def _score(
        *,
        footprint_area: float,
        face_area: float,
        semantic_spaces: list[
            SemanticSpaceObservation
        ],
        face_report: list[SpaceFaceAudit],
        dangle_length_m: float,
    ) -> SpaceConstraintMetrics:
        area_conservation = min(
            1.0,
            face_area
            / max(footprint_area, 1e-9),
        )
        semantic_total = max(
            1,
            len(semantic_spaces),
        )
        separated = sum(
            1
            for face in face_report
            if len(face.semantic_space_ids) == 1
        )
        assigned = sum(
            len(face.semantic_space_ids)
            for face in face_report
        )

        semantic_separation = (
            separated
            / semantic_total
        )
        semantic_assignment = min(
            1.0,
            assigned
            / semantic_total,
        )
        dangle_quality = (
            1.0
            / (
                1.0
                + max(
                    0.0,
                    dangle_length_m,
                )
                / 10.0
            )
        )
        closure_score = (
            0.40 * area_conservation
            + 0.30 * semantic_separation
            + 0.15 * semantic_assignment
            + 0.15 * dangle_quality
        )

        return SpaceConstraintMetrics(
            area_conservation_ratio=(
                area_conservation
            ),
            semantic_separation_ratio=(
                semantic_separation
            ),
            semantic_assignment_ratio=(
                semantic_assignment
            ),
            dangle_quality=dangle_quality,
            closure_score=closure_score,
        )
