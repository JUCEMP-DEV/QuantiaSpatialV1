from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, polygonize_full, unary_union

from app.quantia_spatialV1.stages.walls.adaptive.contracts import (
    SingleLineWall,
    SingleLineWallGraph,
)

from .contracts import (
    SpaceClosureDiagnostics,
    SpaceClosureResult,
    SpaceDerivedWall,
    SpaceGeometryCandidate,
    SpaceLogicalClosure,
)


@dataclass(frozen=True)
class SpaceClosureConfig:
    axis_angle_tolerance_deg: float = 5.0
    axis_cluster_tolerance_m: float = 0.14
    parallel_merge_min_overlap: float = 0.65
    max_logical_gap_m: float = 1.50
    min_space_area_m2: float = 0.75
    min_space_width_m: float = 0.55
    boundary_assign_tolerance_m: float = 0.06


class SpaceClosureEngine:
    """Construye topología de espacios sin modificar el WallGraph canónico."""

    VERSION = "SPACE_CLOSURE_ENGINE_V1"

    def __init__(self, config: SpaceClosureConfig | None = None) -> None:
        self.config = config or SpaceClosureConfig()

    def run(self, *, graph: SingleLineWallGraph) -> SpaceClosureResult:
        projected, clusters = self._project_axes(graph.walls, graph.px_per_m)
        derived, wall_remap, merge_count = self._collapse_parallel_faces(
            projected,
            graph.px_per_m,
        )
        closures = self._logical_closures(
            graph=graph,
            walls=derived,
            wall_remap=wall_remap,
        )
        spaces, wall_owners, topology = self._polygonize(
            level_view_id=graph.level_view_id,
            walls=derived,
            closures=closures,
            px_per_m=graph.px_per_m,
        )
        return SpaceClosureResult(
            level_view_id=graph.level_view_id,
            px_per_m=graph.px_per_m,
            derived_walls=derived,
            logical_closures=closures,
            spaces=spaces,
            wall_owners=wall_owners,
            wall_remap=wall_remap,
            diagnostics=SpaceClosureDiagnostics(
                input_wall_count=len(graph.walls),
                derived_wall_count=len(derived),
                axis_cluster_count=clusters,
                residual_parallel_merge_count=merge_count,
                logical_closure_count=len(closures),
                raw_polygon_count=topology["raw_polygon_count"],
                meaningful_space_count=len(spaces),
                rejected_face_count=topology["rejected_face_count"],
                dangle_length_px=topology["dangle_length_px"],
                cut_length_px=topology["cut_length_px"],
                invalid_ring_length_px=topology["invalid_ring_length_px"],
            ),
        )

    def _orientation(self, wall: SingleLineWall) -> str:
        x1, y1 = wall.start_px
        x2, y2 = wall.end_px
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        if min(angle, 180.0 - angle) <= self.config.axis_angle_tolerance_deg:
            return "H"
        if abs(angle - 90.0) <= self.config.axis_angle_tolerance_deg:
            return "V"
        return "O"

    @staticmethod
    def _length(
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        return math.hypot(end[0] - start[0], end[1] - start[1])

    @staticmethod
    def _axis_value(wall: SingleLineWall, orientation: str) -> float:
        if orientation == "H":
            return (wall.start_px[1] + wall.end_px[1]) / 2.0
        return (wall.start_px[0] + wall.end_px[0]) / 2.0

    @staticmethod
    def _interval(
        wall: SpaceDerivedWall,
        orientation: str,
    ) -> tuple[float, float]:
        values = (
            (wall.start_px[0], wall.end_px[0])
            if orientation == "H"
            else (wall.start_px[1], wall.end_px[1])
        )
        return min(values), max(values)

    def _project_axes(
        self,
        walls: list[SingleLineWall],
        px_per_m: float,
    ) -> tuple[list[SpaceDerivedWall], int]:
        tolerance_px = self.config.axis_cluster_tolerance_m * px_per_m
        by_orientation: dict[
            str,
            list[tuple[float, SingleLineWall]],
        ] = {"H": [], "V": []}
        oblique: list[SingleLineWall] = []

        for wall in walls:
            orientation = self._orientation(wall)
            if orientation in {"H", "V"}:
                by_orientation[orientation].append(
                    (self._axis_value(wall, orientation), wall)
                )
            else:
                oblique.append(wall)

        result: list[SpaceDerivedWall] = []
        cluster_count = 0
        for orientation, items in by_orientation.items():
            items.sort(key=lambda item: item[0])
            groups: list[list[tuple[float, SingleLineWall]]] = []
            for axis, wall in items:
                if not groups:
                    groups.append([(axis, wall)])
                    continue
                candidate = [value for value, _ in groups[-1]] + [axis]
                if max(candidate) - min(candidate) <= tolerance_px:
                    groups[-1].append((axis, wall))
                else:
                    groups.append([(axis, wall)])

            cluster_count += len(groups)
            for group in groups:
                weights = [
                    max(1.0, self._length(wall.start_px, wall.end_px))
                    * max(0.10, wall.confidence)
                    for _, wall in group
                ]
                canonical_axis = (
                    sum(
                        axis * weight
                        for (axis, _), weight in zip(group, weights)
                    )
                    / max(sum(weights), 1e-9)
                )
                for _, wall in group:
                    if orientation == "H":
                        start = (wall.start_px[0], canonical_axis)
                        end = (wall.end_px[0], canonical_axis)
                    else:
                        start = (canonical_axis, wall.start_px[1])
                        end = (canonical_axis, wall.end_px[1])
                    result.append(
                        self._derived(
                            wall,
                            start=start,
                            end=end,
                            orientation=orientation,
                            axis=canonical_axis,
                        )
                    )

        for wall in oblique:
            result.append(
                self._derived(
                    wall,
                    start=wall.start_px,
                    end=wall.end_px,
                    orientation="O",
                    axis=None,
                )
            )
        return result, cluster_count

    @staticmethod
    def _derived(
        wall: SingleLineWall,
        *,
        start: tuple[float, float],
        end: tuple[float, float],
        orientation: str,
        axis: float | None,
    ) -> SpaceDerivedWall:
        return SpaceDerivedWall(
            id=wall.id,
            source_wall_ids=[wall.id],
            start_px=start,
            end_px=end,
            thickness_px=wall.thickness_px,
            role=wall.role,
            confidence=wall.confidence,
            orientation=orientation,
            axis_px=axis,
        )

    @staticmethod
    def _overlap_ratio(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        overlap = max(
            0.0,
            min(first[1], second[1]) - max(first[0], second[0]),
        )
        minimum = max(
            1e-9,
            min(
                first[1] - first[0],
                second[1] - second[0],
            ),
        )
        return overlap / minimum

    @staticmethod
    def _role_rank(role: str) -> int:
        return {
            "PERIMETER": 3,
            "DIVIDER": 2,
            "REVIEW": 1,
        }.get(role.upper(), 0)

    def _collapse_parallel_faces(
        self,
        walls: list[SpaceDerivedWall],
        px_per_m: float,
    ) -> tuple[list[SpaceDerivedWall], dict[str, str], int]:
        del px_per_m
        groups: dict[tuple[str, float], list[SpaceDerivedWall]] = {}
        result: list[SpaceDerivedWall] = []
        remap: dict[str, str] = {}
        merge_count = 0

        for wall in walls:
            if wall.orientation == "O" or wall.axis_px is None:
                result.append(wall)
                remap[wall.id] = wall.id
            else:
                groups.setdefault(
                    (wall.orientation, round(wall.axis_px, 6)),
                    [],
                ).append(wall)

        for (orientation, _), members in groups.items():
            parent = {wall.id: wall.id for wall in members}

            def find(value: str) -> str:
                while parent[value] != value:
                    parent[value] = parent[parent[value]]
                    value = parent[value]
                return value

            def union(first: str, second: str) -> None:
                first_root = find(first)
                second_root = find(second)
                if first_root != second_root:
                    parent[second_root] = first_root

            for index, first in enumerate(members):
                first_interval = self._interval(first, orientation)
                for second in members[index + 1 :]:
                    if (
                        self._overlap_ratio(
                            first_interval,
                            self._interval(second, orientation),
                        )
                        >= self.config.parallel_merge_min_overlap
                    ):
                        union(first.id, second.id)

            components: dict[str, list[SpaceDerivedWall]] = {}
            for wall in members:
                components.setdefault(find(wall.id), []).append(wall)

            for component in components.values():
                if len(component) == 1:
                    wall = component[0]
                    result.append(wall)
                    remap[wall.id] = wall.id
                    continue

                anchor = max(
                    component,
                    key=lambda item: (
                        self._role_rank(item.role),
                        item.confidence,
                        self._length(item.start_px, item.end_px),
                    ),
                )
                intervals = [
                    self._interval(wall, orientation)
                    for wall in component
                ]
                start_value = min(value[0] for value in intervals)
                end_value = max(value[1] for value in intervals)
                axis = float(anchor.axis_px)
                start = (
                    (start_value, axis)
                    if orientation == "H"
                    else (axis, start_value)
                )
                end = (
                    (end_value, axis)
                    if orientation == "H"
                    else (axis, end_value)
                )
                merged_ids = sorted(
                    {
                        source_id
                        for wall in component
                        for source_id in wall.source_wall_ids
                    }
                )
                result.append(
                    SpaceDerivedWall(
                        id=anchor.id,
                        source_wall_ids=merged_ids,
                        start_px=start,
                        end_px=end,
                        thickness_px=anchor.thickness_px,
                        role=anchor.role,
                        confidence=anchor.confidence,
                        orientation=orientation,
                        axis_px=axis,
                    )
                )
                for wall in component:
                    for source_id in wall.source_wall_ids:
                        remap[source_id] = anchor.id
                merge_count += 1

        return result, remap, merge_count

    def _logical_closures(
        self,
        *,
        graph: SingleLineWallGraph,
        walls: list[SpaceDerivedWall],
        wall_remap: dict[str, str],
    ) -> list[SpaceLogicalClosure]:
        by_id = {wall.id: wall for wall in walls}
        seen: set[
            tuple[
                tuple[float, float],
                tuple[float, float],
            ]
        ] = set()
        result: list[SpaceLogicalClosure] = []
        max_gap_px = self.config.max_logical_gap_m * graph.px_per_m

        for gap in graph.logical_gaps:
            if not gap.logical_continuity_only:
                continue

            mapped: list[str] = []
            for wall_id in gap.wall_ids:
                mapped_id = wall_remap.get(wall_id, wall_id)
                if mapped_id in by_id and mapped_id not in mapped:
                    mapped.append(mapped_id)

            if len(mapped) != 2 or mapped[0] == mapped[1]:
                continue

            first = LineString(
                [by_id[mapped[0]].start_px, by_id[mapped[0]].end_px]
            )
            second = LineString(
                [by_id[mapped[1]].start_px, by_id[mapped[1]].end_px]
            )
            if first.intersects(second):
                continue

            first_point, second_point = nearest_points(first, second)
            distance = float(first_point.distance(second_point))
            if distance <= 1e-6 or distance > max_gap_px:
                continue

            start = (float(first_point.x), float(first_point.y))
            end = (float(second_point.x), float(second_point.y))
            key = tuple(
                sorted(
                    (
                        (round(start[0], 4), round(start[1], 4)),
                        (round(end[0], 4), round(end[1], 4)),
                    )
                )
            )
            if key in seen:
                continue
            seen.add(key)

            result.append(
                SpaceLogicalClosure(
                    id=gap.id,
                    kind=gap.kind,
                    start_px=start,
                    end_px=end,
                    length_px=distance,
                    length_m=distance / graph.px_per_m,
                    wall_ids=mapped,
                    source_gap_id=gap.id,
                )
            )

        return result

    @staticmethod
    def _minimum_rotated_width(poly: Polygon) -> float:
        rectangle = poly.minimum_rotated_rectangle
        if rectangle.is_empty or rectangle.geom_type != "Polygon":
            return 0.0
        coords = list(rectangle.exterior.coords)
        lengths = [
            math.hypot(
                coords[index + 1][0] - coords[index][0],
                coords[index + 1][1] - coords[index][1],
            )
            for index in range(min(4, len(coords) - 1))
        ]
        return min(
            (length for length in lengths if length > 1e-9),
            default=0.0,
        )

    def _polygonize(
        self,
        *,
        level_view_id: str,
        walls: list[SpaceDerivedWall],
        closures: list[SpaceLogicalClosure],
        px_per_m: float,
    ) -> tuple[
        list[SpaceGeometryCandidate],
        dict[str, list[str]],
        dict[str, float | int],
    ]:
        physical = [
            LineString([wall.start_px, wall.end_px])
            for wall in walls
        ]
        logical = [
            LineString([item.start_px, item.end_px])
            for item in closures
        ]
        network = unary_union([*physical, *logical])
        polygons, cuts, dangles, invalid = polygonize_full(network)

        min_area_px2 = (
            self.config.min_space_area_m2
            * px_per_m
            * px_per_m
        )
        min_width_px = self.config.min_space_width_m * px_per_m

        accepted: list[tuple[Polygon, float]] = []
        rejected_count = 0
        for poly in polygons.geoms:
            if not isinstance(poly, Polygon):
                continue
            width = self._minimum_rotated_width(poly)
            if poly.area < min_area_px2 or width < min_width_px:
                rejected_count += 1
                continue
            accepted.append((poly, width))

        accepted.sort(
            key=lambda item: (
                round(item[0].centroid.y, 3),
                round(item[0].centroid.x, 3),
            )
        )

        spaces: list[SpaceGeometryCandidate] = []
        poly_by_id: dict[str, Polygon] = {}
        for index, (poly, width) in enumerate(accepted, start=1):
            space_id = f"{level_view_id}__SPACE_{index:03d}"
            vertices_px = [
                (float(x), float(y))
                for x, y in list(poly.exterior.coords)[:-1]
            ]
            space = SpaceGeometryCandidate(
                id=space_id,
                level_view_id=level_view_id,
                vertices_px=vertices_px,
                vertices_m=[
                    (x / px_per_m, y / px_per_m)
                    for x, y in vertices_px
                ],
                area_m2=float(poly.area) / (px_per_m * px_per_m),
                perimeter_m=float(poly.length) / px_per_m,
                minimum_width_m=float(width) / px_per_m,
            )
            spaces.append(space)
            poly_by_id[space_id] = poly

        tolerance_px = (
            self.config.boundary_assign_tolerance_m
            * px_per_m
        )
        wall_owners: dict[str, list[str]] = {}
        for wall in walls:
            line = LineString([wall.start_px, wall.end_px])
            hits: list[tuple[float, str]] = []
            for space_id, poly in poly_by_id.items():
                overlap = (
                    poly.boundary
                    .buffer(
                        tolerance_px,
                        cap_style=2,
                        join_style=2,
                    )
                    .intersection(line)
                    .length
                )
                if overlap >= max(
                    0.12 * line.length,
                    0.06 * px_per_m,
                ):
                    hits.append((float(overlap), space_id))
            hits.sort(reverse=True)
            owners = [space_id for _, space_id in hits[:2]]
            for source_id in wall.source_wall_ids:
                wall_owners[source_id] = owners
            for space_id in owners:
                item = next(
                    space
                    for space in spaces
                    if space.id == space_id
                )
                item.wall_ids = sorted(
                    set(
                        item.wall_ids
                        + wall.source_wall_ids
                    )
                )

        for closure in closures:
            line = LineString(
                [closure.start_px, closure.end_px]
            )
            for space_id, poly in poly_by_id.items():
                overlap = (
                    poly.boundary
                    .buffer(
                        tolerance_px,
                        cap_style=2,
                        join_style=2,
                    )
                    .intersection(line)
                    .length
                )
                if overlap >= max(
                    0.30 * line.length,
                    1.0,
                ):
                    item = next(
                        space
                        for space in spaces
                        if space.id == space_id
                    )
                    item.logical_closure_ids = sorted(
                        set(
                            item.logical_closure_ids
                            + [closure.id]
                        )
                    )

        return (
            spaces,
            wall_owners,
            {
                "raw_polygon_count": len(list(polygons.geoms)),
                "rejected_face_count": rejected_count,
                "dangle_length_px": float(
                    sum(geom.length for geom in dangles.geoms)
                ),
                "cut_length_px": float(
                    sum(geom.length for geom in cuts.geoms)
                ),
                "invalid_ring_length_px": float(
                    sum(geom.length for geom in invalid.geoms)
                ),
            },
        )
