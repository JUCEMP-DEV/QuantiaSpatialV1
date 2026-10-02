from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import LineString
from shapely.ops import polygonize_full, snap, unary_union

from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel

from .candidate_models import WallCandidate
from .perimeter_adapter import perimeter_polygon


class TopologyAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_count: int = Field(ge=0)
    room_polygon_count: int = Field(ge=0)
    meaningful_room_count: int = Field(ge=0)
    dangle_length_px: float = Field(ge=0.0)
    cut_length_px: float = Field(ge=0.0)
    invalid_ring_length_px: float = Field(ge=0.0)
    topology_score: float


class RoomTopologyAnalyzer:
    """Evalúa la solución completa, no candidatos aislados.

    V3 evita que celdas estrechas generadas por escaleras/retículas cuenten como
    habitaciones solo por cerrar polígonos. Un espacio significativo debe superar
    área mínima y ancho mínimo relativo al espesor observado de muro.
    """

    def analyze(
        self,
        *,
        perimeter: EditablePerimeterModel,
        candidates: list[WallCandidate],
    ) -> TopologyAnalysis:
        polygon = perimeter_polygon(perimeter)
        if not candidates:
            return TopologyAnalysis(
                selected_count=0,
                room_polygon_count=1,
                meaningful_room_count=1,
                dangle_length_px=0.0,
                cut_length_px=0.0,
                invalid_ring_length_px=0.0,
                topology_score=0.0,
            )

        median_thickness = self._median([item.thickness_px for item in candidates]) or 4.0
        snap_tolerance = max(2.0, 0.75 * median_thickness)

        boundary = polygon.boundary
        interior_lines = []
        for item in candidates:
            line = LineString([(item.start.x, item.start.y), (item.end.x, item.end.y)])
            line = polygon.buffer(1.0, cap_style=2, join_style=2).intersection(line)
            if line.is_empty:
                continue
            interior_lines.append(line)

        network = unary_union([boundary, *interior_lines])
        snapped = snap(network, network, snap_tolerance)
        polygons, cuts, dangles, invalid = polygonize_full(snapped)
        room_polygons = [
            geom
            for geom in polygons.geoms
            if polygon.buffer(1.0).covers(geom.representative_point())
        ]

        minimum_room_area = max(16.0, polygon.area * 0.0020)
        minimum_room_width = max(12.0, 4.0 * median_thickness)
        meaningful = [
            geom
            for geom in room_polygons
            if geom.area >= minimum_room_area
            and self._minimum_rotated_width(geom) >= minimum_room_width
        ]

        dangle_length = sum(geom.length for geom in dangles.geoms)
        cut_length = sum(geom.length for geom in cuts.geoms)
        invalid_length = sum(geom.length for geom in invalid.geoms)
        scale = max(polygon.length, 1.0)

        # El perímetro solo produce una región. Solo las subdivisiones con escala
        # arquitectónica ganan room bonus; celdas delgadas no lo hacen.
        room_bonus = min(2.0, 0.30 * max(0, len(meaningful) - 1))
        penalty = (
            2.60 * (dangle_length / scale)
            + 1.40 * (cut_length / scale)
            + 1.80 * (invalid_length / scale)
        )
        return TopologyAnalysis(
            selected_count=len(candidates),
            room_polygon_count=len(room_polygons),
            meaningful_room_count=len(meaningful),
            dangle_length_px=float(dangle_length),
            cut_length_px=float(cut_length),
            invalid_ring_length_px=float(invalid_length),
            topology_score=float(room_bonus - penalty),
        )

    @staticmethod
    def _minimum_rotated_width(geometry) -> float:
        rectangle = geometry.minimum_rotated_rectangle
        if rectangle.is_empty or rectangle.geom_type != "Polygon":
            return 0.0
        coords = list(rectangle.exterior.coords)
        lengths: list[float] = []
        for index in range(min(4, len(coords) - 1)):
            x0, y0 = coords[index]
            x1, y1 = coords[index + 1]
            length = math.hypot(x1 - x0, y1 - y0)
            if length > 1e-6:
                lengths.append(length)
        return float(min(lengths)) if lengths else 0.0

    @staticmethod
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        return float(
            ordered[mid]
            if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
