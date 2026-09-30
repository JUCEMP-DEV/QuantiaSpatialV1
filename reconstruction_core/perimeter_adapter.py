from __future__ import annotations

from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon

from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import EditablePerimeterModel


def perimeter_polygon(perimeter: EditablePerimeterModel) -> Polygon:
    vertices = sorted(perimeter.vertices, key=lambda item: item.sequence_index)
    polygon = Polygon([(float(v.point_px.x), float(v.point_px.y)) for v in vertices])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("F02 entregó un perímetro no utilizable por Reconstruction Core.")
    return polygon


def line_is_inside_perimeter(
    *,
    line: LineString,
    polygon: Polygon,
    tolerance_px: float = 2.0,
) -> bool:
    """Invariante F02: ningún DIVIDER publicable puede salir del perímetro."""
    if line.is_empty or line.length <= 1e-6:
        return False
    return bool(
        polygon.buffer(
            max(0.0, float(tolerance_px)),
            cap_style=2,
            join_style=2,
        ).covers(line)
    )


def constrain_line_to_perimeter(
    *,
    line: LineString,
    polygon: Polygon,
    max_rebase_px: float,
    clip_tolerance_px: float = 1.0,
) -> LineString | None:
    """
    Corrige únicamente una pequeña extensión numérica fuera de F02.

    Si una hipótesis necesita recortar más que `max_rebase_px`, no es un pequeño
    error de grounding: se rechaza completa. Esto evita convertir una línea de
    eje/grid que cruza el perímetro en un muro interior válido por simple clip.
    """
    if line.is_empty or line.length <= 1e-6:
        return None

    allowed = polygon.buffer(
        max(0.0, float(clip_tolerance_px)),
        cap_style=2,
        join_style=2,
    )
    if allowed.covers(line):
        return line

    intersection = allowed.intersection(line)
    pieces = _line_pieces(intersection)
    if not pieces:
        return None

    longest = max(pieces, key=lambda item: item.length)
    if longest.length <= 1e-6:
        return None

    removed = max(0.0, float(line.length - longest.length))
    if removed > max(0.0, float(max_rebase_px)) + 1e-6:
        return None

    return LineString(longest.coords)


def _line_pieces(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return [item for item in geometry.geoms if item.length > 1e-6]
    if isinstance(geometry, GeometryCollection):
        pieces: list[LineString] = []
        for item in geometry.geoms:
            pieces.extend(_line_pieces(item))
        return pieces
    return []
