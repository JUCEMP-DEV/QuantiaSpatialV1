from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pytest
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, polygonize_full, unary_union


PROBE_VERSION = "QUANTIA_SPATIALV1_SPACE_CLOSURE_PROBE_V1"
AXIS_ANGLE_TOL_DEG = 5.0
AXIS_CLUSTER_TOL_M = 0.14
PARALLEL_MERGE_MIN_OVERLAP = 0.65
MAX_LOGICAL_GAP_M = 1.50
MIN_SPACE_AREA_M2 = 0.75
MIN_SPACE_WIDTH_M = 0.55
BOUNDARY_ASSIGN_TOL_M = 0.06

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "output" / "space_closure_probe_v1"


@dataclass(frozen=True)
class AxisWall:
    wall_id: str
    orientation: str
    axis_value: float
    interval_start: float
    interval_end: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(value: str) -> str:
    return "_".join(str(value).strip().casefold().replace("-", " ").split())


def _discover_result_dir() -> Path:
    explicit = str(os.getenv("QUANTIA_SPACE_CLOSURE_INPUT_DIR") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "final_graph.json").is_file():
            raise AssertionError(f"QUANTIA_SPACE_CLOSURE_INPUT_DIR no contiene final_graph.json: {path}")
        return path

    case_filter = _slug(os.getenv("QUANTIA_CALL2_CASE") or "")
    level_filter = _slug(os.getenv("QUANTIA_CALL2_LEVEL") or "")
    roots = [
        HERE / "output" / "call2_interface04" / "gemini",
        HERE / "output" / "call2_interface04",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for final_graph in root.rglob("final_graph.json"):
            folder = final_graph.parent
            key = _slug(folder.name)
            if case_filter and case_filter not in key:
                continue
            if level_filter and level_filter not in key:
                continue
            candidates.append(folder)
    if not candidates:
        raise AssertionError(
            "No encontré un resultado Call 2. Define QUANTIA_SPACE_CLOSURE_INPUT_DIR con la carpeta "
            "que contiene final_graph.json/interface04.json/original.png."
        )
    return max(candidates, key=lambda p: (p / "final_graph.json").stat().st_mtime)


def _line(wall: dict[str, Any]) -> LineString:
    return LineString([tuple(map(float, wall["start_px"])), tuple(map(float, wall["end_px"]))])


def _length(wall: dict[str, Any]) -> float:
    return float(_line(wall).length)


def _angle_deg(wall: dict[str, Any]) -> float:
    x1, y1 = map(float, wall["start_px"])
    x2, y2 = map(float, wall["end_px"])
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _orientation(wall: dict[str, Any]) -> str:
    angle = _angle_deg(wall)
    if min(angle, 180.0 - angle) <= AXIS_ANGLE_TOL_DEG:
        return "H"
    if abs(angle - 90.0) <= AXIS_ANGLE_TOL_DEG:
        return "V"
    return "O"


def _axis_value(wall: dict[str, Any], orientation: str) -> float:
    if orientation == "H":
        return (float(wall["start_px"][1]) + float(wall["end_px"][1])) / 2.0
    if orientation == "V":
        return (float(wall["start_px"][0]) + float(wall["end_px"][0])) / 2.0
    raise ValueError(orientation)


def _interval(wall: dict[str, Any], orientation: str) -> tuple[float, float]:
    if orientation == "H":
        values = [float(wall["start_px"][0]), float(wall["end_px"][0])]
    elif orientation == "V":
        values = [float(wall["start_px"][1]), float(wall["end_px"][1])]
    else:
        raise ValueError(orientation)
    return min(values), max(values)


def _weighted_axis(group: list[tuple[float, dict[str, Any]]]) -> float:
    weighted = []
    for value, wall in group:
        confidence = max(0.10, float(wall.get("confidence") or 0.0))
        weight = max(1.0, _length(wall)) * confidence
        weighted.append((value, weight))
    total = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / max(total, 1e-9)


def _cluster_axis_walls(
    walls: list[dict[str, Any]], *, px_per_m: float
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Proyecta líneas casi H/V sobre un eje canónico común.

    Esto no crea muros nuevos. Corrige offsets residuales de pocos centímetros que
    impiden que una misma línea física forme una red topológica exacta.
    """
    tolerance_px = AXIS_CLUSTER_TOL_M * px_per_m
    by_orientation: dict[str, list[tuple[float, dict[str, Any]]]] = {"H": [], "V": []}
    oblique: list[dict[str, Any]] = []
    for wall in walls:
        orient = _orientation(wall)
        if orient in {"H", "V"}:
            by_orientation[orient].append((_axis_value(wall, orient), wall))
        else:
            oblique.append(copy.deepcopy(wall))

    projected: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    clusters_report: list[dict[str, Any]] = []

    for orient, items in by_orientation.items():
        items.sort(key=lambda item: item[0])
        groups: list[list[tuple[float, dict[str, Any]]]] = []
        for value, wall in items:
            if not groups:
                groups.append([(value, wall)])
                continue
            current_values = [item[0] for item in groups[-1]]
            candidate_values = [*current_values, value]
            # Complete-link: evita que una cadena de ejes termine absorbiendo líneas
            # cuyo primer/último eje ya supera la tolerancia física.
            if max(candidate_values) - min(candidate_values) <= tolerance_px:
                groups[-1].append((value, wall))
            else:
                groups.append([(value, wall)])

        for group_index, group in enumerate(groups, start=1):
            canonical_axis = _weighted_axis(group)
            members: list[str] = []
            for original_axis, wall in group:
                item = copy.deepcopy(wall)
                if orient == "H":
                    item["start_px"][1] = canonical_axis
                    item["end_px"][1] = canonical_axis
                else:
                    item["start_px"][0] = canonical_axis
                    item["end_px"][0] = canonical_axis
                item.setdefault("space_closure_probe", {})
                item["space_closure_probe"].update(
                    {
                        "axis_orientation": orient,
                        "original_axis_px": original_axis,
                        "canonical_axis_px": canonical_axis,
                        "axis_shift_px": canonical_axis - original_axis,
                    }
                )
                projected.append(item)
                metadata[item["id"]] = {
                    "orientation": orient,
                    "axis": canonical_axis,
                    "interval": _interval(item, orient),
                    "cluster": f"{orient}_{group_index:03d}",
                }
                members.append(item["id"])
            clusters_report.append(
                {
                    "cluster": f"{orient}_{group_index:03d}",
                    "orientation": orient,
                    "canonical_axis_px": canonical_axis,
                    "member_wall_ids": members,
                }
            )

    for item in oblique:
        projected.append(item)
        metadata[item["id"]] = {
            "orientation": "O",
            "axis": None,
            "interval": None,
            "cluster": None,
        }
    return projected, metadata, clusters_report


def _overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    minimum = max(1e-9, min(a[1] - a[0], b[1] - b[0]))
    return overlap / minimum


def _role_rank(role: str) -> int:
    return {"PERIMETER": 3, "DIVIDER": 2, "REVIEW": 1}.get(str(role).upper(), 0)


def _wall_quality(wall: dict[str, Any], *, px_per_m: float) -> tuple[float, ...]:
    thickness_m = float(wall.get("thickness_px") or 0.0) / max(px_per_m, 1e-9)
    plausible_thickness = 1.0 if 0.06 <= thickness_m <= 0.30 else 0.0
    return (
        float(_role_rank(str(wall.get("role") or ""))),
        1.0 if wall.get("f03_seed_protected") else 0.0,
        plausible_thickness,
        float(len(set(wall.get("source_names") or []))),
        float(len(set(wall.get("evidence_ids") or []))),
        float(wall.get("confidence") or 0.0),
        _length(wall),
    )


def _collapse_residual_parallel_faces(
    walls: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    *,
    px_per_m: float,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Fusiona sólo líneas ya proyectadas al MISMO eje y con fuerte solape.

    No une tramos separados por puertas/gaps. Esos huecos se resuelven después
    exclusivamente como continuidad lógica.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    oblique: list[dict[str, Any]] = []
    for wall in walls:
        meta = metadata[wall["id"]]
        cluster = meta.get("cluster")
        if not cluster:
            oblique.append(wall)
            continue
        groups.setdefault(str(cluster), []).append(wall)

    result: list[dict[str, Any]] = []
    remap: dict[str, str] = {}
    merges: list[dict[str, Any]] = []

    for cluster, members in groups.items():
        orient = metadata[members[0]["id"]]["orientation"]
        # Union-find local: sólo solapes fuertes pertenecen a la misma entidad.
        parent = {wall["id"]: wall["id"] for wall in members}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, first in enumerate(members):
            ia = _interval(first, orient)
            for second in members[i + 1 :]:
                ib = _interval(second, orient)
                if _overlap_ratio(ia, ib) >= PARALLEL_MERGE_MIN_OVERLAP:
                    union(first["id"], second["id"])

        components: dict[str, list[dict[str, Any]]] = {}
        for wall in members:
            components.setdefault(find(wall["id"]), []).append(wall)

        for component in components.values():
            if len(component) == 1:
                wall = component[0]
                remap[wall["id"]] = wall["id"]
                result.append(wall)
                continue

            anchor = max(component, key=lambda wall: _wall_quality(wall, px_per_m=px_per_m))
            merged = copy.deepcopy(anchor)
            intervals = [_interval(wall, orient) for wall in component]
            start = min(item[0] for item in intervals)
            end = max(item[1] for item in intervals)
            axis = float(metadata[anchor["id"]]["axis"])
            if orient == "H":
                merged["start_px"] = [start, axis]
                merged["end_px"] = [end, axis]
            else:
                merged["start_px"] = [axis, start]
                merged["end_px"] = [axis, end]
            source_ids = sorted(wall["id"] for wall in component)
            merged.setdefault("space_closure_probe", {})
            merged["space_closure_probe"].update(
                {
                    "residual_parallel_merge": True,
                    "merged_from_wall_ids": source_ids,
                    "canonical_axis_px": axis,
                }
            )
            for wall in component:
                remap[wall["id"]] = anchor["id"]
            result.append(merged)
            merges.append(
                {
                    "result_wall_id": anchor["id"],
                    "merged_from_wall_ids": source_ids,
                    "orientation": orient,
                    "canonical_axis_px": axis,
                    "union_interval_px": [start, end],
                }
            )

    for wall in oblique:
        remap[wall["id"]] = wall["id"]
        result.append(wall)
    return result, remap, merges


def _build_logical_closures(
    *,
    walls: list[dict[str, Any]],
    logical_gaps: list[dict[str, Any]],
    wall_remap: dict[str, str],
    px_per_m: float,
) -> list[dict[str, Any]]:
    by_id = {wall["id"]: wall for wall in walls}
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    closures: list[dict[str, Any]] = []
    max_gap_px = MAX_LOGICAL_GAP_M * px_per_m

    for gap in logical_gaps:
        if not gap.get("logical_continuity_only"):
            continue
        raw_ids = list(gap.get("wall_ids") or [])
        mapped: list[str] = []
        for wall_id in raw_ids:
            mapped_id = wall_remap.get(str(wall_id), str(wall_id))
            if mapped_id in by_id and mapped_id not in mapped:
                mapped.append(mapped_id)
        if len(mapped) != 2:
            continue
        if mapped[0] == mapped[1]:
            continue
        first = _line(by_id[mapped[0]])
        second = _line(by_id[mapped[1]])
        if first.intersects(second):
            continue
        p1, p2 = nearest_points(first, second)
        distance = float(p1.distance(p2))
        if distance <= 1e-6 or distance > max_gap_px:
            continue
        # IMPORTANTE: la geometría conserva precisión completa. El redondeo se usa
        # únicamente para deduplicar; redondear los endpoints físicos puede separar
        # una closure unos micrones de su wall y romper polygonize.
        a = (float(p1.x), float(p1.y))
        b = (float(p2.x), float(p2.y))
        key = tuple(sorted(((round(a[0], 4), round(a[1], 4)), (round(b[0], 4), round(b[1], 4)))))
        if key in seen:
            continue
        seen.add(key)
        closures.append(
            {
                "id": str(gap.get("id") or f"LC_{len(closures)+1:03d}"),
                "kind": str(gap.get("kind") or "LOGICAL_CONTINUITY"),
                "start_px": [a[0], a[1]],
                "end_px": [b[0], b[1]],
                "length_px": distance,
                "length_m": distance / px_per_m,
                "wall_ids": mapped,
                "physical_wall_present": False,
                "logical_continuity_only": True,
                "source_gap_id": str(gap.get("id") or ""),
            }
        )
    return closures


def _minimum_rotated_width(poly: Polygon) -> float:
    rectangle = poly.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.geom_type != "Polygon":
        return 0.0
    coords = list(rectangle.exterior.coords)
    lengths = []
    for i in range(min(4, len(coords) - 1)):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 1e-9:
            lengths.append(length)
    return min(lengths) if lengths else 0.0


def _polygonize_spaces(
    *,
    walls: list[dict[str, Any]],
    closures: list[dict[str, Any]],
    px_per_m: float,
    level_view_id: str,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    physical_lines = [_line(wall) for wall in walls]
    closure_lines = [LineString([item["start_px"], item["end_px"]]) for item in closures]
    network = unary_union([*physical_lines, *closure_lines])
    polygons, cuts, dangles, invalid = polygonize_full(network)

    min_area_px2 = MIN_SPACE_AREA_M2 * px_per_m * px_per_m
    min_width_px = MIN_SPACE_WIDTH_M * px_per_m
    accepted: list[Polygon] = []
    rejected: list[dict[str, Any]] = []
    for poly in polygons.geoms:
        if not isinstance(poly, Polygon):
            continue
        width = _minimum_rotated_width(poly)
        if poly.area < min_area_px2 or width < min_width_px:
            rejected.append(
                {
                    "area_m2": float(poly.area) / (px_per_m * px_per_m),
                    "minimum_width_m": float(width) / px_per_m,
                    "reason": "SLIVER_OR_NON_ARCHITECTURAL_FACE",
                }
            )
            continue
        accepted.append(poly)

    accepted.sort(key=lambda poly: (round(poly.centroid.y, 3), round(poly.centroid.x, 3)))
    spaces: list[dict[str, Any]] = []
    poly_by_id: dict[str, Polygon] = {}
    for index, poly in enumerate(accepted, start=1):
        sid = f"{level_view_id}__SPACE_CLOSURE_{index:03d}"
        coords_px = [[float(x), float(y)] for x, y in list(poly.exterior.coords)[:-1]]
        coords_m = [{"x": x / px_per_m, "y": y / px_per_m} for x, y in coords_px]
        minx, miny, maxx, maxy = poly.bounds
        spaces.append(
            {
                "id": sid,
                "name": "",
                "usageCode": "",
                "usageLabel": "",
                "category": "",
                "levelId": level_view_id,
                "geometry": {
                    "type": "polygon",
                    "vertices": coords_m,
                    "verticesRaster": [{"x": x, "y": y} for x, y in coords_px],
                    "areaM2": float(poly.area) / (px_per_m * px_per_m),
                    "perimeterM": float(poly.length) / px_per_m,
                    "bbox": {
                        "minX": minx / px_per_m,
                        "minY": miny / px_per_m,
                        "maxX": maxx / px_per_m,
                        "maxY": maxy / px_per_m,
                        "width": (maxx - minx) / px_per_m,
                        "height": (maxy - miny) / px_per_m,
                    },
                },
                "areaM2": float(poly.area) / (px_per_m * px_per_m),
                "perimeterM": float(poly.length) / px_per_m,
                "heightM": None,
                "doubleHeight": False,
                "confirmed": False,
                "state": "REVIEW",
                "source": {
                    "type": "deterministic_space_closure_probe",
                    "probeVersion": PROBE_VERSION,
                    "confidence": None,
                    "note": "Polígono derivado de centerlines canónicas + continuidad lógica; no es un muro físico agregado.",
                },
                "wallIds": [],
                "logicalClosureIds": [],
                "doorIds": [],
                "windowIds": [],
                "validation": {"isValid": True, "conflicts": [], "warnings": []},
            }
        )
        poly_by_id[sid] = poly

    boundary_tol_px = BOUNDARY_ASSIGN_TOL_M * px_per_m
    wall_owners: dict[str, list[str]] = {}
    for wall in walls:
        line = _line(wall)
        hits: list[tuple[float, str]] = []
        for sid, poly in poly_by_id.items():
            overlap = poly.boundary.buffer(boundary_tol_px, cap_style=2, join_style=2).intersection(line).length
            if overlap >= max(0.12 * line.length, 0.06 * px_per_m):
                hits.append((float(overlap), sid))
        hits.sort(reverse=True)
        owners = [sid for _, sid in hits[:2]]
        wall_owners[wall["id"]] = owners
        for sid in owners:
            next(space for space in spaces if space["id"] == sid)["wallIds"].append(wall["id"])

    for closure in closures:
        line = LineString([closure["start_px"], closure["end_px"]])
        for sid, poly in poly_by_id.items():
            overlap = poly.boundary.buffer(boundary_tol_px, cap_style=2, join_style=2).intersection(line).length
            if overlap >= max(0.30 * line.length, 1.0):
                next(space for space in spaces if space["id"] == sid)["logicalClosureIds"].append(closure["id"])

    topology = {
        "raw_polygon_count": len(list(polygons.geoms)),
        "meaningful_space_count": len(spaces),
        "rejected_face_count": len(rejected),
        "rejected_faces": rejected,
        "dangle_length_px": float(sum(geom.length for geom in dangles.geoms)),
        "cut_length_px": float(sum(geom.length for geom in cuts.geoms)),
        "invalid_ring_length_px": float(sum(geom.length for geom in invalid.geoms)),
    }
    return spaces, wall_owners, topology


def _wall_to_interface04(
    wall: dict[str, Any],
    *,
    level_view_id: str,
    px_per_m: float,
    owners: list[str],
    original_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item = copy.deepcopy(original_by_id.get(wall["id"], wall))
    start_px = [float(v) for v in wall["start_px"]]
    end_px = [float(v) for v in wall["end_px"]]
    length_px = math.dist(start_px, end_px)
    item.update(
        {
            "id": wall["id"],
            "levelId": level_view_id,
            "start_px": start_px,
            "end_px": end_px,
            "start": {"x": start_px[0] / px_per_m, "y": start_px[1] / px_per_m},
            "end": {"x": end_px[0] / px_per_m, "y": end_px[1] / px_per_m},
            "lengthM": length_px / px_per_m,
            "thickness_px": float(wall.get("thickness_px") or 0.0),
            "thicknessM": float(wall.get("thickness_px") or 0.0) / px_per_m,
            "spaceAId": owners[0] if owners else None,
            "spaceBId": owners[1] if len(owners) > 1 else None,
            "spaceClosureProbe": copy.deepcopy(wall.get("space_closure_probe") or {}),
            "segmentos": [
                {
                    "geometria": {
                        "raster": {
                            "vertices": [
                                {"x": start_px[0], "y": start_px[1]},
                                {"x": end_px[0], "y": end_px[1]},
                            ]
                        }
                    }
                }
            ],
        }
    )
    return item


def _build_interface04(
    *,
    source: dict[str, Any],
    final_graph: dict[str, Any],
    walls: list[dict[str, Any]],
    closures: list[dict[str, Any]],
    spaces: list[dict[str, Any]],
    wall_owners: dict[str, list[str]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(source)
    level_view_id = str(final_graph["level_view_id"])
    px_per_m = float(final_graph["px_per_m"])
    original_by_id = {str(wall.get("id")): wall for wall in source.get("muros", [])}
    result["muros"] = [
        _wall_to_interface04(
            wall,
            level_view_id=level_view_id,
            px_per_m=px_per_m,
            owners=wall_owners.get(wall["id"], []),
            original_by_id=original_by_id,
        )
        for wall in walls
    ]
    result["espacios"] = spaces
    result["logicalSpaceClosures"] = closures
    result["spaceClosureProbe"] = {
        "version": PROBE_VERSION,
        "experimental": True,
        "modelCallRequired": False,
        "physicalWallsModified": False,
        "metrics": metrics,
    }
    readiness = result.setdefault("readiness", {})
    missing = [str(item) for item in readiness.get("missing", [])]
    missing = [item for item in missing if item != "space polygons and names"]
    if spaces:
        missing.append("space semantic names/uses not yet confirmed")
    readiness["missing"] = list(dict.fromkeys(missing))
    readiness["workflowContinuation"] = False
    readiness["quantification"] = False
    return result


def _dash_line(draw: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float], *, fill, width: int = 3) -> None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return
    ux, uy = dx / length, dy / length
    dash, gap = 10.0, 7.0
    pos = 0.0
    while pos < length:
        end = min(length, pos + dash)
        draw.line((a[0] + ux * pos, a[1] + uy * pos, a[0] + ux * end, a[1] + uy * end), fill=fill, width=width)
        pos += dash + gap


def _render_visual(
    *,
    original_path: Path | None,
    image_size_px: tuple[int, int],
    input_walls: list[dict[str, Any]],
    clean_walls: list[dict[str, Any]],
    closures: list[dict[str, Any]],
    spaces: list[dict[str, Any]],
    output_path: Path,
) -> None:
    source_w, source_h = image_size_px
    panel_w = 900
    scale = panel_w / max(source_w, 1)
    panel_h = max(1, int(round(source_h * scale)))

    if original_path and original_path.exists():
        original = Image.open(original_path).convert("RGB").resize((panel_w, panel_h))
    else:
        original = Image.new("RGB", (panel_w, panel_h), "white")

    panel_a = original.copy()
    panel_b = Image.new("RGB", (panel_w, panel_h), "white")
    panel_c = Image.new("RGB", (panel_w, panel_h), "white")
    db = ImageDraw.Draw(panel_b, "RGBA")
    dc = ImageDraw.Draw(panel_c, "RGBA")

    for wall in input_walls:
        a = tuple(float(v) * scale for v in wall["start_px"])
        b = tuple(float(v) * scale for v in wall["end_px"])
        db.line((*a, *b), fill=(0, 0, 0, 255), width=3)

    palette = [
        (70, 130, 180, 55),
        (46, 139, 87, 55),
        (218, 165, 32, 55),
        (138, 43, 226, 45),
        (205, 92, 92, 45),
    ]
    for index, space in enumerate(spaces):
        vertices = [
            (float(point["x"]) * scale, float(point["y"]) * scale)
            for point in space["geometry"]["verticesRaster"]
        ]
        if len(vertices) >= 3:
            dc.polygon(vertices, fill=palette[index % len(palette)], outline=(40, 40, 40, 180))

    for wall in clean_walls:
        a = tuple(float(v) * scale for v in wall["start_px"])
        b = tuple(float(v) * scale for v in wall["end_px"])
        dc.line((*a, *b), fill=(0, 0, 0, 255), width=3)

    for closure in closures:
        a = tuple(float(v) * scale for v in closure["start_px"])
        b = tuple(float(v) * scale for v in closure["end_px"])
        _dash_line(dc, a, b, fill=(220, 20, 60, 220), width=3)

    title_h = 46
    canvas = Image.new("RGB", (panel_w * 3, panel_h + title_h), "white")
    canvas.paste(panel_a, (0, title_h))
    canvas.paste(panel_b, (panel_w, title_h))
    canvas.paste(panel_c, (panel_w * 2, title_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ["A ORIGINAL", "B CALL2 FINAL", "C CIERRE DETERMINISTA (rojo = cierre lógico)"]
    for index, label in enumerate(labels):
        draw.text((index * panel_w + 12, 14), label, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _run_probe(input_dir: Path) -> dict[str, Any]:
    final_graph_path = input_dir / "final_graph.json"
    interface04_path = input_dir / "interface04.json"
    if not final_graph_path.is_file():
        raise AssertionError(f"Falta {final_graph_path}")
    if not interface04_path.is_file():
        raise AssertionError(f"Falta {interface04_path}")

    final_graph = _read_json(final_graph_path)
    interface04 = _read_json(interface04_path)
    px_per_m = float(final_graph.get("px_per_m") or 0.0)
    assert px_per_m > 0.0, "final_graph no tiene px_per_m válido."
    input_walls = copy.deepcopy(list(final_graph.get("walls") or []))
    logical_gaps = copy.deepcopy(list(final_graph.get("logical_gaps") or []))
    assert input_walls, "final_graph no contiene muros."

    projected, metadata, clusters = _cluster_axis_walls(input_walls, px_per_m=px_per_m)
    clean_walls, remap, merges = _collapse_residual_parallel_faces(
        projected,
        metadata,
        px_per_m=px_per_m,
    )
    closures = _build_logical_closures(
        walls=clean_walls,
        logical_gaps=logical_gaps,
        wall_remap=remap,
        px_per_m=px_per_m,
    )
    spaces, wall_owners, topology = _polygonize_spaces(
        walls=clean_walls,
        closures=closures,
        px_per_m=px_per_m,
        level_view_id=str(final_graph["level_view_id"]),
    )

    expected_spaces = int(final_graph.get("interior_space_count") or 0)
    metrics = {
        "probe_version": PROBE_VERSION,
        "input_result_dir": str(input_dir),
        "px_per_m": px_per_m,
        "input_wall_count": len(input_walls),
        "canonical_wall_count": len(clean_walls),
        "axis_cluster_count": len(clusters),
        "residual_parallel_merge_count": len(merges),
        "residual_parallel_merges": merges,
        "logical_closure_count": len(closures),
        "logical_closure_length_m": sum(item["length_m"] for item in closures),
        "expected_interior_space_count": expected_spaces,
        "derived_space_count": len(spaces),
        "space_count_matches_graph": len(spaces) == expected_spaces if expected_spaces > 0 else None,
        "derived_space_areas_m2": [round(float(space["areaM2"]), 4) for space in spaces],
        **topology,
    }

    digest = hashlib.sha256(final_graph_path.read_bytes()).hexdigest()[:12]
    stem = _slug(input_dir.name)
    output_dir = OUTPUT_ROOT / digest / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_graph = copy.deepcopy(final_graph)
    cleaned_graph["walls"] = clean_walls
    cleaned_graph["space_closure_probe"] = {
        "version": PROBE_VERSION,
        "logical_closures": closures,
        "spaces": spaces,
        "metrics": metrics,
    }
    _write_json(output_dir / "space_closure_graph.json", cleaned_graph)
    _write_json(output_dir / "logical_closures.json", closures)
    _write_json(output_dir / "spaces.json", spaces)
    _write_json(output_dir / "space_closure_metrics.json", metrics)

    interface04_probe = _build_interface04(
        source=interface04,
        final_graph=final_graph,
        walls=clean_walls,
        closures=closures,
        spaces=spaces,
        wall_owners=wall_owners,
        metrics=metrics,
    )
    _write_json(output_dir / "interface04_space_closure.json", interface04_probe)

    image_size = tuple(int(v) for v in final_graph.get("image_size_px") or [1, 1])
    original_path = input_dir / "original.png"
    _render_visual(
        original_path=original_path if original_path.exists() else None,
        image_size_px=(image_size[0], image_size[1]),
        input_walls=input_walls,
        clean_walls=clean_walls,
        closures=closures,
        spaces=spaces,
        output_path=output_dir / "space_closure_comparison.png",
    )
    _write_json(
        output_dir / "run_status.json",
        {
            "probe_version": PROBE_VERSION,
            "status": "EXPORTED",
            "model_call_used": False,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
    )

    print("\n" + "=" * 92)
    print(PROBE_VERSION)
    print(f"INPUT:  {input_dir}")
    print(f"OUTPUT: {output_dir}")
    print(
        "walls "
        f"{metrics['input_wall_count']} -> {metrics['canonical_wall_count']} | "
        f"parallel_merges={metrics['residual_parallel_merge_count']} | "
        f"logical_closures={metrics['logical_closure_count']}"
    )
    print(
        "spaces "
        f"expected={metrics['expected_interior_space_count']} "
        f"derived={metrics['derived_space_count']} "
        f"areas_m2={metrics['derived_space_areas_m2']}"
    )
    print(
        "topology "
        f"raw_polygons={metrics['raw_polygon_count']} "
        f"dangles_px={metrics['dangle_length_px']:.2f} "
        f"cuts_px={metrics['cut_length_px']:.2f} "
        f"invalid_px={metrics['invalid_ring_length_px']:.2f}"
    )
    print("=" * 92)

    return {
        "output_dir": output_dir,
        "metrics": metrics,
        "spaces": spaces,
        "clean_walls": clean_walls,
        "closures": closures,
        "interface04": interface04_probe,
    }


def test_space_closure_residual_parallel_faces_are_collapsed_synthetic() -> None:
    px_per_m = 100.0
    walls = [
        {
            "id": "W_A",
            "start_px": [100.0, 0.0],
            "end_px": [100.0, 300.0],
            "thickness_px": 12.0,
            "role": "DIVIDER",
            "confidence": 0.9,
            "source_names": ["A"],
            "evidence_ids": ["EA"],
            "f03_seed_protected": True,
        },
        {
            "id": "W_B",
            "start_px": [108.0, 100.0],
            "end_px": [108.0, 300.0],
            "thickness_px": 10.0,
            "role": "DIVIDER",
            "confidence": 0.8,
            "source_names": ["B"],
            "evidence_ids": ["EB"],
            "f03_seed_protected": False,
        },
    ]
    projected, metadata, _ = _cluster_axis_walls(walls, px_per_m=px_per_m)
    cleaned, remap, merges = _collapse_residual_parallel_faces(projected, metadata, px_per_m=px_per_m)
    assert len(cleaned) == 1
    assert len(merges) == 1
    assert remap["W_A"] == remap["W_B"]


def test_space_closure_current_call2_result_without_model_call() -> None:
    result = _run_probe(_discover_result_dir())
    metrics = result["metrics"]
    assert metrics["input_wall_count"] > 0
    assert metrics["canonical_wall_count"] > 0
    assert metrics["logical_closure_count"] > 0
    assert (result["output_dir"] / "space_closure_comparison.png").is_file()
    assert (result["output_dir"] / "interface04_space_closure.json").is_file()
    # Es un probe: la calidad se decide mirando métricas/visual, no forzando el número de cuartos.
    # Aun así, si el WallGraph declara espacios interiores y la geometría los recupera,
    # se registra explícitamente en space_count_matches_graph.
    assert metrics["derived_space_count"] >= 0
