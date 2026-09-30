from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize_full, snap, unary_union

from app.quantia_spatialV1.tests.test_quantia_spatial_v1_space_closure_probe_v1 import (
    _build_logical_closures,
    _cluster_axis_walls,
    _collapse_residual_parallel_faces,
    _discover_result_dir,
    _polygonize_spaces,
    _read_json,
    _slug,
    _write_json,
)

PROBE_VERSION = "QUANTIA_SPATIALV1_SPACE_CLOSURE_CONSTRAINT_PROBE_V2"
HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "output" / "space_closure_constraint_probe_v2"

FOOTPRINT_SNAP_M = 0.20
MIN_FACE_AREA_M2 = 0.50
SEMANTIC_MATCH_MIN_OVERLAP = 0.10
AREA_CONSERVATION_WARN = 0.95
SEMANTIC_SEPARATION_WARN = 0.80


def _find_sibling(input_dir: Path, filename: str) -> Path | None:
    path = input_dir / filename
    return path if path.is_file() else None


def _space_polygon(space: dict[str, Any], *, px_per_m: float) -> Polygon:
    geom = space.get("geometry") or {}
    raster = geom.get("verticesRaster")
    if raster:
        return Polygon([(float(v["x"]), float(v["y"])) for v in raster]).buffer(0)
    metric = geom.get("vertices") or []
    return Polygon([(float(v["x"]) * px_per_m, float(v["y"]) * px_per_m) for v in metric]).buffer(0)


def _extract_reference_footprint(evidence_bundle: dict[str, Any], *, px_per_m: float) -> dict[str, Any] | None:
    """Usa el mayor contorno raster cerrado como *candidato* de footprint.

    No se etiqueta como terreno ni como área construida confirmada. Es una restricción
    geométrica reproducible para detectar pérdidas masivas de área tras Call 2.
    """
    candidates: list[tuple[float, Polygon, dict[str, Any]]] = []
    for evidence in evidence_bundle.get("raw_evidence", []):
        if str(evidence.get("kind")) != "RASTER_CONTOUR":
            continue
        metadata = evidence.get("metadata") or {}
        if not metadata.get("closed"):
            continue
        points = metadata.get("raw_contour_points_px") or []
        area_px2 = float(metadata.get("area_px2") or 0.0)
        if area_px2 <= 0.0 or len(points) < 4:
            continue
        try:
            poly = Polygon([(float(p["x"]), float(p["y"])) for p in points]).buffer(0)
        except Exception:
            continue
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if not isinstance(poly, Polygon) or poly.area <= 0.0:
            continue
        candidates.append((area_px2, poly, evidence))
    if not candidates:
        return None
    _, poly, evidence = max(candidates, key=lambda item: item[0])
    metadata = evidence.get("metadata") or {}
    return {
        "evidence_id": evidence.get("id"),
        "polygon": poly,
        "area_px2": float(poly.area),
        "area_m2": float(poly.area) / (px_per_m * px_per_m),
        "perimeter_m": float(poly.length) / px_per_m,
        "bbox_px": [float(v) for v in poly.bounds],
        "source_area_px2": float(metadata.get("area_px2") or poly.area),
        "classification": "REFERENCE_FOOTPRINT_CANDIDATE_NOT_CONFIRMED_TERRAIN_AREA",
    }


def _extract_semantic_spaces(evidence_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for evidence in evidence_bundle.get("raw_evidence", []):
        metadata = evidence.get("metadata") or {}
        if metadata.get("semantic_category") != "SPACE":
            continue
        bbox_px = (evidence.get("geometry") or {}).get("bbox_px")
        if not bbox_px:
            continue
        measurements = list(metadata.get("measurements") or [])
        result.append(
            {
                "id": next(
                    (
                        p.get("value_text")
                        for p in metadata.get("properties", [])
                        if p.get("name") == "id_proposed"
                    ),
                    None,
                ),
                "name": metadata.get("semantic_name"),
                "description": metadata.get("description"),
                "confidence": float(evidence.get("confidence") or 0.0),
                "bbox_px": {
                    "x_min": float(bbox_px["x_min"]),
                    "y_min": float(bbox_px["y_min"]),
                    "x_max": float(bbox_px["x_max"]),
                    "y_max": float(bbox_px["y_max"]),
                },
                "measurements": measurements,
                "relations": copy.deepcopy(metadata.get("relations") or []),
                "has_explicit_dimensions": bool(measurements),
            }
        )
    return result


def _build_envelope_faces(
    *,
    walls: list[dict[str, Any]],
    closures: list[dict[str, Any]],
    footprint: Polygon,
    px_per_m: float,
) -> tuple[list[Polygon], dict[str, float]]:
    physical = [LineString([w["start_px"], w["end_px"]]) for w in walls]
    logical = [LineString([c["start_px"], c["end_px"]]) for c in closures]
    network = unary_union([*physical, *logical])
    tolerance_px = FOOTPRINT_SNAP_M * px_per_m
    network = snap(network, footprint.boundary, tolerance_px)
    boundary = snap(footprint.boundary, network, tolerance_px)
    combined = unary_union([network, boundary])
    polygons, cuts, dangles, invalid = polygonize_full(combined)

    faces: list[Polygon] = []
    for poly in polygons.geoms:
        if poly.area / (px_per_m * px_per_m) < MIN_FACE_AREA_M2:
            continue
        if poly.intersection(footprint).area / max(poly.area, 1e-9) < 0.95:
            continue
        faces.append(poly)
    return faces, {
        "dangle_length_m": float(sum(g.length for g in dangles.geoms)) / px_per_m,
        "cut_length_m": float(sum(g.length for g in cuts.geoms)) / px_per_m,
        "invalid_ring_length_m": float(sum(g.length for g in invalid.geoms)) / px_per_m,
    }


def _semantic_partition_audit(
    *, semantic_spaces: list[dict[str, Any]], faces: list[Polygon], px_per_m: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments: list[dict[str, Any]] = []
    face_semantics: dict[int, list[str]] = {index: [] for index in range(len(faces))}
    for semantic in semantic_spaces:
        bbox_data = semantic["bbox_px"]
        semantic_bbox = box(
            bbox_data["x_min"], bbox_data["y_min"], bbox_data["x_max"], bbox_data["y_max"]
        )
        best_index = None
        best_overlap = 0.0
        center_matches: list[int] = []
        for index, face in enumerate(faces):
            overlap = face.intersection(semantic_bbox).area / max(semantic_bbox.area, 1e-9)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
            if face.covers(semantic_bbox.centroid):
                center_matches.append(index)
        assigned_index = center_matches[0] if len(center_matches) == 1 else best_index
        status = "UNMATCHED"
        if assigned_index is not None and (center_matches or best_overlap >= SEMANTIC_MATCH_MIN_OVERLAP):
            status = "MATCHED"
            face_semantics[assigned_index].append(str(semantic.get("id") or semantic.get("name") or "SPACE"))
        assignments.append(
            {
                **copy.deepcopy(semantic),
                "assigned_face_index": assigned_index,
                "center_match_count": len(center_matches),
                "bbox_overlap_ratio": round(best_overlap, 6),
                "status": status,
            }
        )

    face_report: list[dict[str, Any]] = []
    for index, face in enumerate(faces):
        semantic_ids = face_semantics[index]
        if len(semantic_ids) == 0:
            state = "UNASSIGNED_FACE"
        elif len(semantic_ids) == 1:
            state = "SEMANTICALLY_SEPARATED"
        else:
            state = "UNDER_SEGMENTED_MULTIPLE_SPACES"
        face_report.append(
            {
                "face_index": index,
                "area_m2": round(float(face.area) / (px_per_m * px_per_m), 4),
                "semantic_space_ids": semantic_ids,
                "semantic_space_count": len(semantic_ids),
                "state": state,
            }
        )
    return assignments, face_report


def _audit_call2_removals(
    *, input_graph: dict[str, Any] | None, review: dict[str, Any] | None, final_graph: dict[str, Any], px_per_m: float
) -> list[dict[str, Any]]:
    if not input_graph or not review:
        return []
    source = {str(w.get("id")): w for w in input_graph.get("walls", [])}
    final_ids = {str(w.get("id")) for w in final_graph.get("walls", [])}
    result: list[dict[str, Any]] = []
    for delta in review.get("deltas", []):
        if delta.get("action") != "REMOVE_WALL":
            continue
        wall_id = str(delta.get("wall_id") or "")
        wall = source.get(wall_id)
        if not wall:
            continue
        length_m = LineString([wall["start_px"], wall["end_px"]]).length / px_per_m
        reason = str(delta.get("reason") or "")
        result.append(
            {
                "wall_id": wall_id,
                "role_before_call2": wall.get("role"),
                "confidence_before_call2": wall.get("confidence"),
                "f03_seed_protected": bool(wall.get("f03_seed_protected")),
                "length_m": round(float(length_m), 4),
                "call2_reason": reason,
                "still_in_final_graph": wall_id in final_ids,
                "requires_area_topology_review": bool(
                    str(wall.get("role") or "").upper() == "PERIMETER"
                    or wall.get("f03_seed_protected")
                ),
            }
        )
    return result


def _score(
    *,
    footprint_area: float,
    face_area: float,
    semantic_spaces: list[dict[str, Any]],
    face_report: list[dict[str, Any]],
    dangle_length_m: float,
) -> dict[str, float]:
    area_conservation = min(1.0, face_area / max(footprint_area, 1e-9))
    semantic_total = max(1, len(semantic_spaces))
    separated = sum(1 for face in face_report if face["semantic_space_count"] == 1)
    assigned = sum(face["semantic_space_count"] for face in face_report)
    semantic_separation = separated / semantic_total
    semantic_assignment = min(1.0, assigned / semantic_total)
    dangle_quality = 1.0 / (1.0 + max(0.0, dangle_length_m) / 10.0)
    closure_score = (
        0.40 * area_conservation
        + 0.30 * semantic_separation
        + 0.15 * semantic_assignment
        + 0.15 * dangle_quality
    )
    return {
        "area_conservation_ratio": round(area_conservation, 6),
        "semantic_separation_ratio": round(semantic_separation, 6),
        "semantic_assignment_ratio": round(semantic_assignment, 6),
        "dangle_quality": round(dangle_quality, 6),
        "closure_score": round(closure_score, 6),
    }


def _render_constraint_visual(
    *,
    original_path: Path | None,
    image_size_px: tuple[int, int],
    footprint: Polygon,
    semantic_spaces: list[dict[str, Any]],
    faces: list[Polygon],
    output_path: Path,
) -> None:
    if original_path and original_path.is_file():
        base = Image.open(original_path).convert("RGBA")
    else:
        base = Image.new("RGBA", image_size_px, "white")
    scale = min(1.0, 1400.0 / max(base.width, 1))
    if scale != 1.0:
        base = base.resize((int(base.width * scale), int(base.height * scale)))
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")

    fp = [(float(x) * scale, float(y) * scale) for x, y in footprint.exterior.coords]
    draw.line(fp, fill=(220, 40, 40, 230), width=4)

    for index, face in enumerate(faces):
        pts = [(float(x) * scale, float(y) * scale) for x, y in face.exterior.coords]
        draw.polygon(pts, fill=(60, 160, 90, 30), outline=(40, 120, 70, 150))
        c = face.representative_point()
        draw.text((c.x * scale, c.y * scale), f"F{index+1}", fill="black", font=ImageFont.load_default())

    for semantic in semantic_spaces:
        b = semantic["bbox_px"]
        rect = [b["x_min"] * scale, b["y_min"] * scale, b["x_max"] * scale, b["y_max"] * scale]
        draw.rectangle(rect, outline=(20, 80, 220, 220), width=3)
        draw.text((rect[0] + 4, rect[1] + 4), str(semantic.get("id") or semantic.get("name")), fill=(20, 50, 160, 255), font=ImageFont.load_default())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _run_constraint_probe(input_dir: Path) -> dict[str, Any]:
    final_graph = _read_json(input_dir / "final_graph.json")
    interface04 = _read_json(input_dir / "interface04.json")
    evidence_path = _find_sibling(input_dir, "evidence_bundle.json")
    assert evidence_path, "La prueba V2 requiere evidence_bundle.json de la misma corrida Call 2."
    evidence_bundle = _read_json(evidence_path)
    input_graph = _read_json(input_dir / "input_graph.json") if (input_dir / "input_graph.json").is_file() else None
    review = _read_json(input_dir / "review.json") if (input_dir / "review.json").is_file() else None

    px_per_m = float(final_graph.get("px_per_m") or 0.0)
    assert px_per_m > 0.0

    projected, metadata, clusters = _cluster_axis_walls(copy.deepcopy(final_graph.get("walls") or []), px_per_m=px_per_m)
    clean_walls, wall_remap, merges = _collapse_residual_parallel_faces(projected, metadata, px_per_m=px_per_m)
    closures = _build_logical_closures(
        walls=clean_walls,
        logical_gaps=copy.deepcopy(final_graph.get("logical_gaps") or []),
        wall_remap=wall_remap,
        px_per_m=px_per_m,
    )
    v1_spaces, _, v1_topology = _polygonize_spaces(
        walls=clean_walls,
        closures=closures,
        px_per_m=px_per_m,
        level_view_id=str(final_graph["level_view_id"]),
    )

    footprint_info = _extract_reference_footprint(evidence_bundle, px_per_m=px_per_m)
    assert footprint_info is not None, "No fue posible obtener un footprint de referencia de la evidencia raster."
    footprint: Polygon = footprint_info.pop("polygon")
    semantic_spaces = _extract_semantic_spaces(evidence_bundle)

    envelope_faces, envelope_topology = _build_envelope_faces(
        walls=clean_walls,
        closures=closures,
        footprint=footprint,
        px_per_m=px_per_m,
    )
    semantic_assignments, face_report = _semantic_partition_audit(
        semantic_spaces=semantic_spaces,
        faces=envelope_faces,
        px_per_m=px_per_m,
    )

    v1_polys = [_space_polygon(space, px_per_m=px_per_m) for space in v1_spaces]
    v1_union = unary_union(v1_polys) if v1_polys else Polygon()
    envelope_area_m2 = sum(face.area for face in envelope_faces) / (px_per_m * px_per_m)
    v1_area_m2 = float(v1_union.area) / (px_per_m * px_per_m)
    removals = _audit_call2_removals(
        input_graph=input_graph,
        review=review,
        final_graph=final_graph,
        px_per_m=px_per_m,
    )

    score = _score(
        footprint_area=float(footprint_info["area_m2"]),
        face_area=envelope_area_m2,
        semantic_spaces=semantic_spaces,
        face_report=face_report,
        dangle_length_m=float(envelope_topology["dangle_length_m"]),
    )

    under_segmented = [face for face in face_report if face["state"] == "UNDER_SEGMENTED_MULTIPLE_SPACES"]
    recommendations: list[dict[str, Any]] = []
    if score["area_conservation_ratio"] >= AREA_CONSERVATION_WARN and score["semantic_separation_ratio"] < SEMANTIC_SEPARATION_WARN:
        recommendations.append(
            {
                "code": "FOOTPRINT_CLOSED_BUT_INTERNAL_PARTITIONS_MISSING",
                "detail": "El área exterior se conserva, pero múltiples espacios semánticos siguen dentro del mismo polígono.",
                "action": "No crear muros por área. Revisar junctions/divisiones internas con evidencia existente.",
            }
        )
    if any(item["requires_area_topology_review"] for item in removals):
        recommendations.append(
            {
                "code": "CALL2_REMOVED_TOPOLOGICALLY_RELEVANT_LINES",
                "detail": "Call 2 eliminó líneas perimetrales o protegidas; deben auditarse contra footprint/topología antes de descartarlas como límites espaciales.",
                "action": "Mantenerlas como evidencia y permitir uso como boundary lógico si mejoran cierre sin publicarlas como muro físico.",
            }
        )

    metrics = {
        "probe_version": PROBE_VERSION,
        "input_result_dir": str(input_dir),
        "px_per_m": px_per_m,
        "physical_wall_count": len(clean_walls),
        "residual_parallel_merge_count": len(merges),
        "logical_gap_closure_count": len(closures),
        "first_call_semantic_space_count": len(semantic_spaces),
        "first_call_spaces_with_explicit_dimensions": sum(1 for s in semantic_spaces if s["has_explicit_dimensions"]),
        "v1_derived_space_count": len(v1_spaces),
        "v1_derived_area_m2": round(v1_area_m2, 4),
        "reference_footprint": footprint_info,
        "reference_footprint_area_m2": round(float(footprint_info["area_m2"]), 4),
        "v1_footprint_coverage_ratio": round(v1_union.intersection(footprint).area / max(footprint.area, 1e-9), 6),
        "envelope_face_count": len(envelope_faces),
        "envelope_area_m2": round(envelope_area_m2, 4),
        "under_segmented_face_count": len(under_segmented),
        "call2_removed_wall_count": len(removals),
        "call2_removed_topology_review_count": sum(1 for item in removals if item["requires_area_topology_review"]),
        "v1_dangle_length_m": round(float(v1_topology.get("dangle_length_px", 0.0)) / px_per_m, 4),
        **{k: round(float(v), 6) for k, v in envelope_topology.items()},
        **score,
    }

    digest = hashlib.sha256((input_dir / "final_graph.json").read_bytes()).hexdigest()[:12]
    out = OUTPUT_ROOT / digest / _slug(input_dir.name)
    out.mkdir(parents=True, exist_ok=True)

    _write_json(out / "space_constraint_metrics.json", metrics)
    _write_json(out / "first_call_semantic_spaces.json", semantic_spaces)
    _write_json(out / "semantic_space_assignments.json", semantic_assignments)
    _write_json(out / "envelope_faces.json", face_report)
    _write_json(out / "call2_removal_area_topology_audit.json", removals)
    _write_json(out / "recommendations.json", recommendations)

    result_interface = copy.deepcopy(interface04)
    result_interface["spaceConstraintProbeV2"] = {
        "version": PROBE_VERSION,
        "experimental": True,
        "modelCallRequired": False,
        "referenceFootprint": footprint_info,
        "metrics": metrics,
        "semanticAssignments": semantic_assignments,
        "faceReport": face_report,
        "recommendations": recommendations,
    }
    # No promovemos los envelope_faces a espacios de 04: los que contienen varias
    # semánticas son evidencia de sub-segmentación, no espacios finales.
    _write_json(out / "interface04_space_constraint_review.json", result_interface)

    image_size = tuple(int(v) for v in final_graph.get("image_size_px") or [1, 1])
    original = input_dir / "original.png"
    _render_constraint_visual(
        original_path=original if original.is_file() else None,
        image_size_px=(image_size[0], image_size[1]),
        footprint=footprint,
        semantic_spaces=semantic_spaces,
        faces=envelope_faces,
        output_path=out / "space_constraint_audit.png",
    )

    print("\n" + "=" * 96)
    print(PROBE_VERSION)
    print(f"INPUT:  {input_dir}")
    print(f"OUTPUT: {out}")
    print(
        f"first_call_spaces={metrics['first_call_semantic_space_count']} "
        f"with_dimensions={metrics['first_call_spaces_with_explicit_dimensions']}"
    )
    print(
        f"V1 coverage={metrics['v1_footprint_coverage_ratio']:.3f} "
        f"({metrics['v1_derived_area_m2']:.2f}/{metrics['reference_footprint_area_m2']:.2f} m2)"
    )
    print(
        f"envelope area_conservation={metrics['area_conservation_ratio']:.3f} "
        f"semantic_separation={metrics['semantic_separation_ratio']:.3f} "
        f"under_segmented_faces={metrics['under_segmented_face_count']}"
    )
    print(
        f"dangles V1={metrics['v1_dangle_length_m']:.2f}m "
        f"envelope={metrics['dangle_length_m']:.2f}m score={metrics['closure_score']:.3f}"
    )
    print("=" * 96)

    return {
        "output_dir": out,
        "metrics": metrics,
        "semantic_spaces": semantic_spaces,
        "semantic_assignments": semantic_assignments,
        "face_report": face_report,
        "removals": removals,
        "recommendations": recommendations,
    }


def test_space_constraint_probe_v2_current_call2_result_without_model_call() -> None:
    result = _run_constraint_probe(_discover_result_dir())
    metrics = result["metrics"]
    assert metrics["reference_footprint_area_m2"] > 0.0
    assert metrics["first_call_semantic_space_count"] >= 0
    assert 0.0 <= metrics["area_conservation_ratio"] <= 1.05
    assert 0.0 <= metrics["semantic_separation_ratio"] <= 1.0
    assert (result["output_dir"] / "space_constraint_audit.png").is_file()
    assert (result["output_dir"] / "call2_removal_area_topology_audit.json").is_file()


def test_space_constraint_probe_v2_does_not_invent_room_dimensions() -> None:
    input_dir = _discover_result_dir()
    evidence = _read_json(input_dir / "evidence_bundle.json")
    spaces = _extract_semantic_spaces(evidence)
    # La prueba conserva únicamente measurements explícitos del primer llamado.
    # No calcula ancho/largo de bbox como si fueran dimensiones arquitectónicas.
    for space in spaces:
        if not space["measurements"]:
            assert space["has_explicit_dimensions"] is False
