"""Review delivery experiment; does not publish or persist a workflow model."""
import hashlib
import json
import math

from app.quantia_spatialV1.stages.walls.process import QuantiaSpatialV1ProcessEngine


def export_review(*, view, graph, runtime, post, evidence, review, output):
    if review.level_view_id != view.id or graph.level_view_id != view.id:
        raise ValueError("Review/graph belongs to another LevelView")

    class SavedAdaptive:
        def run(self, **kwargs):
            return runtime

    class SavedFilter:
        def run(self, **kwargs):
            return post

    class CapturedReviewer:
        def review(self, **kwargs):
            if kwargs["graph"].model_dump() != graph.model_dump():
                raise ValueError("Captured review input differs from canonical graph")
            return review

    result = QuantiaSpatialV1ProcessEngine(
        adaptive_engine=SavedAdaptive(), post_filter_engine=SavedFilter(),
        multimodal_reviewer=CapturedReviewer(),
    ).run_level(level_view=view, perimeter=None, evidence=evidence,
                scale_profile=None, call2_mode="FORCE")
    final = result.final_wall_graph
    bundle = result.canonical.evidence_bundle
    scale = final.px_per_m
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Metric scale must be finite and positive")
    walls = []
    for wall in final.walls:
        vertices = [dict(zip(("x", "y"), point)) for point in (wall.start_px, wall.end_px)]
        walls.append({
            **wall.model_dump(mode="json"), "levelId": view.id,
            "start": {k: v / scale for k, v in vertices[0].items()},
            "end": {k: v / scale for k, v in vertices[1].items()},
            "lengthM": math.dist(wall.start_px, wall.end_px) / scale,
            "thicknessM": wall.thickness_px / scale, "heightM": None,
            "confirmed": False, "state": "REVIEW",
            "sourceWallIds": bundle.final_source_mapping[wall.id],
            "segmentos": [{"geometria": {"raster": {"vertices": vertices}}}],
        })
    encoded_graph = json.dumps(final.model_dump(mode="json"), sort_keys=True, allow_nan=False)
    payload = {
        "schemaVersion": "SPATIAL_INTERFACE04_REVIEW_V1",
        "revision": hashlib.sha256(encoded_graph.encode()).hexdigest(),
        "levelViewId": view.id,
        "coordinateSystem": {"raster": "local_px_xy_down", "metric": "local_m_xy_down", "pxPerM": scale},
        "planoBase": {"referencia": "original.png", "anchoPx": view.raster_width_px,
                      "altoPx": view.raster_height_px, "sha256": bundle.source_raster_sha256},
        "source": {"documentId": view.source_document_id, "pageNumber": view.source_page_number,
                   "bboxPx": bundle.source_bbox_px},
        "niveles": [{"id": view.id, "key": view.id, "name": view.level_name, "confirmed": False}],
        "muros": walls, "puertas": [], "ventanas": [], "espacios": [], "ejes": [], "cotas": [],
        "buckets": {"WALL": [w.id for w in final.walls],
                    "ARCHITECTURAL_ELEMENT": bundle.architectural_elements,
                    "EXCLUDED_GRAPHIC": bundle.excluded_graphics, "UNRESOLVED": bundle.unresolved},
        "call2Candidates": result.call2_validation.accepted_architectural_regions,
        "call2Unresolved": result.call2_validation.accepted_unresolved_regions,
        "logicalGaps": [g.model_dump(mode="json") for g in final.logical_gaps],
        "evidenceBundle": "evidence_bundle.json", "validation": "validation.json",
        "readiness": {"wallReview": True, "workflowContinuation": False, "quantification": False,
                      "missing": ["validated openings with host/offset/width", "space polygons and names",
                                  "grounded axes and dimensions", "wall and opening heights"]},
        "interiorSpaceCount": final.interior_space_count,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "original.png").write_bytes(view.raster_bytes)
    for name, data in {
        "interface04.json": payload, "final_graph.json": final.model_dump(mode="json"),
        "evidence_bundle.json": bundle.model_dump(mode="json"),
        "validation.json": result.call2_validation.model_dump(mode="json"),
        "review.json": review.model_dump(mode="json"),
        "integrity.json": result.canonical.integrity.model_dump(mode="json"),
    }.items():
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return payload
