"""Exact saved inputs for testing Call 2 without rerunning F01/F03 or providers."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.quantia_spatialV1.adaptive_reconstruction.contracts import AdaptiveReconstructionRuntime, SingleLineWall, LogicalGap, SingleLineWallGraph
from app.quantia_spatialV1.adaptive_reconstruction.hybrid_engine import AdaptiveReconstructionEngine
from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import PixelBBox
from app.quantia_spatialV1.post_reconstruction_filters.contracts import PostReconstructionFilterResult
from app.quantia_spatialV1.reconstruction_core.candidate_models import WallCandidate

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SavedLevelRaster:
    """Only fields actually preserved in the snapshot; no invented page dimensions."""
    id: str
    level_name: str
    source_document_id: str | None
    source_page_number: int
    source_bbox_px: PixelBBox
    raster_width_px: int
    raster_height_px: int
    raster_bytes: bytes
    raster_mime_type: str = "image/png"


def load_snapshot(case="casa_viri", level="planta_alta"):
    validation = json.loads((ROOT / "documentation/LOCAL_VALIDATION_CANONICAL_WALLGRAPH_INTEGRITY_V2.json").read_text(encoding="utf-8"))
    run = ROOT / validation["output_directory"]
    paths = [p for p in (run / case).glob("*__canonical_wallgraph.json") if level in p.name]
    if len(paths) != 1:
        raise ValueError("Select exactly one saved LevelView")
    graph_path = paths[0]
    stem = graph_path.name.removesuffix("__canonical_wallgraph.json")
    bundle_path = graph_path.with_name(stem + "__reconstruction_evidence_bundle.json")
    post_path = graph_path.with_name(stem + "__filter_audit.json")
    for path in (graph_path, bundle_path, post_path):
        expected = validation["artifacts_sha256"][str(path.relative_to(ROOT))]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot hash mismatch: {path.name}")
    graph = SingleLineWallGraph.model_validate_json(graph_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    post = PostReconstructionFilterResult.model_validate_json(post_path.read_text(encoding="utf-8"))
    raster = (ROOT / "tests/output/canonical_visual_review" / run.name / (stem + "__original.png")).read_bytes()
    if hashlib.sha256(raster).hexdigest() != bundle["source_raster_sha256"]:
        raise ValueError("Original LevelView raster hash mismatch")
    view = SavedLevelRaster(graph.level_view_id, graph.level_name, bundle["source_document_id"], bundle["source_page_number"],
                           PixelBBox(**bundle["source_bbox_px"]), *graph.image_size_px, raster)
    adaptive = post.filtered_wall_graph.model_copy(update={
        "walls": [SingleLineWall(**row) for row in bundle["adaptive_walls"]],
        "logical_gaps": [LogicalGap(**row) for row in bundle["adaptive_logical_gaps"]],
    }, deep=True)
    artifacts = {name: bundle[name] for name in (
        "f03_seed_candidates", "discovered_candidates", "quarantined_candidates", "hybrid_candidates", "context_decisions", "context_regions")}
    for name, saved in bundle["preserved_artifacts"].items():
        if isinstance(saved, dict) and saved.get("kind") == "ndarray":
            raw = zlib.decompress(base64.b64decode(saved["data"]))
            if hashlib.sha256(raw).hexdigest() != saved["sha256"]:
                raise ValueError("Corrupt saved mask")
            artifacts[name] = np.frombuffer(raw, dtype=saved["dtype"]).reshape(saved["shape"]).copy()
        else:
            artifacts[name] = saved
    ids = {wall.id for wall in adaptive.walls}
    tracks = [track for track in AdaptiveReconstructionEngine._physical_walls(
        [WallCandidate(**row) for row in bundle["hybrid_candidates"]]) if track.wall_id in ids]
    if {track.wall_id for track in tracks} != ids:
        raise ValueError("Incomplete source wall tracks")
    runtime = AdaptiveReconstructionRuntime(
        graph=adaptive, wall_tracks=tracks, bridges=[],
        topology=AdaptiveReconstructionEngine._space_topology(artifacts["logical_mask"], px_per_m=graph.px_per_m),
        context_quarantined_count=adaptive.diagnostics.quarantined_candidate_count, artifacts=artifacts)
    evidence = [RawEvidence(**row) for row in bundle["raw_evidence"]]
    return view, graph, runtime, post, evidence, stem
