from __future__ import annotations

import base64
import zlib
import dataclasses
import hashlib
import json
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel

from app.quantia_spatialV1.stages.walls.adaptive.contracts import AdaptiveReconstructionRuntime, SingleLineWallGraph
from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.stages.walls.postfilter.contracts import PostReconstructionFilterResult

from .contracts import ReconstructionEvidenceBundle


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "encoding": "zlib+base64",
            "data": base64.b64encode(zlib.compress(value.tobytes())).decode("ascii"),
        }
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


class ReconstructionEvidenceBundleBuilder:
    """Publica evidencia serializable sin mezclarla con el WallGraph."""

    @staticmethod
    def _artifact_hash(value: Any) -> str:
        normalized = _json_safe(value)
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def build(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
        runtime: AdaptiveReconstructionRuntime,
        post_filter: PostReconstructionFilterResult,
        final_graph: SingleLineWallGraph,
        source_mapping: dict[str, list[str]],
        correction_review: dict | None = None,
    ) -> ReconstructionEvidenceBundle:
        artifacts = runtime.artifacts or {}
        artifact_hashes = {
            key: self._artifact_hash(value)
            for key, value in artifacts.items()
            if key in {"wall_mask", "logical_mask", "junctions", "component_report"}
        }
        return ReconstructionEvidenceBundle(
            level_view_id=level_view.id,
            level_name=level_view.level_name,
            source_document_id=level_view.source_document_id,
            source_page_number=level_view.source_page_number,
            source_bbox_px=level_view.source_bbox_px.model_dump(mode="json"),
            source_raster_sha256=hashlib.sha256(level_view.raster_bytes).hexdigest(),
            raw_evidence=[_json_safe(item) for item in evidence],
            f03_seed_candidates=[_json_safe(item) for item in artifacts.get("f03_seed_candidates", [])],
            discovered_candidates=[_json_safe(item) for item in artifacts.get("discovered_candidates", [])],
            quarantined_candidates=[_json_safe(item) for item in artifacts.get("quarantined_candidates", [])],
            hybrid_candidates=[_json_safe(item) for item in artifacts.get("hybrid_candidates", [])],
            context_decisions=[_json_safe(item) for item in artifacts.get("context_decisions", [])],
            context_regions=[_json_safe(item) for item in artifacts.get("context_regions", [])],
            adaptive_walls=[item.model_dump(mode="json") for item in runtime.graph.walls],
            adaptive_logical_gaps=[item.model_dump(mode="json") for item in runtime.graph.logical_gaps],
            post_filter_decisions=[item.model_dump(mode="json") for item in post_filter.decisions],
            architectural_elements=[item.model_dump(mode="json") for item in post_filter.architectural_elements],
            excluded_graphics=[item.model_dump(mode="json") for item in post_filter.excluded_graphics],
            unresolved=[item.model_dump(mode="json") for item in post_filter.unresolved],
            canonical_mapping=[item.model_dump(mode="json") for item in post_filter.canonical_mapping],
            final_wall_ids=[item.id for item in final_graph.walls],
            final_logical_gap_ids=[item.id for item in final_graph.logical_gaps],
            artifact_hashes=artifact_hashes,
            preserved_artifacts={
                key: _json_safe(value) for key, value in artifacts.items()
                if key in {"wall_mask", "logical_mask", "junctions", "component_report"}
            },
            final_source_mapping=source_mapping,
            correction_review=correction_review,
        )
