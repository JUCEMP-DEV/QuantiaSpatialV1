from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.quantia_spatialV1.adaptive_reconstruction.contracts import (
    AdaptiveRoutePlan,
    ModuleDecision,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
)
from app.quantia_spatialV1.adaptive_reconstruction.correction_applier import WallGraphCorrectionApplier
from app.quantia_spatialV1.adaptive_reconstruction.multimodal_review import WallGraphMultimodalReviewer
from app.quantia_spatialV1.models.level_view import LevelView, LevelViewTransform, PixelBBox


class _FakeProvider:
    def analyze(self, **kwargs):
        return SimpleNamespace(
            provider="fake",
            model="fake-reviewer",
            fallback_used=False,
            raw={"usageMetadata": {"promptTokenCount": 1}},
            data={
                "level_view_id": "LV_SYNTH",
                "graph_state": "PARTIAL",
                "deltas": [
                    {
                        "action": "REMOVE_WALL",
                        "wall_id": "W2",
                        "confidence": 0.95,
                        "reason": "No coincide con un muro del plano base.",
                    },
                    {
                        "action": "ADD_WALL",
                        "wall_id": "W3",
                        "start_px": [10.0, 80.0],
                        "end_px": [90.0, 80.0],
                        "role_hint": "DIVIDER",
                        "confidence": 0.91,
                        "reason": "Muro visible omitido por Quantia.",
                    },
                ],
                "gap_decisions": [],
                "non_wall_architectural_regions": [],
                "unresolved_regions": [],
                "summary": "Corrección sintética",
            },
        )


def _graph() -> SingleLineWallGraph:
    plan = AdaptiveRoutePlan(
        mode="BALANCED",
        modules=[ModuleDecision(module="MULTIMODAL_CALL2", enabled=True, reason="synthetic")],
        recovery_strength=0.6,
        require_multimodal_review=True,
    )
    diag = ReconstructionDiagnostics(
        level_view_id="LV_SYNTH",
        f03_seed_count=1,
        discovered_candidate_count=2,
        quarantined_candidate_count=0,
        hybrid_candidate_count=2,
        selected_wall_count=2,
        component_count=1,
        junction_count=1,
        virtual_bridge_count=0,
        interior_space_count=1,
        review_wall_count=1,
        perimeter_wall_count=1,
        divider_wall_count=0,
        seed_coverage_ratio=1.0,
        quarantine_ratio=0.0,
        review_ratio=0.5,
    )
    return SingleLineWallGraph(
        level_view_id="LV_SYNTH",
        level_name="Synthetic",
        image_size_px=(100, 100),
        px_per_m=100.0,
        walls=[
            SingleLineWall(id="W1", start_px=(10, 10), end_px=(90, 10), thickness_px=10, role="PERIMETER", confidence=0.9),
            SingleLineWall(id="W2", start_px=(10, 50), end_px=(90, 50), thickness_px=10, role="REVIEW", confidence=0.5),
        ],
        logical_gaps=[],
        interior_space_count=1,
        route_plan=plan,
        diagnostics=diag,
    )


def _level_view() -> LevelView:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return LevelView(
        id="LV_SYNTH",
        level_name="Synthetic",
        source_document_id="SYNTH",
        source_page_number=1,
        source_bbox_px=PixelBBox(x_min=0, y_min=0, x_max=100, y_max=100),
        source_page_width_px=100,
        source_page_height_px=100,
        raster_width_px=100,
        raster_height_px=100,
        raster_bytes=encoded.tobytes(),
        raster_mime_type="image/png",
        transform=LevelViewTransform(
            offset_x_px=0, offset_y_px=0,
            source_page_width_px=100, source_page_height_px=100,
            local_width_px=100, local_height_px=100,
        ),
        state="DETECTADO",
    )


def test_call2_packet_and_delta_application_are_provider_agnostic(tmp_path) -> None:
    graph = _graph()
    reviewer = WallGraphMultimodalReviewer(provider=_FakeProvider(), history_path=tmp_path / "history.jsonl")
    level_view = _level_view()

    image_bytes = reviewer.build_review_image(level_view=level_view, graph=graph)
    assert len(image_bytes) > 100

    review = reviewer.review(level_view=level_view, graph=graph)
    corrected = WallGraphCorrectionApplier().apply(graph=graph, review=review)

    ids = {wall.id for wall in corrected.walls}
    assert "W1" in ids
    assert "W2" not in ids
    assert "W3" in ids
    assert (tmp_path / "history.jsonl").exists()
