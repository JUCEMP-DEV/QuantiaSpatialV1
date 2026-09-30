from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.quantia_spatialV1.adaptive_reconstruction.contracts import (
    AdaptiveRoutePlan,
    LogicalGap,
    ModuleDecision,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
)
from app.quantia_spatialV1.adaptive_reconstruction.multimodal_review import (
    CALL2_PROMPT_VERSION,
    WallGraphMultimodalReviewer,
)
from app.quantia_spatialV1.models.level_view import LevelView, LevelViewTransform, PixelBBox


@dataclass
class _Result:
    provider: str = "fake"
    model: str = "fake-call2-v2"
    fallback_used: bool = False
    raw: dict = None
    data: dict = None


class _FakeProvider:
    def __init__(self) -> None:
        self.last_prompt = ""
        self.last_media = b""
        self.last_schema = None

    def analyze(self, *, prompt, media_bytes, media_mime_type, response_json_schema=None):
        self.last_prompt = prompt
        self.last_media = media_bytes
        self.last_schema = response_json_schema
        return _Result(
            raw={"fake": True},
            data={
                "level_view_id": "LV_SYNTH",
                "graph_state": "VALID",
                "deltas": [],
                "non_wall_architectural_regions": [],
                "unresolved_regions": [],
                "summary": "No corrections required.",
            },
        )


def _level_view() -> LevelView:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    cv2.line(image, (20, 40), (220, 40), (0, 0, 0), 4)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return LevelView(
        id="LV_SYNTH",
        level_name="Synthetic",
        source_document_id="DOC_SYNTH",
        source_page_number=1,
        source_bbox_px=PixelBBox(x_min=10, y_min=20, x_max=250, y_max=180),
        source_page_width_px=300,
        source_page_height_px=220,
        raster_width_px=240,
        raster_height_px=160,
        raster_mime_type="image/png",
        raster_bytes=encoded.tobytes(),
        transform=LevelViewTransform(
            offset_x_px=10,
            offset_y_px=20,
            source_page_width_px=300,
            source_page_height_px=220,
            local_width_px=240,
            local_height_px=160,
        ),
        state="DETECTADO",
        confidence=1.0,
    )


def _graph() -> SingleLineWallGraph:
    route = AdaptiveRoutePlan(
        mode="BALANCED",
        modules=[ModuleDecision(module="MULTIMODAL_CALL2", enabled=True, reason="synthetic")],
        recovery_strength=0.5,
        require_multimodal_review=True,
    )
    diagnostics = ReconstructionDiagnostics(
        level_view_id="LV_SYNTH",
        f03_seed_count=1,
        discovered_candidate_count=2,
        quarantined_candidate_count=0,
        hybrid_candidate_count=2,
        selected_wall_count=2,
        component_count=1,
        junction_count=1,
        virtual_bridge_count=500,
        interior_space_count=1,
        review_wall_count=0,
        perimeter_wall_count=2,
        divider_wall_count=0,
        seed_coverage_ratio=1.0,
        quarantine_ratio=0.0,
        review_ratio=0.0,
    )
    gaps = [
        LogicalGap(
            id=f"G{i:03d}",
            kind="SYNTHETIC_INTERNAL_GAP",
            start_px=(float(i % 200), 80.0),
            end_px=(float((i % 200) + 2), 80.0),
            gap_px=2.0,
            wall_ids=["W1", "W2"],
        )
        for i in range(500)
    ]
    return SingleLineWallGraph(
        level_view_id="LV_SYNTH",
        level_name="Synthetic",
        image_size_px=(240, 160),
        px_per_m=100.0,
        walls=[
            SingleLineWall(
                id="W1", start_px=(20.0, 40.0), end_px=(110.0, 40.0),
                thickness_px=12.0, role="PERIMETER", confidence=0.9,
            ),
            SingleLineWall(
                id="W2", start_px=(130.0, 40.0), end_px=(220.0, 40.0),
                thickness_px=12.0, role="PERIMETER", confidence=0.9,
            ),
        ],
        logical_gaps=gaps,
        interior_space_count=1,
        route_plan=route,
        diagnostics=diagnostics,
    )


def test_call2_v2_prompt_is_source_anchored_and_compact() -> None:
    provider = _FakeProvider()
    reviewer = WallGraphMultimodalReviewer(provider=provider, persist_input_artifact=False)
    review = reviewer.review(level_view=_level_view(), graph=_graph(), filter_context={
        "architectural_candidates": [],
        "excluded_graphic_candidates": [{"source_wall_id": f"E{i}"} for i in range(200)],
        "unresolved_candidates": [{"source_wall_id": f"U{i}"} for i in range(200)],
    })

    assert review.level_view_id == "LV_SYNTH"
    assert CALL2_PROMPT_VERSION in provider.last_prompt
    assert "exact original LevelView crop" in provider.last_prompt
    assert "logical_gaps" not in provider.last_prompt
    assert "G499" not in provider.last_prompt
    assert "W1" in provider.last_prompt and "W2" in provider.last_prompt
    # The 500 internal gaps and 400 filter candidates must not inflate the prompt.
    assert len(provider.last_prompt) < 10_000
    assert provider.last_schema is not None
    assert "gap_decisions" not in provider.last_schema["properties"]

    decoded = cv2.imdecode(np.frombuffer(provider.last_media, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # two exact 160 px panels + 42 px header each + 12 px divider
    assert decoded.shape[0] == 160 * 2 + 42 * 2 + 12
    assert decoded.shape[1] == 240
