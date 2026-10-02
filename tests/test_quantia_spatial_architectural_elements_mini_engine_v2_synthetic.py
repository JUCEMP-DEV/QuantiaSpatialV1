from __future__ import annotations

import io

from PIL import Image

from app.quantia_spatialV1.stages.elements import (
    ArchitecturalElementsMiniEngine,
    ArchitecturalRAGIndex,
)
from app.quantia_spatialV1.core.models.level_view import (
    LevelView,
    LevelViewTransform,
    PixelBBox,
)
from app.quantia_spatialV1.core.models.parametric import (
    ParametricLevel,
    ParametricPoint,
    ParametricWall,
    QuantiaParametricModel,
)
from app.quantia_spatialV1.stages.walls.core.context_models import ContextRegion
from app.quantia_spatialV1.stages.walls.core.drawing_model import (
    DrawingBBox,
    DrawingLine,
    DrawingModel,
    DrawingModelDiagnostics,
    DrawingPoint,
)


def _png_bytes(width: int = 400, height: int = 300) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _level_view() -> LevelView:
    return LevelView(
        id="LV_TEST",
        level_name="Planta prueba",
        source_document_id="DOC",
        source_page_number=1,
        source_bbox_px=PixelBBox(x_min=0, y_min=0, x_max=400, y_max=300),
        source_page_width_px=400,
        source_page_height_px=300,
        raster_width_px=400,
        raster_height_px=300,
        raster_mime_type="image/png",
        raster_bytes=_png_bytes(),
        transform=LevelViewTransform(
            offset_x_px=0,
            offset_y_px=0,
            source_page_width_px=400,
            source_page_height_px=300,
            local_width_px=400,
            local_height_px=300,
        ),
        state="DETECTADO",
        confidence=1.0,
    )


def _drawing() -> DrawingModel:
    return DrawingModel(
        level_view_id="LV_TEST",
        width_px=400,
        height_px=300,
        lines=[
            DrawingLine(
                id="HOST_RAW",
                start=DrawingPoint(x=30, y=100),
                end=DrawingPoint(x=370, y=100),
                length_px=340,
                angle_deg=0,
                evidence_ids=["EV_HOST"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="WINDOW_FRAME_1",
                start=DrawingPoint(x=150, y=96),
                end=DrawingPoint(x=210, y=96),
                length_px=60,
                angle_deg=0,
                evidence_ids=["EV_WINDOW_1"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="WINDOW_FRAME_2",
                start=DrawingPoint(x=150, y=100),
                end=DrawingPoint(x=210, y=100),
                length_px=60,
                angle_deg=0,
                evidence_ids=["EV_WINDOW_2"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="WINDOW_FRAME_3",
                start=DrawingPoint(x=150, y=104),
                end=DrawingPoint(x=210, y=104),
                length_px=60,
                angle_deg=0,
                evidence_ids=["EV_WINDOW_3"],
                sources=["PYMUPDF"],
            ),
        ],
        regions=[],
        curves=[],
        texts=[],
        semantic_observations=[],
        raster_layers=[],
        diagnostics=DrawingModelDiagnostics(
            raw_evidence_count=4,
            normalized_line_count=4,
            normalized_region_count=0,
            normalized_curve_count=0,
            text_count=0,
            semantic_observation_count=0,
            lsd_line_count=0,
            region_centerline_count=0,
            source_counts={"PYMUPDF": 4},
        ),
    )


def _spatial_model() -> QuantiaParametricModel:
    wall = ParametricWall(
        id="WALL_1",
        level_id="LV_TEST",
        role="DIVIDER",
        reference_path=[
            ParametricPoint(x_px=30, y_px=100),
            ParametricPoint(x_px=370, y_px=100),
        ],
        anchor_mode="CENTERLINE",
        length_px=340,
        length_m=6.8,
        thickness_px=10,
        thickness_m=0.2,
        source_candidate_id="CAND_HOST",
        evidence_ids=["HOST_RAW", "EV_HOST"],
        confidence=0.95,
    )
    return QuantiaParametricModel(
        id="QPM_TEST",
        levels=[
            ParametricLevel(
                id="LV_TEST",
                name="Planta prueba",
                source_level_view_id="LV_TEST",
                source_f02_model_id="F02_TEST",
                walls=[wall],
            )
        ],
        model_revision=0,
    )


def _window_region(*, through: float = 0.05, verified: bool = True) -> ContextRegion:
    return ContextRegion(
        id="CTX_WINDOW",
        region_type="WINDOW_REGION",
        bbox=DrawingBBox(x_min=148, y_min=93, x_max=212, y_max=107),
        confidence=0.90,
        quarantine_enabled=verified,
        member_line_ids=["WINDOW_FRAME_1", "WINDOW_FRAME_2", "WINDOW_FRAME_3"],
        metadata={
            "detector": "F03_ELEMENT_CONTEXT_ISOLATION_V5",
            "source": "EMBEDDED_PARALLEL_WINDOW_FRAME_V4",
            "opening_verified": verified,
            "parallel_overlap_support": 0.95,
            "track_count": 3,
            "host_wall_support": 0.90,
            "host_embedding_support": 0.90,
            "through_wall_support": through,
            "architecture_support": 0.95,
            "host_line_ids": ["HOST_RAW", "EV_HOST"],
        },
    )


def _door_region(*, host_line_ids=None, verified: bool = True) -> ContextRegion:
    return ContextRegion(
        id="CTX_DOOR",
        region_type="DOOR_REGION",
        bbox=DrawingBBox(x_min=235, y_min=70, x_max=285, y_max=120),
        confidence=0.91,
        quarantine_enabled=verified,
        member_line_ids=["DOOR_LEAF"],
        metadata={
            "detector": "F03_ELEMENT_CONTEXT_ISOLATION_V5",
            "source": "CUBIC_BEZIER_DOOR_SWING",
            "opening_verified": verified,
            "door_arc_score": 0.92,
            "door_leaf_support": 0.82,
            "door_host_support": 0.78,
            "architecture_support": 0.95,
            "estimated_swing_radius_px": 46,
            "host_line_ids": host_line_ids if host_line_ids is not None else ["HOST_RAW", "EV_HOST"],
        },
    )


def test_rag_default_recovers_window_pattern():
    index = ArchitecturalRAGIndex.default()
    engine = ArchitecturalElementsMiniEngine(rag_index=index)
    result, _ = engine.run(
        level_view=_level_view(),
        drawing=_drawing(),
        spatial_model=_spatial_model(),
        regions=[_window_region()],
        publish=False,
    )
    assert result.proposals[0].rag_hits
    assert result.proposals[0].rag_hits[0].label == "WINDOW"


def test_verified_window_is_accepted_and_hosted():
    engine = ArchitecturalElementsMiniEngine()
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=_drawing(),
        spatial_model=_spatial_model(),
        regions=[_window_region()],
    )
    proposal = result.proposals[0]
    assert proposal.predicted_class == "WINDOW"
    assert proposal.decision == "ACCEPTED"
    assert proposal.host_wall.wall_id == "WALL_1"
    assert result.diagnostics.published_opening_count == 1
    assert merged.levels[0].openings[0].opening_type == "WINDOW"
    assert merged.levels[0].openings[0].host_wall_id == "WALL_1"


def test_verified_door_is_accepted_and_spatial_walls_are_immutable():
    engine = ArchitecturalElementsMiniEngine()
    source = _spatial_model()
    walls_before = [item.model_dump(mode="json") for item in source.levels[0].walls]
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=_drawing(),
        spatial_model=source,
        regions=[_door_region()],
    )
    walls_after = [item.model_dump(mode="json") for item in merged.levels[0].walls]
    assert result.proposals[0].decision == "ACCEPTED"
    assert result.proposals[0].predicted_class == "DOOR"
    assert merged.levels[0].openings[0].opening_type == "DOOR"
    assert walls_after == walls_before
    assert source.levels[0].openings == []


def test_continuous_wall_window_conflict_is_not_published():
    engine = ArchitecturalElementsMiniEngine()
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=_drawing(),
        spatial_model=_spatial_model(),
        regions=[_window_region(through=0.95, verified=False)],
    )
    proposal = result.proposals[0]
    assert proposal.decision == "REJECTED"
    assert result.diagnostics.published_opening_count == 0
    assert merged.levels[0].openings == []


def test_opening_without_host_remains_review_and_is_not_published():
    engine = ArchitecturalElementsMiniEngine()
    far_wall = ParametricWall(
        id="WALL_FAR",
        level_id="LV_TEST",
        role="DIVIDER",
        reference_path=[ParametricPoint(x_px=20, y_px=250), ParametricPoint(x_px=380, y_px=250)],
        anchor_mode="CENTERLINE",
        length_px=360,
        thickness_px=10,
        confidence=0.9,
    )
    model = _spatial_model()
    model.levels[0].walls = [far_wall]
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=_drawing(),
        spatial_model=model,
        regions=[_door_region(host_line_ids=[])],
    )
    assert result.proposals[0].decision == "REVIEW"
    assert result.diagnostics.published_opening_count == 0
    assert merged.levels[0].openings == []


def test_duplicate_openings_are_suppressed_in_publication():
    engine = ArchitecturalElementsMiniEngine()
    first = _window_region()
    second = first.model_copy(update={"id": "CTX_WINDOW_DUP"})
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=_drawing(),
        spatial_model=_spatial_model(),
        regions=[first, second],
    )
    assert result.diagnostics.accepted_count == 2
    assert result.diagnostics.published_opening_count == 1
    assert len(merged.levels[0].openings) == 1


def test_publish_false_never_changes_model_revision_or_openings():
    engine = ArchitecturalElementsMiniEngine()
    source = _spatial_model()
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=_drawing(),
        spatial_model=source,
        regions=[_window_region()],
        publish=False,
    )
    assert result.diagnostics.accepted_count == 1
    assert result.diagnostics.published_opening_count == 0
    assert merged.model_revision == source.model_revision
    assert merged.levels[0].openings == []


def test_broad_curve_hypothesis_survives_as_review_or_accept_without_touching_walls():
    from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingCurve
    from app.quantia_spatialV1.core.scale.level_scale_normalizer import LevelScaleProfile

    drawing = _drawing()
    drawing.curves = [
        DrawingCurve(
            id="CURVE_DOOR_BROAD",
            control_points=[
                DrawingPoint(x=250, y=100),
                DrawingPoint(x=250, y=72),
                DrawingPoint(x=278, y=72),
                DrawingPoint(x=278, y=100),
            ],
            bbox=DrawingBBox(x_min=250, y_min=72, x_max=278, y_max=100),
            evidence_ids=["EV_CURVE_DOOR"],
            sources=["PYMUPDF"],
            confidence=0.8,
        )
    ]
    # Hoja aproximada + host ya presente en drawing.
    drawing.lines.append(
        DrawingLine(
            id="DOOR_LEAF_BROAD",
            start=DrawingPoint(x=250, y=100),
            end=DrawingPoint(x=274, y=78),
            length_px=32.6,
            angle_deg=137.5,
            evidence_ids=["EV_DOOR_LEAF_BROAD"],
            sources=["PYMUPDF"],
        )
    )
    profile = LevelScaleProfile(
        level_view_id="LV_TEST",
        state="RESOLVED",
        local_m_per_px=0.02,
        canonical_m_per_px=0.02,
        scale_factor_to_canonical=1.0,
        local_min_dim_px=300,
        normalized_min_dim_px=300,
        project_reference_min_dim_normalized_px=300,
        project_reference_area_normalized_px2=120000,
        source="TEST",
        confidence=1.0,
    )
    source = _spatial_model()
    walls_before = [item.model_dump(mode="json") for item in source.levels[0].walls]
    engine = ArchitecturalElementsMiniEngine()
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=drawing,
        spatial_model=source,
        scale_profile=profile,
        regions=None,
    )
    assert any(
        item.metadata.get("source_detector") == "ARCHITECTURAL_ELEMENTS_HYPOTHESES_V2"
        for item in result.proposals
    )
    assert all(
        item.decision in {"ACCEPTED", "REVIEW", "REJECTED"}
        for item in result.proposals
    )
    walls_after = [item.model_dump(mode="json") for item in merged.levels[0].walls]
    assert walls_after == walls_before



def _independent_drawing() -> DrawingModel:
    from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingCurve

    return DrawingModel(
        level_view_id="LV_TEST",
        width_px=400,
        height_px=300,
        lines=[
            DrawingLine(
                id="HOST_LEFT",
                start=DrawingPoint(x=30, y=100),
                end=DrawingPoint(x=145, y=100),
                length_px=115,
                angle_deg=0,
                evidence_ids=["EV_HOST"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="HOST_RIGHT",
                start=DrawingPoint(x=215, y=100),
                end=DrawingPoint(x=370, y=100),
                length_px=155,
                angle_deg=0,
                evidence_ids=["EV_HOST"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="WIN_1",
                start=DrawingPoint(x=150, y=96),
                end=DrawingPoint(x=210, y=96),
                length_px=60,
                angle_deg=0,
                evidence_ids=["EV_WIN_1"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="WIN_2",
                start=DrawingPoint(x=150, y=100),
                end=DrawingPoint(x=210, y=100),
                length_px=60,
                angle_deg=0,
                evidence_ids=["EV_WIN_2"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="WIN_3",
                start=DrawingPoint(x=150, y=104),
                end=DrawingPoint(x=210, y=104),
                length_px=60,
                angle_deg=0,
                evidence_ids=["EV_WIN_3"],
                sources=["PYMUPDF"],
            ),
            DrawingLine(
                id="DOOR_LEAF_INDEPENDENT",
                start=DrawingPoint(x=250, y=100),
                end=DrawingPoint(x=290, y=60),
                length_px=56.57,
                angle_deg=135,
                evidence_ids=["EV_DOOR_LEAF"],
                sources=["PYMUPDF"],
            ),
        ],
        regions=[],
        curves=[
            DrawingCurve(
                id="DOOR_CURVE_INDEPENDENT",
                control_points=[
                    DrawingPoint(x=250, y=100),
                    DrawingPoint(x=250, y=78),
                    DrawingPoint(x=268, y=60),
                    DrawingPoint(x=290, y=60),
                ],
                bbox=DrawingBBox(x_min=250, y_min=60, x_max=290, y_max=100),
                evidence_ids=["EV_DOOR_CURVE"],
                sources=["PYMUPDF"],
                confidence=0.9,
            )
        ],
        texts=[],
        semantic_observations=[],
        raster_layers=[],
        diagnostics=DrawingModelDiagnostics(
            raw_evidence_count=7,
            normalized_line_count=6,
            normalized_region_count=0,
            normalized_curve_count=1,
            text_count=0,
            semantic_observation_count=0,
            lsd_line_count=0,
            region_centerline_count=0,
            source_counts={"PYMUPDF": 7},
        ),
    )


def _scale_profile():
    from app.quantia_spatialV1.core.scale.level_scale_normalizer import LevelScaleProfile
    return LevelScaleProfile(
        level_view_id="LV_TEST",
        state="RESOLVED",
        local_m_per_px=0.02,
        canonical_m_per_px=0.02,
        scale_factor_to_canonical=1.0,
        local_min_dim_px=300,
        normalized_min_dim_px=300,
        project_reference_min_dim_normalized_px=300,
        project_reference_area_normalized_px2=120000,
        source="TEST",
        confidence=1.0,
    )


def test_v2_discovers_door_and_window_without_f03_regions():
    engine = ArchitecturalElementsMiniEngine()
    source = _spatial_model()
    walls_before = [item.model_dump(mode="json") for item in source.levels[0].walls]
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=_independent_drawing(),
        spatial_model=source,
        scale_profile=_scale_profile(),
        regions=None,
    )
    assert result.diagnostics.independent_candidate_count >= 2
    assert result.diagnostics.door_count >= 1
    assert result.diagnostics.window_count >= 1
    assert all(
        proposal.metadata.get("source_detector") == "ARCHITECTURAL_ELEMENTS_HYPOTHESES_V2"
        for proposal in result.proposals
    )
    assert {opening.opening_type for opening in merged.levels[0].openings} >= {"DOOR", "WINDOW"}
    assert [item.model_dump(mode="json") for item in merged.levels[0].walls] == walls_before


def test_v2_no_longer_depends_on_element_context_detector_for_generation():
    from app.quantia_spatialV1.stages.elements.proposal_generator import OpeningHypothesisGenerator
    generator = OpeningHypothesisGenerator()
    regions = generator.generate(
        drawing=_independent_drawing(),
        walls=_spatial_model().levels[0].walls,
        scale_profile=_scale_profile(),
        architectural_polygon=None,
    )
    assert regions
    assert all(region.metadata.get("detector") == "ARCHITECTURAL_ELEMENTS_HYPOTHESES_V2" for region in regions)
    assert any(region.region_type == "DOOR_REGION" for region in regions)
    assert any(region.region_type == "WINDOW_REGION" for region in regions)



def test_v2_wall_gap_plus_persisted_semantics_recovers_door_without_curve():
    from app.quantia_spatialV1.stages.walls.core.drawing_model import SemanticObservation

    drawing = DrawingModel(
        level_view_id="LV_TEST",
        width_px=400,
        height_px=300,
        lines=[
            DrawingLine(id="FACE_A_L", start=DrawingPoint(x=30,y=95), end=DrawingPoint(x=150,y=95), length_px=120, angle_deg=0, evidence_ids=["EV_A"], sources=["PYMUPDF"]),
            DrawingLine(id="FACE_A_R", start=DrawingPoint(x=200,y=95), end=DrawingPoint(x=370,y=95), length_px=170, angle_deg=0, evidence_ids=["EV_A"], sources=["PYMUPDF"]),
            DrawingLine(id="FACE_B_L", start=DrawingPoint(x=30,y=105), end=DrawingPoint(x=150,y=105), length_px=120, angle_deg=0, evidence_ids=["EV_B"], sources=["PYMUPDF"]),
            DrawingLine(id="FACE_B_R", start=DrawingPoint(x=200,y=105), end=DrawingPoint(x=370,y=105), length_px=170, angle_deg=0, evidence_ids=["EV_B"], sources=["PYMUPDF"]),
        ],
        regions=[],
        curves=[],
        texts=[],
        semantic_observations=[
            SemanticObservation(
                id="SEM_DOOR",
                family="DOOR",
                bbox=DrawingBBox(x_min=148,y_min=88,x_max=202,y_max=112),
                confidence=0.92,
                payload={"semantic_category":"DOOR"},
            )
        ],
        raster_layers=[],
        diagnostics=DrawingModelDiagnostics(
            raw_evidence_count=5,
            normalized_line_count=4,
            normalized_region_count=0,
            normalized_curve_count=0,
            text_count=0,
            semantic_observation_count=1,
            lsd_line_count=0,
            region_centerline_count=0,
            source_counts={"PYMUPDF":4,"GEMINI":1},
        ),
    )
    engine = ArchitecturalElementsMiniEngine()
    result, merged = engine.run(
        level_view=_level_view(),
        drawing=drawing,
        spatial_model=_spatial_model(),
        scale_profile=_scale_profile(),
        regions=None,
    )
    assert any(
        item.metadata.get("source_detector") == "ARCHITECTURAL_ELEMENTS_HYPOTHESES_V2"
        and item.metadata.get("decision_reason") in {"VERIFIED_OPENING_WITH_HOST", "GEOMETRY_RAG_HOST_CONSENSUS"}
        and item.predicted_class == "DOOR"
        for item in result.proposals
    )
    assert result.diagnostics.door_count >= 1
    assert any(item.opening_type == "DOOR" for item in merged.levels[0].openings)
