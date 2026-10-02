from __future__ import annotations

import pymupdf

from app.quantia_spatialV1.engine import QuantiaSpatialEngine
from app.quantia_spatialV1.stages.walls.core.reconstruction_pipeline import QuantiaReconstructionPipeline


def _pdf_declared_scale_1_50() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((40, 40), "PLANTA BAJA   ESCALA 1:50", fontsize=12)
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(100, 150, 500, 550))
    shape.draw_line((300, 150), (300, 550))
    shape.draw_line((100, 350), (500, 350))
    shape.finish(width=2)
    shape.commit()
    data = document.tobytes()
    document.close()
    return data


def _run(*, normalize: bool):
    return QuantiaSpatialEngine().run(
        document_bytes=_pdf_declared_scale_1_50(),
        media_mime_type="application/pdf",
        source_document_id="SYNTH_CANONICAL_RASTER_V2",
        render_scale=0.7935,
        known_level_names_by_page={1: ["Planta Baja"]},
        isolated_pages={1},
        enable_gemini_discovery=False,
        enable_gemini_extraction=False,
        enable_metric_raster_normalization=normalize,
    )


def test_f02_se_reproyecta_sin_redetectarse_y_f03_recibe_escala_levelview() -> None:
    bootstrap = _run(normalize=False)
    canonical = _run(normalize=True)

    assert len(bootstrap.levels) == 1
    assert len(canonical.levels) == 1
    before = bootstrap.levels[0]
    after = canonical.levels[0]

    assert before.perimeter.editable_perimeter is not None
    assert after.perimeter.editable_perimeter is not None
    assert after.perimeter.editable_perimeter.id == before.perimeter.editable_perimeter.id
    assert (
        after.perimeter.editable_perimeter.geometry_revision
        == before.perimeter.editable_perimeter.geometry_revision
    )
    assert after.level_view.raster_width_px > before.level_view.raster_width_px
    assert after.level_view.metric_scale_m_per_px is not None
    assert 89.0 < 1.0 / after.level_view.metric_scale_m_per_px < 91.0
    assert after.level_view.metric_scale_source == "CANONICAL_METRIC_RASTER_V2"

    project_scale = QuantiaReconstructionPipeline().build_project_scale_context(
        levels=[(after.level_view, after.perimeter.editable_perimeter)]
    )
    assert project_scale.state == "RESOLVED"
    profile = project_scale.for_level(after.level_view.id)
    assert profile.state == "RESOLVED"
    assert profile.local_m_per_px == after.level_view.metric_scale_m_per_px
    assert profile.source == "CANONICAL_METRIC_RASTER_V2"


def test_f02_reproyectado_permanece_valido_en_raster_final() -> None:
    canonical = _run(normalize=True)
    after = canonical.levels[0]

    assert canonical.raster_normalization is not None
    assert canonical.raster_normalization.state == "NORMALIZED"
    assert after.perimeter.editable_perimeter is not None
    assert after.perimeter.state in {"VALID", "REVIEW"}
    assert after.perimeter.validation.inside_level_view is True
    assert after.perimeter.validation.closed_wall_chain is True


def test_entrypoint_routes_post_f02_flags_without_changing_default_contract() -> None:
    default_result = _run(normalize=False)
    assert default_result.process_results == {}
    assert default_result.process_errors == {}
    assert default_result.project_scale_context is None

    class SpyEngine(QuantiaSpatialEngine):
        seen: tuple[bool, str] | None = None

        def _finalize_engine_result(self, **kwargs):
            self.seen = (
                bool(kwargs["run_post_f02"]),
                str(kwargs["call2_mode"]),
            )
            kwargs["run_post_f02"] = False
            return super()._finalize_engine_result(**kwargs)

    engine = SpyEngine()
    result = engine.run(
        document_bytes=_pdf_declared_scale_1_50(),
        media_mime_type="application/pdf",
        source_document_id="SYNTH_ENTRYPOINT_V1",
        render_scale=0.7935,
        known_level_names_by_page={1: ["Planta Baja"]},
        isolated_pages={1},
        enable_gemini_discovery=False,
        enable_gemini_extraction=False,
        enable_metric_raster_normalization=False,
        run_post_f02=True,
        call2_mode="AUTO",
    )

    assert engine.seen == (True, "AUTO")
    assert len(result.levels) == 1
