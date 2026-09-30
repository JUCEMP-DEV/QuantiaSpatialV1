from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from app.quantia_spatialV1.adaptive_reconstruction import (
    AdaptiveReconstructionEngine,
    AdaptiveReconstructionRuntime,
    Call2DeltaValidator,
    Call2ValidationResult,
    MultimodalWallReview,
    SingleLineWallGraph,
    WallGraphCorrectionApplier,
    WallGraphMultimodalReviewer,
)
from app.quantia_spatialV1.canonical_wallgraph import (
    CanonicalWallGraphFinalizer,
    CanonicalWallGraphResult,
)
from app.quantia_spatialV1.canonical_wallgraph.correction_lineage import apply_with_lineage
from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import EditablePerimeterModel
from app.quantia_spatialV1.post_reconstruction_filters import (
    PostReconstructionFilterEngine,
    PostReconstructionFilterResult,
)
from app.quantia_spatialV1.reconstruction_core.level_scale_normalizer import LevelScaleProfile


Call2Mode = Literal["OFF", "AUTO", "FORCE"]


@dataclass
class QuantiaSpatialV1ProcessResult:
    """Resultado productivo de reconstrucción posterior a F01.5/F02.

    Adaptive Reconstruction permanece intacta. La capa post-filter trabaja sobre
    su salida y Call 2, cuando se habilite, consume exclusivamente el WallGraph
    filtrado/canónico.
    """

    adaptive: AdaptiveReconstructionRuntime
    post_filter: PostReconstructionFilterResult
    canonical: CanonicalWallGraphResult
    call2_mode: Call2Mode
    call2_required: bool
    call2_executed: bool
    call2_review: MultimodalWallReview | None
    call2_validation: Call2ValidationResult | None
    final_wall_graph: SingleLineWallGraph


class QuantiaSpatialV1ProcessEngine:
    """Orquestador real de reconstrucción V1.

    Flujo vigente:
        F01.5/F02 inputs
        -> AdaptiveReconstructionEngine
        -> PostReconstructionFilterEngine
        -> [Call 2 opcional]
        -> WallGraph final

    No contiene reglas por Casa Viri/Miguel H/Miguel V. Cada submotor decide por
    evidencia y deja trazabilidad de los módulos utilizados.
    """

    def __init__(
        self,
        *,
        adaptive_engine: AdaptiveReconstructionEngine | None = None,
        post_filter_engine: PostReconstructionFilterEngine | None = None,
        multimodal_reviewer: WallGraphMultimodalReviewer | None = None,
        correction_applier: WallGraphCorrectionApplier | None = None,
        call2_validator: Call2DeltaValidator | None = None,
        wallgraph_finalizer: CanonicalWallGraphFinalizer | None = None,
    ) -> None:
        self.adaptive_engine = adaptive_engine or AdaptiveReconstructionEngine()
        self.post_filter_engine = post_filter_engine or PostReconstructionFilterEngine()
        self.multimodal_reviewer = multimodal_reviewer
        self.correction_applier = correction_applier or WallGraphCorrectionApplier()
        self.call2_validator = call2_validator or Call2DeltaValidator()
        self.wallgraph_finalizer = wallgraph_finalizer or CanonicalWallGraphFinalizer()

    def run_level(
        self,
        *,
        level_view: LevelView,
        perimeter: EditablePerimeterModel,
        evidence: Sequence[RawEvidence],
        scale_profile: LevelScaleProfile,
        call2_mode: Call2Mode = "OFF",
    ) -> QuantiaSpatialV1ProcessResult:
        if call2_mode not in {"OFF", "AUTO", "FORCE"}:
            raise ValueError(f"call2_mode no soportado: {call2_mode}")

        adaptive = self.adaptive_engine.run(
            level_view=level_view,
            perimeter=perimeter,
            evidence=evidence,
            scale_profile=scale_profile,
        )
        post_filter = self.post_filter_engine.run(runtime=adaptive)
        canonical = self.wallgraph_finalizer.finalize(
            level_view=level_view,
            evidence=evidence,
            runtime=adaptive,
            post_filter=post_filter,
        )

        unresolved_ratio = (
            post_filter.diagnostics.unresolved_count
            / max(post_filter.diagnostics.input_wall_count, 1)
        )
        call2_required = bool(
            canonical.graph.route_plan.require_multimodal_review
            or unresolved_ratio >= 0.15
        )
        execute_call2 = call2_mode == "FORCE" or (call2_mode == "AUTO" and call2_required)

        review: MultimodalWallReview | None = None
        validation: Call2ValidationResult | None = None
        final_graph = canonical.graph
        if execute_call2:
            reviewer = self.multimodal_reviewer or WallGraphMultimodalReviewer()
            filter_context = self._build_call2_filter_context(post_filter)
            review = reviewer.review(
                level_view=level_view,
                graph=final_graph,
                filter_context=filter_context,
            )
            validation = self.call2_validator.validate(graph=final_graph, review=review)
            corrected_graph, final_sources = apply_with_lineage(
                applier=self.correction_applier,
                graph=final_graph,
                review=validation.accepted_review,
                source_mapping=canonical.evidence_bundle.final_source_mapping,
            )
            # Call 2 puede cambiar muros; las relaciones topológicas se vuelven
            # a calcular sobre esas paredes para no publicar referencias obsoletas.
            canonical = self.wallgraph_finalizer.finalize(
                level_view=level_view,
                evidence=evidence,
                runtime=adaptive,
                post_filter=post_filter,
                walls_override=corrected_graph.walls,
                source_mapping_override=final_sources,
                correction_review=validation.accepted_review.model_dump(mode="json"),
            )
            final_graph = canonical.graph

        return QuantiaSpatialV1ProcessResult(
            adaptive=adaptive,
            post_filter=post_filter,
            canonical=canonical,
            call2_mode=call2_mode,
            call2_required=call2_required,
            call2_executed=execute_call2,
            call2_review=review,
            call2_validation=validation,
            final_wall_graph=final_graph,
        )

    @staticmethod
    def _build_call2_filter_context(post_filter: PostReconstructionFilterResult) -> dict:
        def bucket(items):
            return [
                {
                    "source_wall_id": item.source_wall.id,
                    "start_px": [round(item.source_wall.start_px[0], 2), round(item.source_wall.start_px[1], 2)],
                    "end_px": [round(item.source_wall.end_px[0], 2), round(item.source_wall.end_px[1], 2)],
                    "family_hint": item.decision.architectural_family_hint,
                    "confidence": round(item.decision.confidence, 4),
                    "reasons": item.decision.reasons,
                }
                for item in items
            ]

        return {
            "version": post_filter.version,
            "diagnostics": post_filter.diagnostics.model_dump(mode="json"),
            "architectural_candidates": bucket(post_filter.architectural_elements),
            "excluded_graphic_candidates": bucket(post_filter.excluded_graphics),
            "unresolved_candidates": bucket(post_filter.unresolved),
            "filter_semantics": "evidence preserved; excluded from WallGraph does not mean deleted",
        }
