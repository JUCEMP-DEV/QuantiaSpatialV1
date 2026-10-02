from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from app.quantia_spatialV1.stages.walls.adaptive import (
    AdaptiveReconstructionEngine,
    AdaptiveReconstructionRuntime,
    Call2DeltaValidator,
    Call2ValidationResult,
    MultimodalWallReview,
    SingleLineWallGraph,
    WallGraphCorrectionApplier,
    WallGraphMultimodalReviewer,
)
from app.quantia_spatialV1.stages.walls.canonical import (
    CanonicalWallGraphFinalizer,
    CanonicalWallGraphResult,
)
from app.quantia_spatialV1.stages.walls.canonical.correction_lineage import apply_with_lineage
from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel
from app.quantia_spatialV1.stages.walls.postfilter import (
    PostReconstructionFilterEngine,
    PostReconstructionFilterResult,
)
from app.quantia_spatialV1.core.scale.level_scale_normalizer import LevelScaleProfile
from app.quantia_spatialV1.stages.spaces import (
    SpaceClosureEngine,
    SpaceClosureResult,
    SpaceConstraintResult,
    SpaceConstraintValidator,
)


Call2Mode = Literal["OFF", "AUTO", "FORCE"]


@dataclass
class QuantiaSpatialV1ProcessResult:
    """Resultado productivo de reconstrucción posterior a F01.5/F02."""

    adaptive: AdaptiveReconstructionRuntime
    post_filter: PostReconstructionFilterResult
    canonical: CanonicalWallGraphResult
    call2_mode: Call2Mode
    call2_required: bool
    call2_executed: bool
    call2_review: MultimodalWallReview | None
    call2_validation: Call2ValidationResult | None
    final_wall_graph: SingleLineWallGraph
    space_closure: SpaceClosureResult
    space_constraints: SpaceConstraintResult


class QuantiaSpatialV1ProcessEngine:
    """Orquesta walls -> Call 2 -> espacios sin reglas por caso."""

    def __init__(
        self,
        *,
        adaptive_engine: AdaptiveReconstructionEngine | None = None,
        post_filter_engine: PostReconstructionFilterEngine | None = None,
        multimodal_reviewer: WallGraphMultimodalReviewer | None = None,
        correction_applier: WallGraphCorrectionApplier | None = None,
        call2_validator: Call2DeltaValidator | None = None,
        wallgraph_finalizer: CanonicalWallGraphFinalizer | None = None,
        space_closure_engine: SpaceClosureEngine | None = None,
        space_constraint_validator: SpaceConstraintValidator | None = None,
    ) -> None:
        self.adaptive_engine = adaptive_engine or AdaptiveReconstructionEngine()
        self.post_filter_engine = post_filter_engine or PostReconstructionFilterEngine()
        self.multimodal_reviewer = multimodal_reviewer
        self.correction_applier = correction_applier or WallGraphCorrectionApplier()
        self.call2_validator = call2_validator or Call2DeltaValidator()
        self.wallgraph_finalizer = wallgraph_finalizer or CanonicalWallGraphFinalizer()
        self.space_closure_engine = space_closure_engine or SpaceClosureEngine()
        self.space_constraint_validator = (
            space_constraint_validator
            or SpaceConstraintValidator()
        )

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
            raise ValueError(
                f"call2_mode no soportado: {call2_mode}"
            )

        adaptive = self.adaptive_engine.run(
            level_view=level_view,
            perimeter=perimeter,
            evidence=evidence,
            scale_profile=scale_profile,
        )
        post_filter = self.post_filter_engine.run(
            runtime=adaptive
        )
        canonical = self.wallgraph_finalizer.finalize(
            level_view=level_view,
            evidence=evidence,
            runtime=adaptive,
            post_filter=post_filter,
        )

        unresolved_ratio = (
            post_filter.diagnostics.unresolved_count
            / max(
                post_filter.diagnostics.input_wall_count,
                1,
            )
        )
        call2_required = bool(
            canonical.graph.route_plan.require_multimodal_review
            or unresolved_ratio >= 0.15
        )
        execute_call2 = (
            call2_mode == "FORCE"
            or (
                call2_mode == "AUTO"
                and call2_required
            )
        )

        review: MultimodalWallReview | None = None
        validation: Call2ValidationResult | None = None
        pre_call2_graph = canonical.graph
        final_graph = canonical.graph

        if execute_call2:
            reviewer = (
                self.multimodal_reviewer
                or WallGraphMultimodalReviewer()
            )
            review = reviewer.review(
                level_view=level_view,
                graph=final_graph,
                filter_context=self._build_call2_filter_context(
                    post_filter
                ),
            )
            validation = self.call2_validator.validate(
                graph=final_graph,
                review=review,
            )
            corrected_graph, final_sources = apply_with_lineage(
                applier=self.correction_applier,
                graph=final_graph,
                review=validation.accepted_review,
                source_mapping=(
                    canonical.evidence_bundle
                    .final_source_mapping
                ),
            )
            canonical = self.wallgraph_finalizer.finalize(
                level_view=level_view,
                evidence=evidence,
                runtime=adaptive,
                post_filter=post_filter,
                walls_override=corrected_graph.walls,
                source_mapping_override=final_sources,
                correction_review=(
                    validation.accepted_review
                    .model_dump(mode="json")
                ),
            )
            final_graph = canonical.graph

        space_closure = self.space_closure_engine.run(
            graph=final_graph,
            gap_decisions=(
                validation.accepted_review.gap_decisions
                if validation is not None
                else None
            ),
        )
        space_constraints = (
            self.space_constraint_validator.validate(
                closure=space_closure,
                evidence=evidence,
                final_graph=final_graph,
                input_graph=(
                    pre_call2_graph
                    if execute_call2
                    else None
                ),
                call2_review=(
                    validation.accepted_review
                    if validation is not None
                    else None
                ),
            )
        )

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
            space_closure=space_closure,
            space_constraints=space_constraints,
        )

    @staticmethod
    def _build_call2_filter_context(
        post_filter: PostReconstructionFilterResult,
    ) -> dict:
        def bucket(items):
            return [
                {
                    "source_wall_id": item.source_wall.id,
                    "start_px": [
                        round(
                            item.source_wall.start_px[0],
                            2,
                        ),
                        round(
                            item.source_wall.start_px[1],
                            2,
                        ),
                    ],
                    "end_px": [
                        round(
                            item.source_wall.end_px[0],
                            2,
                        ),
                        round(
                            item.source_wall.end_px[1],
                            2,
                        ),
                    ],
                    "family_hint": (
                        item.decision
                        .architectural_family_hint
                    ),
                    "confidence": round(
                        item.decision.confidence,
                        4,
                    ),
                    "reasons": item.decision.reasons,
                }
                for item in items
            ]

        return {
            "version": post_filter.version,
            "diagnostics": (
                post_filter.diagnostics
                .model_dump(mode="json")
            ),
            "architectural_candidates": bucket(
                post_filter.architectural_elements
            ),
            "excluded_graphic_candidates": bucket(
                post_filter.excluded_graphics
            ),
            "unresolved_candidates": bucket(
                post_filter.unresolved
            ),
            "filter_semantics": (
                "evidence preserved; excluded from "
                "WallGraph does not mean deleted"
            ),
        }
