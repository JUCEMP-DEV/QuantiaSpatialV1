from __future__ import annotations

import hashlib
import math
from typing import Sequence

from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.core.models.parametric import QuantiaParametricModel
from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel
from app.quantia_spatialV1.stages.walls.core.context_models import ContextRegion
from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingModel, DrawingPoint
from app.quantia_spatialV1.core.scale.level_scale_normalizer import LevelScaleProfile
from app.quantia_spatialV1.stages.walls.core.perimeter_adapter import perimeter_polygon

from .contracts import (
    ArchitecturalElementsDiagnostics,
    ArchitecturalElementsResult,
    ElementSemanticResolution,
    OpeningElementProposal,
)
from .feature_builder import ElementFeatureBuilder
from .host_wall_matcher import HostWallMatcher
from .proposal_generator import OpeningHypothesisGenerator
from .rag import ArchitecturalRAGIndex
from .reconciler import OpeningReconciler
from .resolver import ElementSemanticResolver
from .visual_context import ElementVisualContextBuilder


class ArchitecturalElementsMiniEngine:
    """Mini-motor V2 para DOOR / WINDOW / NOT_OPENING.

    Principio de aislamiento:
    - Spatial conserva autoridad exclusiva sobre muros.
    - este motor interpreta elementos y propone host_wall;
    - el reconciliador publica openings sin alterar geometría de muros.

    V2 descubre openings de forma independiente desde DrawingModel + muros finales.
    No consume ElementContextDetector, CandidateContextGate ni regiones F03.
    """

    VERSION = "ARCHITECTURAL_ELEMENTS_MINI_ENGINE_V2"

    def __init__(
        self,
        *,
        rag_index: ArchitecturalRAGIndex | None = None,
        semantic_resolver: ElementSemanticResolver | None = None,
    ) -> None:
        self.region_provider = OpeningHypothesisGenerator()
        self.wall_matcher = HostWallMatcher()
        self.feature_builder = ElementFeatureBuilder()
        self.rag = rag_index or ArchitecturalRAGIndex.default()
        self.semantic_resolver = semantic_resolver
        self.visual_context = ElementVisualContextBuilder()
        self.reconciler = OpeningReconciler()

    def run(
        self,
        *,
        level_view: LevelView,
        drawing: DrawingModel,
        spatial_model: QuantiaParametricModel,
        perimeter: EditablePerimeterModel | None = None,
        scale_profile: LevelScaleProfile | None = None,
        regions: Sequence[ContextRegion] | None = None,
        publish: bool = True,
    ) -> tuple[ArchitecturalElementsResult, QuantiaParametricModel]:
        if drawing.level_view_id != level_view.id:
            raise ValueError("DrawingModel pertenece a otro LevelView.")

        level = next(
            (
                item for item in spatial_model.levels
                if item.source_level_view_id == level_view.id or item.id == level_view.id
            ),
            None,
        )
        if level is None:
            raise ValueError("Spatial model no contiene el LevelView solicitado.")

        if regions is None:
            polygon = perimeter_polygon(perimeter) if perimeter is not None else None
            all_regions = self.region_provider.generate(
                drawing=drawing,
                walls=level.walls,
                scale_profile=scale_profile,
                architectural_polygon=polygon,
            )
        else:
            all_regions = list(regions)

        opening_regions = [
            region for region in all_regions
            if region.region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"}
        ]
        proposals: list[OpeningElementProposal] = []
        for region in opening_regions:
            host = self.wall_matcher.match(region=region, drawing=drawing, walls=level.walls)
            features = self.feature_builder.build(region=region, host_wall=host)
            hits = self.rag.retrieve(features, top_k=4)
            semantic_resolution = self._resolve_semantics(
                level_view=level_view,
                region=region,
                features=features,
                rag_hits=hits,
            )
            predicted_class, decision, confidence, reason = self._decide(
                region=region,
                features=features,
                rag_hits=hits,
                semantic_resolution=semantic_resolution,
            )
            proposal = self._proposal(
                level_view_id=level_view.id,
                region=region,
                drawing=drawing,
                predicted_class=predicted_class,
                decision=decision,
                confidence=confidence,
                reason=reason,
                host=host,
                features=features,
                rag_hits=hits,
                semantic_resolution=semantic_resolution,
            )
            proposals.append(proposal)

        if publish:
            merged, published_count = self.reconciler.merge(
                model=spatial_model,
                level_view_id=level_view.id,
                proposals=proposals,
            )
        else:
            merged, published_count = spatial_model.model_copy(deep=True), 0

        warnings: list[str] = []
        if not opening_regions:
            warnings.append("NO_OPENING_REGIONS")
        review_count = sum(1 for item in proposals if item.decision == "REVIEW")
        if review_count:
            warnings.append(f"OPENING_REVIEW_PENDING:{review_count}")

        accepted = [item for item in proposals if item.decision == "ACCEPTED"]
        source_counts: dict[str, int] = {}
        for region in opening_regions:
            source = str(region.metadata.get("source") or "UNKNOWN")
            source_counts[source] = source_counts.get(source, 0) + 1
        independent_candidate_count = sum(
            count for source, count in source_counts.items()
            if source.startswith("INDEPENDENT_")
        )
        semantic_candidate_count = sum(
            count for source, count in source_counts.items()
            if "SEMANTIC" in source or "GEMINI" in source
        )
        semantic_resolution_count = sum(1 for item in proposals if item.semantic_resolution is not None)
        result = ArchitecturalElementsResult(
            level_view_id=level_view.id,
            proposals=proposals,
            diagnostics=ArchitecturalElementsDiagnostics(
                source_region_count=len(all_regions),
                opening_region_count=len(opening_regions),
                proposal_count=len(proposals),
                accepted_count=len(accepted),
                review_count=review_count,
                rejected_count=sum(1 for item in proposals if item.decision == "REJECTED"),
                published_opening_count=published_count,
                door_count=sum(1 for item in accepted if item.predicted_class == "DOOR"),
                window_count=sum(1 for item in accepted if item.predicted_class == "WINDOW"),
                independent_candidate_count=independent_candidate_count,
                semantic_candidate_count=semantic_candidate_count,
                semantic_resolution_count=semantic_resolution_count,
                source_counts=source_counts,
            ),
            warnings=warnings,
        )
        return result, merged

    def _resolve_semantics(self, *, level_view, region, features, rag_hits):
        if self.semantic_resolver is None:
            return None
        image_png = self.visual_context.crop_png(level_view=level_view, bbox=region.bbox)
        return self.semantic_resolver.resolve(
            image_png=image_png,
            features=features,
            rag_hits=rag_hits,
        )

    def _decide(self, *, region, features, rag_hits, semantic_resolution):
        scores = {"DOOR": 0.0, "WINDOW": 0.0, "NOT_OPENING": 0.0}
        for index, hit in enumerate(rag_hits):
            weight = 1.0 / (1.0 + 0.35 * index)
            scores[hit.label] += hit.similarity * weight
        normalizer = sum(scores.values()) or 1.0
        rag_prob = {key: value / normalizer for key, value in scores.items()}
        rag_label = max(rag_prob, key=rag_prob.get)

        expected = features.expected_class
        predicted = expected if expected in {"DOOR", "WINDOW"} else rag_label
        semantic_support = 0.0
        if semantic_resolution is not None:
            semantic_support = semantic_resolution.confidence
            if semantic_resolution.confidence >= 0.62:
                predicted = semantic_resolution.label

        negative_conflict = (
            rag_label == "NOT_OPENING" and rag_prob["NOT_OPENING"] >= 0.45
        ) or (
            predicted == "WINDOW" and features.wall_through_support >= 0.72
        )

        class_rag_support = rag_prob.get(predicted, 0.0)
        geometry_support = max(
            features.opening_verified,
            0.55 * features.door_arc_support + 0.45 * features.door_leaf_support,
            0.60 * features.window_frame_support + 0.40 * features.host_local_support,
        )
        confidence = max(
            0.0,
            min(
                0.99,
                0.24 * features.region_confidence
                + 0.22 * geometry_support
                + 0.24 * features.host_wall_match
                + 0.12 * features.semantic_support
                + 0.10 * class_rag_support
                + 0.08 * semantic_support,
            ),
        )

        if predicted == "NOT_OPENING":
            return "NOT_OPENING", "REJECTED", max(confidence, rag_prob["NOT_OPENING"]), "RAG_NOT_OPENING"
        if negative_conflict and features.opening_verified < 1.0:
            return predicted, "REJECTED", confidence, "OPENING_CONFLICT"
        if features.host_wall_match < 0.34:
            return predicted, "REVIEW", confidence, "HOST_WALL_UNRESOLVED"
        if (
            features.opening_verified >= 1.0
            and confidence >= 0.50
            and not negative_conflict
        ):
            return predicted, "ACCEPTED", confidence, "VERIFIED_OPENING_WITH_HOST"
        if (
            geometry_support >= 0.62
            and features.host_wall_match >= 0.50
            and class_rag_support >= 0.20
            and not negative_conflict
        ):
            return predicted, "ACCEPTED", confidence, "GEOMETRY_RAG_HOST_CONSENSUS"
        return predicted, "REVIEW", confidence, "OPENING_EVIDENCE_INCOMPLETE"

    def _proposal(
        self,
        *,
        level_view_id,
        region,
        drawing,
        predicted_class,
        decision,
        confidence,
        reason,
        host,
        features,
        rag_hits,
        semantic_resolution,
    ) -> OpeningElementProposal:
        cx = (region.bbox.x_min + region.bbox.x_max) * 0.5
        cy = (region.bbox.y_min + region.bbox.y_max) * 0.5
        orientation = float(host.metadata.get("wall_angle_deg", 0.0)) % 180.0
        width = self._estimate_width(region=region, predicted_class=predicted_class)
        evidence_ids = sorted(
            {
                *region.member_line_ids,
                *(region.metadata.get("curve_evidence_ids") or []),
                *(semantic_resolution.evidence_ids if semantic_resolution else []),
            }
        )
        digest = hashlib.sha1(
            f"{level_view_id}|{region.id}|{predicted_class}|{host.wall_id}".encode()
        ).hexdigest()[:14]
        return OpeningElementProposal(
            id=f"{level_view_id}__ELEMENT__{digest}",
            level_view_id=level_view_id,
            source_region_id=region.id,
            source_bbox=region.bbox,
            predicted_class=predicted_class,
            decision=decision,
            confidence=confidence,
            center_px=DrawingPoint(x=cx, y=cy),
            orientation_deg=orientation,
            estimated_width_px=width,
            host_wall=host,
            features=features,
            rag_hits=rag_hits,
            semantic_resolution=semantic_resolution,
            evidence_ids=evidence_ids,
            metadata={
                "engine_version": self.VERSION,
                "decision_reason": reason,
                "source_detector": region.metadata.get("detector"),
                "source_region_type": region.region_type,
                "walls_modified": False,
            },
        )

    @staticmethod
    def _estimate_width(*, region: ContextRegion, predicted_class: str) -> float:
        meta = region.metadata
        if predicted_class == "DOOR":
            radius = meta.get("estimated_swing_radius_px")
            if radius is not None:
                try:
                    return max(2.0, float(radius))
                except (TypeError, ValueError):
                    pass
        width = region.bbox.x_max - region.bbox.x_min
        height = region.bbox.y_max - region.bbox.y_min
        # Para ventanas, el eje largo corresponde mejor al ancho de opening. En
        # puertas sin radio disponible también es una aproximación conservadora.
        return max(2.0, float(max(width, height)))
