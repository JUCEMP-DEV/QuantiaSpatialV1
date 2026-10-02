from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from app.quantia_spatialV1.adaptive_reconstruction.contracts import (
    AdaptiveReconstructionRuntime,
    PhysicalWallTrack,
    SingleLineWall,
)

from .contracts import (
    EvidenceBucketItem,
    FilterDecision,
    PatternEvidence,
    PostFilterDiagnostics,
    PostFilterPlan,
    PostReconstructionFilterResult,
    SourceContextSummary,
)
from .geometry_pattern_analyzer import PostFilterGeometryPatternAnalyzer
from .selector import PostFilterSelector
from .wall_hypothesis_canonicalizer import WallHypothesisCanonicalizer


class PostReconstructionFilterEngine:
    """Capa independiente posterior a Adaptive Reconstruction.

    Regla fija: filtrar = clasificar/excluir de una salida, nunca borrar evidencia.
    El runtime adaptativo recibido es inmutable desde el punto de vista logico.
    """

    ARCHITECTURAL_REGION_FAMILIES = {
        "STAIR_FLIGHT_REGION": "STAIR",
        "DOOR_REGION": "DOOR_OR_OPENING",
        "WINDOW_REGION": "WINDOW_OR_OPENING",
        "OPENING_REGION": "OPENING",
        "SYMBOL_FIXTURE_REGION": "ARCHITECTURAL_SYMBOL",
    }
    GRAPHIC_REGION_TYPES = {
        "FLOOR_FINISH_GRID_REGION",
        "HATCH_FILL_REGION",
        "AXIS_GRID_REGION",
        "DIMENSION_REGION",
        "TEXT_SYMBOL_REGION",
        "UNKNOWN_REPETITIVE_REGION",
        "REPETITIVE_PARALLEL_REGION",
        "REPETITIVE_GRID_REGION",
    }

    def __init__(
        self,
        *,
        selector: PostFilterSelector | None = None,
        canonicalizer: WallHypothesisCanonicalizer | None = None,
        geometry_analyzer: PostFilterGeometryPatternAnalyzer | None = None,
    ) -> None:
        self.selector = selector or PostFilterSelector()
        self.canonicalizer = canonicalizer or WallHypothesisCanonicalizer()
        self.geometry_analyzer = geometry_analyzer or PostFilterGeometryPatternAnalyzer()

    def run(self, *, runtime: AdaptiveReconstructionRuntime) -> PostReconstructionFilterResult:
        pattern_evidence = self.geometry_analyzer.analyze(
            walls=runtime.graph.walls,
            px_per_m=runtime.graph.px_per_m,
        )
        plan = self.selector.plan(runtime=runtime, pattern_evidence=pattern_evidence)
        source_context = self._source_context_index(runtime)
        pattern_index = self._pattern_index(pattern_evidence)
        tracks = {item.wall_id: item for item in runtime.wall_tracks}

        decisions: list[FilterDecision] = []
        included: list[SingleLineWall] = []
        architectural: list[EvidenceBucketItem] = []
        graphics: list[EvidenceBucketItem] = []
        unresolved: list[EvidenceBucketItem] = []

        for wall in runtime.graph.walls:
            track = tracks.get(wall.id)
            context = self._aggregate_wall_context(wall=wall, source_context=source_context)
            patterns = pattern_index.get(wall.id, [])
            decision = self._classify(
                wall=wall,
                track=track,
                context=context,
                patterns=patterns,
                plan=plan,
                px_per_m=runtime.graph.px_per_m,
            )
            decisions.append(decision)
            if decision.output_class == "WALL":
                included.append(wall)
            elif decision.output_class == "ARCHITECTURAL_ELEMENT":
                architectural.append(EvidenceBucketItem(source_wall=wall, decision=decision))
            elif decision.output_class == "EXCLUDED_GRAPHIC":
                graphics.append(EvidenceBucketItem(source_wall=wall, decision=decision))
            else:
                unresolved.append(EvidenceBucketItem(source_wall=wall, decision=decision))

        canonical, mapping = self.canonicalizer.canonicalize(
            walls=included,
            px_per_m=runtime.graph.px_per_m,
            enable_duplicates=plan.enabled("DUPLICATE_WALL_CANONICALIZER"),
            enable_near_coincident=plan.enabled("NEAR_COINCIDENT_CANONICALIZER"),
            enable_wall_faces=plan.enabled("WALL_FACE_CANONICALIZER"),
        )
        filtered_graph = runtime.graph.model_copy(update={"walls": canonical}, deep=True)
        collapsed = max(0, len(included) - len(canonical))
        return PostReconstructionFilterResult(
            level_view_id=runtime.graph.level_view_id,
            plan=plan,
            decisions=decisions,
            pattern_evidence=pattern_evidence,
            filtered_wall_graph=filtered_graph,
            architectural_elements=architectural,
            excluded_graphics=graphics,
            unresolved=unresolved,
            canonical_mapping=mapping,
            diagnostics=PostFilterDiagnostics(
                input_wall_count=len(runtime.graph.walls),
                wall_class_count=len(included),
                architectural_element_count=len(architectural),
                excluded_graphic_count=len(graphics),
                unresolved_count=len(unresolved),
                canonical_wall_count=len(canonical),
                collapsed_hypothesis_count=collapsed,
                pattern_group_count=len(pattern_evidence),
                repetitive_pattern_count=sum(
                    item.kind in {"REPETITIVE_PARALLEL", "ORTHOGONAL_GRID"}
                    for item in pattern_evidence
                ),
                angular_hub_count=sum(item.kind == "ANGULAR_HUB" for item in pattern_evidence),
                enabled_module_count=sum(item.enabled for item in plan.modules),
                near_coincident_pair_count=self.selector._count_near_coincident_pairs(
                    runtime.graph.walls,
                    runtime.graph.px_per_m,
                ),
            ),
            audit={
                "evidence_preserved": True,
                "adaptive_graph_unchanged": True,
                "filter_semantics": "classify/exclude from wallgraph; never delete source evidence",
                "canonical_semantics": "one published centerline per accepted physical wall hypothesis group",
                "selector_is_operational": True,
                "enabled_modules": [item.module for item in plan.modules if item.enabled],
                "adaptive_input_wall_ids": [wall.id for wall in runtime.graph.walls],
            },
        )

    def _classify(
        self,
        *,
        wall: SingleLineWall,
        track: PhysicalWallTrack | None,
        context: SourceContextSummary,
        patterns: Sequence[PatternEvidence],
        plan: PostFilterPlan,
        px_per_m: float,
    ) -> FilterDecision:
        reasons: list[str] = []
        region_types = set(context.region_types)
        architectural = [item for item in region_types if item in self.ARCHITECTURAL_REGION_FAMILIES]
        graphics = [item for item in region_types if item in self.GRAPHIC_REGION_TYPES]

        topology_strong = wall.role in {"PERIMETER", "DIVIDER"} and wall.confidence >= 0.60
        wall_band_strong = bool(
            track is not None
            and track.thickness_support >= 0.40
            and (track.region_support >= 0.40 or track.source_consensus >= 0.45)
        )
        wall_band_moderate = bool(
            track is not None
            and track.thickness_support >= 0.30
            and (track.region_support >= 0.25 or track.source_consensus >= 0.35)
        )
        structural_protected = bool(wall.f03_seed_protected or topology_strong)
        hard_structural = bool(topology_strong and wall_band_strong)

        # 1) Contexto explicito ya reconocido por las capas previas. El post-filter
        # usa esa evidencia, pero nunca altera la decision original aguas arriba.
        if architectural and plan.enabled("ARCHITECTURAL_ELEMENT_FILTER"):
            family = self.ARCHITECTURAL_REGION_FAMILIES[architectural[0]]
            context_hard = context.quarantined_source_count > 0 or (
                context.max_region_confidence >= 0.72 and context.max_context_risk >= 0.60
            )
            if hard_structural:
                reasons.extend([f"architectural_context={architectural[0]}", "hard_structural_conflict"])
                return self._decision(
                    wall, "UNRESOLVED", False, 0.58, reasons, context, patterns, family,
                    modules=["ARCHITECTURAL_ELEMENT_FILTER", "TOPOLOGY_GUARD"],
                )
            if context_hard:
                reasons.append(f"architectural_context={architectural[0]}")
                return self._decision(
                    wall, "ARCHITECTURAL_ELEMENT", False, 0.86, reasons, context, patterns, family,
                    modules=["ARCHITECTURAL_ELEMENT_FILTER"],
                )

        if graphics and plan.enabled("GRAPHIC_EVIDENCE_FILTER"):
            context_hard = context.quarantined_source_count > 0 or (
                context.max_region_confidence >= 0.72 and context.max_context_risk >= 0.62
            )
            if hard_structural:
                reasons.extend([f"graphic_context={graphics[0]}", "hard_structural_conflict"])
                return self._decision(
                    wall, "UNRESOLVED", False, 0.58, reasons, context, patterns,
                    modules=["GRAPHIC_EVIDENCE_FILTER", "TOPOLOGY_GUARD"],
                )
            if context_hard:
                reasons.append(f"graphic_context={graphics[0]}")
                return self._decision(
                    wall, "EXCLUDED_GRAPHIC", False, 0.88, reasons, context, patterns,
                    modules=["GRAPHIC_EVIDENCE_FILTER"],
                )

        # 2) Familias lineales arquitectonicas frecuentes que no deben entrar al
        # WallGraph pero tampoco eliminarse: barandales, pasamanos, bordes de losa,
        # cierres de opening y canceleria/aluminio piso-techo.
        if not hard_structural:
            special_family = self._infer_special_architectural_family(
                wall=wall,
                track=track,
                context=context,
                patterns=patterns,
                plan=plan,
                px_per_m=px_per_m,
                wall_band_strong=wall_band_strong,
                wall_band_moderate=wall_band_moderate,
                structural_protected=structural_protected,
            )
            if special_family is not None:
                family, confidence, special_reasons, modules = special_family
                reasons.extend(special_reasons)
                return self._decision(
                    wall,
                    "ARCHITECTURAL_ELEMENT",
                    False,
                    confidence,
                    reasons,
                    context,
                    patterns,
                    family,
                    modules=modules,
                )

        # 3) Motivos geometricos de la capa independiente. Solo los patrones de
        # alta certeza excluyen del WallGraph; conflictos estructurales quedan REVIEW.
        strong_patterns = [
            item
            for item in patterns
            if item.confidence >= 0.80
            and (
                (item.kind in {"REPETITIVE_PARALLEL", "ORTHOGONAL_GRID"} and plan.enabled("REPETITIVE_PATTERN_FILTER"))
                or (item.kind == "ANGULAR_HUB" and plan.enabled("ANGULAR_HUB_FILTER"))
            )
        ]
        if strong_patterns:
            strongest = max(strong_patterns, key=lambda item: item.confidence)
            pattern_module = "ANGULAR_HUB_FILTER" if strongest.kind == "ANGULAR_HUB" else "REPETITIVE_PATTERN_FILTER"
            if hard_structural:
                reasons.extend([f"pattern={strongest.kind}", "pattern_vs_structural_conflict"])
                return self._decision(
                    wall,
                    "UNRESOLVED",
                    False,
                    0.56,
                    reasons,
                    context,
                    patterns,
                    strongest.architectural_family_hint,
                    modules=[pattern_module, "TOPOLOGY_GUARD"],
                )
            if strongest.output_class_hint == "ARCHITECTURAL_ELEMENT":
                reasons.append(f"pattern={strongest.kind}")
                return self._decision(
                    wall,
                    "ARCHITECTURAL_ELEMENT",
                    False,
                    strongest.confidence,
                    reasons,
                    context,
                    patterns,
                    strongest.architectural_family_hint,
                    modules=[pattern_module],
                )
            if strongest.output_class_hint == "EXCLUDED_GRAPHIC":
                reasons.append(f"pattern={strongest.kind}")
                return self._decision(
                    wall,
                    "EXCLUDED_GRAPHIC",
                    False,
                    strongest.confidence,
                    reasons,
                    context,
                    patterns,
                    modules=[pattern_module],
                )

        # 4) Cuarentena sin familia semantica conocida: se conserva, pero se
        # excluye solo cuando no existe proteccion estructural suficiente.
        if context.quarantined_source_count > 0 and plan.enabled("ORPHAN_GRAPHIC_FILTER"):
            if structural_protected or wall_band_moderate:
                reasons.extend(["quarantined_source_present", "structural_conflict"])
                return self._decision(
                    wall, "UNRESOLVED", False, 0.50, reasons, context, patterns,
                    modules=["ORPHAN_GRAPHIC_FILTER", "TOPOLOGY_GUARD"],
                )
            reasons.append("quarantined_source_present")
            return self._decision(
                wall, "EXCLUDED_GRAPHIC", False, 0.70, reasons, context, patterns,
                modules=["ORPHAN_GRAPHIC_FILTER"],
            )

        # 5) Evidencia estructural. F03/topologia se conserva; el filtro no intenta
        # reabrir la reconstruccion que ya fue validada.
        if structural_protected and plan.enabled("TOPOLOGY_GUARD"):
            reasons.append("f03_or_topology_structural_protection")
            return self._decision(
                wall, "WALL", True, max(0.65, wall.confidence), reasons, context, patterns,
                modules=["TOPOLOGY_GUARD"],
            )

        if wall_band_strong:
            reasons.append("strong_physical_wall_band")
            return self._decision(wall, "WALL", True, max(0.62, wall.confidence), reasons, context, patterns)

        if wall.role == "REVIEW":
            if wall_band_moderate and wall.confidence >= 0.50:
                reasons.append("moderate_wall_band_but_topology_review")
                return self._decision(wall, "WALL", True, 0.58, reasons, context, patterns)
            if (
                plan.enabled("GRAPHIC_EVIDENCE_FILTER")
                and track is not None
                and track.dashed_penalty >= 0.50
                and track.region_support < 0.40
            ):
                reasons.append("dashed_reference_without_wall_band")
                return self._decision(
                    wall, "EXCLUDED_GRAPHIC", False, 0.72, reasons, context, patterns,
                    modules=["GRAPHIC_EVIDENCE_FILTER"],
                )
            reasons.append("insufficient_evidence_for_wall_or_non_wall")
            return self._decision(wall, "UNRESOLVED", False, 0.45, reasons, context, patterns)

        reasons.append("structural_role_without_conflicting_context")
        return self._decision(wall, "WALL", True, max(0.60, wall.confidence), reasons, context, patterns)

    def _infer_special_architectural_family(
        self,
        *,
        wall: SingleLineWall,
        track: PhysicalWallTrack | None,
        context: SourceContextSummary,
        patterns: Sequence[PatternEvidence],
        plan: PostFilterPlan,
        px_per_m: float,
        wall_band_strong: bool,
        wall_band_moderate: bool,
        structural_protected: bool,
    ) -> tuple[str, float, list[str], list[str]] | None:
        thickness_m = wall.thickness_px / max(px_per_m, 1e-6)
        length_m = self._segment_length_px(wall) / max(px_per_m, 1e-6)
        has_stair = "STAIR_FLIGHT_REGION" in set(context.region_types)
        has_opening = bool(set(context.region_types) & {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"})
        repetitive_pattern = any(
            item.kind in {"REPETITIVE_PARALLEL", "ORTHOGONAL_GRID"} and item.confidence >= 0.70
            for item in patterns
        )
        angular_pattern = any(item.kind == "ANGULAR_HUB" and item.confidence >= 0.78 for item in patterns)
        thin_line = thickness_m <= 0.12
        medium_line = thickness_m <= 0.18
        open_edge, both_open = self._open_edge_signature(wall)
        low_wall_band = not wall_band_strong and (track is None or track.thickness_support < 0.40)

        if has_stair and thin_line and length_m >= 0.40 and plan.enabled("STAIR_HANDRAIL_FILTER") and low_wall_band:
            return (
                "STAIR_HANDRAIL",
                0.82,
                ["stair_context", "thin_linear_element", "stair_edge_or_run_alignment"],
                ["STAIR_HANDRAIL_FILTER"],
            )

        if has_opening and repetitive_pattern and thin_line and plan.enabled("ALUMINUM_GLAZING_FILTER"):
            return (
                "FLOOR_TO_CEILING_GLAZING",
                0.84,
                ["opening_context", "repetitive_modulation", "thin_linear_partition"],
                ["ALUMINUM_GLAZING_FILTER"],
            )

        if has_opening and repetitive_pattern and plan.enabled("OPENING_CLOSURE_FILTER"):
            return (
                "OPENING_CLOSURE_CANDIDATE",
                0.80,
                ["opening_context", "repetitive_linear_closure_pattern"],
                ["OPENING_CLOSURE_FILTER"],
            )

        if open_edge and thin_line and (repetitive_pattern or length_m >= 0.70) and plan.enabled("RAILING_GUARDRAIL_FILTER") and low_wall_band:
            return (
                "RAILING_GUARDRAIL",
                0.80 if repetitive_pattern else 0.76,
                ["open_edge_signature", "thin_guard_line", "repetitive_posts_or_long_edge"],
                ["RAILING_GUARDRAIL_FILTER"],
            )

        if open_edge and medium_line and not repetitive_pattern and not both_open and not has_opening and plan.enabled("SLAB_EDGE_LEVEL_CHANGE_FILTER") and low_wall_band and not structural_protected:
            return (
                "SLAB_EDGE_LEVEL_CHANGE",
                0.74,
                ["open_edge_signature", "single_edge_without_wall_band"],
                ["SLAB_EDGE_LEVEL_CHANGE_FILTER"],
            )

        return None

    @staticmethod
    def _segment_length_px(wall: SingleLineWall) -> float:
        return ((wall.end_px[0] - wall.start_px[0]) ** 2 + (wall.end_px[1] - wall.start_px[1]) ** 2) ** 0.5

    @staticmethod
    def _open_edge_signature(wall: SingleLineWall) -> tuple[bool, bool]:
        topo = wall.topology_support or {}
        entries = topo.get("source_topology") if isinstance(topo, dict) else None
        if not entries:
            entries = [topo] if isinstance(topo, dict) else []
        one_side_open = 0
        both_sides_open = 0
        total = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            left = [int(v) for v in entry.get("left_labels", [])]
            right = [int(v) for v in entry.get("right_labels", [])]
            if not left and not right:
                continue
            total += 1
            left_open = left and sum(v == 0 for v in left) / len(left) >= 0.80
            right_open = right and sum(v == 0 for v in right) / len(right) >= 0.80
            if left_open and right_open:
                both_sides_open += 1
            elif left_open != right_open:
                one_side_open += 1
        if total == 0:
            return False, False
        return one_side_open / total >= 0.50, both_sides_open / total >= 0.50

    @staticmethod
    def _decision(
        wall: SingleLineWall,
        output_class: str,
        included: bool,
        confidence: float,
        reasons: Sequence[str],
        context: SourceContextSummary,
        patterns: Sequence[PatternEvidence],
        family: str | None = None,
        *,
        modules: Sequence[str] = (),
    ) -> FilterDecision:
        return FilterDecision(
            source_wall_id=wall.id,
            output_class=output_class,
            included_in_wallgraph=included,
            confidence=max(0.0, min(1.0, confidence)),
            reasons=list(reasons),
            source_candidate_ids=list(wall.source_candidate_ids),
            source_context=context,
            architectural_family_hint=family,
            pattern_ids=sorted(item.id for item in patterns),
            applied_modules=sorted(set(modules)),
            preserved=True,
        )

    @staticmethod
    def _pattern_index(patterns: Sequence[PatternEvidence]) -> dict[str, list[PatternEvidence]]:
        index: dict[str, list[PatternEvidence]] = defaultdict(list)
        for item in patterns:
            for wall_id in item.wall_ids:
                index[wall_id].append(item)
        return index

    @staticmethod
    def _source_context_index(runtime: AdaptiveReconstructionRuntime) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for decision in runtime.artifacts.get("context_decisions", []) or []:
            candidate_id = str(getattr(decision, "candidate_id", "") or "")
            if not candidate_id:
                continue
            index[candidate_id].append({
                "state": str(getattr(decision, "state", "ACTIVE") or "ACTIVE"),
                "region_type": getattr(decision, "region_type", None),
                "region_confidence": float(getattr(decision, "region_confidence", 0.0) or 0.0),
                "context_risk": float(getattr(decision, "context_risk", 0.0) or 0.0),
                "candidate_id": candidate_id,
            })

        candidates = []
        for key in ("discovered_candidates", "hybrid_candidates", "quarantined_candidates"):
            candidates.extend(runtime.artifacts.get(key, []) or [])
        for candidate in candidates:
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            state = metadata.get("adaptive_context_state") or metadata.get("context_state")
            region = metadata.get("adaptive_context_region_type") or metadata.get("context_region_type")
            record = {
                "state": str(state or "ACTIVE"),
                "region_type": region,
                "region_confidence": float(
                    metadata.get("adaptive_context_region_confidence")
                    or metadata.get("context_region_confidence")
                    or 0.0
                ),
                "context_risk": float(metadata.get("context_risk") or 0.0),
                "candidate_id": str(candidate.id),
            }
            ids = {str(candidate.id)}
            merged = metadata.get("consolidated_from")
            if isinstance(merged, list):
                ids.update(str(item) for item in merged if item)
            for candidate_id in ids:
                existing = index[candidate_id]
                if not any(
                    item.get("state") == record["state"]
                    and item.get("region_type") == record["region_type"]
                    for item in existing
                ):
                    existing.append(record)
        return index

    @staticmethod
    def _aggregate_wall_context(*, wall: SingleLineWall, source_context: dict[str, list[dict[str, Any]]]) -> SourceContextSummary:
        records: list[dict[str, Any]] = []
        for candidate_id in wall.source_candidate_ids:
            records.extend(source_context.get(str(candidate_id), []))
        states = sorted({str(item.get("state") or "ACTIVE") for item in records})
        region_types = sorted({str(item.get("region_type")) for item in records if item.get("region_type")})
        ids = sorted({str(item.get("candidate_id")) for item in records if item.get("candidate_id")})
        return SourceContextSummary(
            states=states,
            region_types=region_types,
            candidate_ids=ids,
            quarantined_source_count=sum(1 for item in records if str(item.get("state")) == "QUARANTINE"),
            review_source_count=sum(1 for item in records if str(item.get("state")) == "REVIEW"),
            protected_source_count=sum(1 for item in records if str(item.get("state")) == "PROTECTED"),
            max_region_confidence=max((float(item.get("region_confidence") or 0.0) for item in records), default=0.0),
            max_context_risk=max((float(item.get("context_risk") or 0.0) for item in records), default=0.0),
        )
