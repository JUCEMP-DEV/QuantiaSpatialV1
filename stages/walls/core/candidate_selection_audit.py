from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import LineString

from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel

from .candidate_models import CandidateRelation, WallCandidate, WallCandidateGraph
from .global_topology_solver import GlobalTopologySolution, GlobalTopologySolver
from .perimeter_adapter import line_is_inside_perimeter, perimeter_polygon


CandidateAuditCategory = Literal[
    "SELECTED",
    "OUTSIDE_F02",
    "CONFLICT_WITH_SELECTED",
    "DASHED_REFERENCE",
    "WEAK_EVIDENCE",
    "TOPOLOGY_REJECTED",
    "SEARCH_COMPETITION",
    "OTHER_REJECTED",
]


class CandidateSelectionAuditItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    generator: str
    selected: bool
    category: CandidateAuditCategory
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    angle_deg: float
    length_px: float
    thickness_px: float
    prior_score: float
    unary_score: float
    relation_bonus_to_selected: float
    topology_degree: int = Field(ge=0)
    positive_relation_count: int = Field(ge=0)
    marginal_topology: float | None = None
    marginal_global: float | None = None
    hard_conflicts_with_selected: list[str] = Field(default_factory=list)
    wall_identity_probability: float
    observation_confidence: float
    context_penalty: float
    context_state: str
    context_region_id: str | None = None
    context_region_type: str | None = None
    context_risk: float = 0.0
    structural_lineage_score: float = 0.0
    structural_lineage_strong: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
    source_names: list[str] = Field(default_factory=list)
    face_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    topology_if_added: dict[str, Any] | None = None
    reasons: list[str] = Field(default_factory=list)


class CandidateSelectionAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    level_view_id: str
    selected_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    items: list[CandidateSelectionAuditItem] = Field(default_factory=list)


class CandidateSelectionAudit:
    """Observabilidad de la decisión global sin intervenir en el solver.

    Recupera la clasificación diagnóstica que históricamente vivía en probes
    standardized. Usa exactamente el resultado ya elegido por el solver y solo
    explica por qué cada hipótesis quedó dentro o fuera de la solución.
    """

    VERSION = "CANDIDATE_SELECTION_AUDIT_V1"

    def build(
        self,
        *,
        graph: WallCandidateGraph,
        solution: GlobalTopologySolution,
        perimeter: EditablePerimeterModel,
        solver: GlobalTopologySolver,
    ) -> CandidateSelectionAuditResult:
        by_id = {candidate.id: candidate for candidate in graph.candidates}
        selected_ids = set(solution.selected_candidate_ids)
        selected_candidates = [
            by_id[candidate_id]
            for candidate_id in solution.selected_candidate_ids
            if candidate_id in by_id
        ]
        hard_conflicts, positive_relations = self._relation_maps(graph.relations)

        items = [
            self._classify(
                candidate=candidate,
                selected_ids=selected_ids,
                selected_candidates=selected_candidates,
                perimeter=perimeter,
                hard_conflicts=hard_conflicts,
                positive_relations=positive_relations,
                solver=solver,
                baseline_topology=solution.topology,
            )
            for candidate in graph.candidates
        ]
        counts = Counter(item.category for item in items)
        return CandidateSelectionAuditResult(
            version=self.VERSION,
            level_view_id=graph.level_view_id,
            selected_count=sum(item.selected for item in items),
            rejected_count=sum(not item.selected for item in items),
            category_counts=dict(sorted(counts.items())),
            items=items,
        )

    @staticmethod
    def _relation_maps(
        relations: list[CandidateRelation],
    ) -> tuple[dict[str, set[str]], dict[str, list[CandidateRelation]]]:
        hard: dict[str, set[str]] = defaultdict(set)
        positive: dict[str, list[CandidateRelation]] = defaultdict(list)
        for relation in relations:
            if relation.hard_conflict:
                hard[relation.a_id].add(relation.b_id)
                hard[relation.b_id].add(relation.a_id)
            elif relation.relation_type in {"JUNCTION", "CONTINUATION"}:
                positive[relation.a_id].append(relation)
                positive[relation.b_id].append(relation)
        return hard, positive

    @staticmethod
    def _candidate_line(candidate: WallCandidate) -> LineString:
        return LineString([
            (float(candidate.start.x), float(candidate.start.y)),
            (float(candidate.end.x), float(candidate.end.y)),
        ])

    @staticmethod
    def _evidence_reasons(candidate: WallCandidate) -> list[str]:
        evidence = candidate.evidence
        reasons: list[str] = []
        if evidence.dashed_penalty >= 0.5:
            reasons.append(f"dashed_penalty={evidence.dashed_penalty:.3f}")
        if evidence.axis_support >= 0.50:
            reasons.append(f"axis_support={evidence.axis_support:.3f}")
        if evidence.pair_overlap >= 0.60:
            reasons.append(f"pair_overlap={evidence.pair_overlap:.3f}")
        if evidence.region_support >= 0.50:
            reasons.append(f"region_support={evidence.region_support:.3f}")
        if evidence.source_consensus >= 0.50:
            reasons.append(f"source_consensus={evidence.source_consensus:.3f}")
        if evidence.vector_support >= 0.50:
            reasons.append(f"vector_support={evidence.vector_support:.3f}")
        if evidence.raster_line_support >= 0.50:
            reasons.append(f"raster_support={evidence.raster_line_support:.3f}")
        if evidence.semantic_support > 0.0:
            reasons.append(f"semantic_support={evidence.semantic_support:.3f}")
        return reasons

    def _classify(
        self,
        *,
        candidate: WallCandidate,
        selected_ids: set[str],
        selected_candidates: list[WallCandidate],
        perimeter: EditablePerimeterModel,
        hard_conflicts: dict[str, set[str]],
        positive_relations: dict[str, list[CandidateRelation]],
        solver: GlobalTopologySolver,
        baseline_topology,
    ) -> CandidateSelectionAuditItem:
        candidate_id = candidate.id
        positive = positive_relations.get(candidate_id, [])
        topology_degree = min(2, len(positive))
        unary = float(solver._unary(candidate, topology_degree))
        relation_bonus = float(
            solver._incremental_relation_bonus(
                candidate_id=candidate_id,
                selected=frozenset(selected_ids),
                relations=positive_relations,
            )
        )
        lineage = candidate.metadata.get("structural_lineage") or {}
        reasons = self._evidence_reasons(candidate)
        common = dict(
            candidate_id=candidate_id,
            generator=candidate.generator,
            selected=candidate_id in selected_ids,
            start_px=(float(candidate.start.x), float(candidate.start.y)),
            end_px=(float(candidate.end.x), float(candidate.end.y)),
            angle_deg=float(candidate.angle_deg),
            length_px=float(candidate.length_px),
            thickness_px=float(candidate.thickness_px),
            prior_score=float(candidate.prior_score),
            unary_score=unary,
            relation_bonus_to_selected=relation_bonus,
            topology_degree=topology_degree,
            positive_relation_count=len(positive),
            wall_identity_probability=float(
                solver._wall_identity_probability(candidate)
            ),
            observation_confidence=float(
                solver._observation_confidence(candidate)
            ),
            context_penalty=float(solver._context_penalty(candidate)),
            context_state=str(candidate.metadata.get("context_state", "ACTIVE")),
            context_region_id=candidate.metadata.get("context_region_id"),
            context_region_type=candidate.metadata.get("context_region_type"),
            context_risk=float(candidate.metadata.get("context_risk", 0.0) or 0.0),
            structural_lineage_score=float(lineage.get("score", 0.0) or 0.0),
            structural_lineage_strong=bool(lineage.get("strong", False)),
            evidence=candidate.evidence.model_dump(mode="json"),
            source_names=list(candidate.source_names),
            face_ids=list(candidate.face_ids),
            metadata=dict(candidate.metadata),
        )

        if candidate_id in selected_ids:
            return CandidateSelectionAuditItem(
                **common,
                category="SELECTED",
                reasons=reasons,
            )

        inside = line_is_inside_perimeter(
            line=self._candidate_line(candidate),
            polygon=perimeter_polygon(perimeter),
            tolerance_px=2.0,
        )
        if not inside:
            reasons.append("line_is_inside_perimeter=False")
            return CandidateSelectionAuditItem(
                **common,
                category="OUTSIDE_F02",
                reasons=reasons,
            )

        conflicts = sorted(hard_conflicts.get(candidate_id, set()) & selected_ids)
        if conflicts:
            reasons.append("hard_conflict_with_selected=" + ",".join(conflicts))
            return CandidateSelectionAuditItem(
                **common,
                category="CONFLICT_WITH_SELECTED",
                hard_conflicts_with_selected=conflicts,
                reasons=reasons,
            )

        topology_if_added = solver.topology.analyze(
            perimeter=perimeter,
            candidates=[*selected_candidates, candidate],
        )
        marginal_topology = float(
            topology_if_added.topology_score - baseline_topology.topology_score
        )
        marginal_global = float(unary + relation_bonus + marginal_topology)

        evidence = candidate.evidence
        reference_like = (
            evidence.dashed_penalty >= 0.5
            or (
                evidence.axis_support >= 0.60
                and evidence.region_support < 0.45
                and evidence.pair_overlap < 0.60
            )
        )
        if reference_like:
            category: CandidateAuditCategory = "DASHED_REFERENCE"
            reasons.append("reference_like_geometry")
        elif unary <= 0.0:
            category = "WEAK_EVIDENCE"
            reasons.append("unary_score<=0")
        elif marginal_topology < -0.02 and marginal_global <= 0.0:
            category = "TOPOLOGY_REJECTED"
            reasons.append("candidate_worsens_global_topology")
        elif marginal_global > 0.0:
            category = "SEARCH_COMPETITION"
            reasons.append(
                "positive_marginal_vs_final_solution_but_not_selected"
            )
        else:
            category = "OTHER_REJECTED"
            reasons.append("non_positive_marginal_without_primary_rule")

        return CandidateSelectionAuditItem(
            **common,
            category=category,
            marginal_topology=marginal_topology,
            marginal_global=marginal_global,
            topology_if_added=topology_if_added.model_dump(mode="json"),
            reasons=reasons,
        )
