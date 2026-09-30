from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import EditablePerimeterModel

from .candidate_models import CandidateRelation, WallCandidate, WallCandidateGraph
from .perimeter_adapter import line_is_inside_perimeter, perimeter_polygon
from .topology_analyzer import RoomTopologyAnalyzer, TopologyAnalysis


class SolverDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_count: int = Field(ge=0)
    beam_width: int = Field(gt=0)
    evaluated_global_states: int = Field(ge=0)
    hard_conflict_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    perimeter_ineligible_count: int = Field(default=0, ge=0)
    unary_objective: float
    relational_objective: float
    topology_objective: float
    total_objective: float


class GlobalTopologySolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    topology: TopologyAnalysis
    diagnostics: SolverDiagnostics
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _BeamState:
    selected: frozenset[str]
    score: float


class GlobalTopologySolver:
    """Selecciona el conjunto global de hipótesis mediante beam search.

    Solver Evidence V1 usa la misma política para cualquier LevelView: separa
    identidad de muro, certeza de observación y riesgo contextual. QUARANTINE no
    llega al grafo; REVIEW permanece disponible con evidencia negativa explícita.
    Los conflictos F02 siguen siendo restricciones duras.
    """

    def __init__(self, *, beam_width: int = 96, global_eval_top_k: int = 96) -> None:
        self.beam_width = max(8, int(beam_width))
        # Certeza > velocidad: V1.2 evalúa topología sobre todo el beam por
        # defecto. Antes solo 28 estados locales llegaban al criterio global y
        # una solución más limpia podía quedar fuera antes de medir dangles/cuts.
        self.global_eval_top_k = max(4, min(self.beam_width, int(global_eval_top_k)))
        self.topology = RoomTopologyAnalyzer()

    def solve(
        self,
        *,
        graph: WallCandidateGraph,
        perimeter: EditablePerimeterModel,
    ) -> GlobalTopologySolution:
        by_id = {item.id: item for item in graph.candidates}
        polygon = perimeter_polygon(perimeter)
        eligible_candidates = [
            item
            for item in graph.candidates
            if line_is_inside_perimeter(
                line=self._candidate_line(item),
                polygon=polygon,
                tolerance_px=2.0,
            )
        ]
        eligible_ids = {item.id for item in eligible_candidates}
        ineligible_ids = set(by_id) - eligible_ids

        hard_conflicts: dict[str, set[str]] = {item.id: set() for item in eligible_candidates}
        positive_relations: dict[str, list[CandidateRelation]] = {
            item.id: [] for item in eligible_candidates
        }
        for relation in graph.relations:
            if relation.a_id not in eligible_ids or relation.b_id not in eligible_ids:
                continue
            if relation.hard_conflict:
                hard_conflicts.setdefault(relation.a_id, set()).add(relation.b_id)
                hard_conflicts.setdefault(relation.b_id, set()).add(relation.a_id)
            elif relation.relation_type in {"JUNCTION", "CONTINUATION"}:
                positive_relations.setdefault(relation.a_id, []).append(relation)
                positive_relations.setdefault(relation.b_id, []).append(relation)

        # V3: la densidad de una retícula no debe convertirse en evidencia casi
        # cuadrática. Para ordenar solo importa si existe soporte estructural en
        # 0, 1 o 2+ conexiones relevantes.
        topology_degree = {
            item.id: min(2, len(positive_relations.get(item.id, [])))
            for item in eligible_candidates
        }
        ordered = sorted(
            eligible_candidates,
            key=lambda item: (
                self._unary(item, topology_degree.get(item.id, 0)),
                item.length_px,
            ),
            reverse=True,
        )

        states = [_BeamState(selected=frozenset(), score=0.0)]
        for candidate in ordered:
            next_states: list[_BeamState] = []
            unary = self._unary(candidate, topology_degree.get(candidate.id, 0))
            for state in states:
                next_states.append(state)
                if any(
                    conflict in state.selected
                    for conflict in hard_conflicts.get(candidate.id, set())
                ):
                    continue
                relation_bonus = self._incremental_relation_bonus(
                    candidate_id=candidate.id,
                    selected=state.selected,
                    relations=positive_relations,
                )
                next_states.append(
                    _BeamState(
                        selected=state.selected | {candidate.id},
                        score=state.score + unary + relation_bonus,
                    )
                )

            next_states.sort(key=lambda item: item.score, reverse=True)
            deduped: list[_BeamState] = []
            seen: set[frozenset[str]] = set()
            for item in next_states:
                if item.selected in seen:
                    continue
                seen.add(item.selected)
                deduped.append(item)
                if len(deduped) >= self.beam_width:
                    break
            states = deduped

        finalists = states[: self.global_eval_top_k]
        if not any(not state.selected for state in finalists):
            finalists.append(_BeamState(selected=frozenset(), score=0.0))

        best = None
        best_topology = None
        best_total = -math.inf
        for state in finalists:
            selected_candidates = [by_id[item_id] for item_id in state.selected]
            topology = self.topology.analyze(
                perimeter=perimeter,
                candidates=selected_candidates,
            )
            total = state.score + topology.topology_score
            if total > best_total:
                best_total = total
                best = state
                best_topology = topology

        assert best is not None and best_topology is not None
        selected_ids = sorted(best.selected)
        unary_objective = sum(
            self._unary(by_id[item_id], topology_degree.get(item_id, 0))
            for item_id in selected_ids
        )
        relational_objective = self._selected_relation_objective(
            selected_ids,
            graph.relations,
        )

        warnings = []
        if ineligible_ids:
            warnings.append(
                f"{len(ineligible_ids)} candidatos fueron excluidos por violar el perímetro F02."
            )

        return GlobalTopologySolution(
            selected_candidate_ids=selected_ids,
            rejected_candidate_ids=sorted(set(by_id) - set(selected_ids)),
            topology=best_topology,
            diagnostics=SolverDiagnostics(
                candidate_count=len(graph.candidates),
                beam_width=self.beam_width,
                evaluated_global_states=len(finalists),
                hard_conflict_count=graph.diagnostics.hard_conflict_count,
                selected_count=len(selected_ids),
                perimeter_ineligible_count=len(ineligible_ids),
                unary_objective=float(unary_objective),
                relational_objective=float(relational_objective),
                topology_objective=float(best_topology.topology_score),
                total_objective=float(best_total),
            ),
            warnings=warnings,
        )

    @staticmethod
    def _candidate_line(candidate: WallCandidate):
        from shapely.geometry import LineString

        return LineString([
            (candidate.start.x, candidate.start.y),
            (candidate.end.x, candidate.end.y),
        ])

    @staticmethod
    def _pattern_penalty(candidate: WallCandidate) -> float:
        raw = candidate.metadata.get("repetitive_pattern_logodds_penalty", 0.0)
        try:
            penalty = max(0.0, min(3.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

        # Los ejes son una restricción arquitectónica externa a Candidate Discovery.
        # Pueden reducir el prior negativo, nunca anularlo por completo.
        axis_support = max(0.0, min(1.0, float(candidate.evidence.axis_support)))
        semantic_support = max(0.0, min(1.0, float(candidate.evidence.semantic_support)))
        independent_relief = min(0.45, 0.30 * axis_support + 0.25 * semantic_support)
        return penalty * (1.0 - independent_relief)

    @staticmethod
    def _wall_identity_probability(candidate: WallCandidate) -> float:
        """Probabilidad geométrica/arquitectónica de identidad de muro.

        Selection Contracts V1.2 mantiene la separación entre observación e
        identidad, pero además impide que axis_support convierta una banda
        físicamente implausible en muro. Un eje puede coincidir con un muro real,
        pero la coincidencia solo aporta identidad cuando el espesor métrico es
        razonablemente compatible con una banda arquitectónica.
        """
        evidence = candidate.evidence
        raw_metric = candidate.metadata.get("metric_thickness_support")
        try:
            metric_support = (
                1.0 if raw_metric is None else max(0.0, min(1.0, float(raw_metric)))
            )
        except (TypeError, ValueError):
            metric_support = 1.0
        effective_axis = float(evidence.axis_support) * (0.25 + 0.75 * metric_support)

        if candidate.generator == "DOUBLE_FACE":
            value = (
                0.31 * evidence.pair_overlap
                + 0.24 * evidence.thickness_support
                + 0.22 * evidence.region_support
                + 0.13 * evidence.perimeter_containment
                + 0.06 * effective_axis
                + 0.04 * evidence.semantic_support
                - 0.30 * evidence.dashed_penalty
            )
        else:
            value = (
                0.42 * evidence.region_support
                + 0.28 * evidence.thickness_support
                + 0.17 * evidence.perimeter_containment
                + 0.08 * effective_axis
                + 0.05 * evidence.semantic_support
                - 0.26 * evidence.dashed_penalty
            )
        return max(0.02, min(0.98, float(value)))

    @staticmethod
    def _observation_confidence(candidate: WallCandidate) -> float:
        """Certeza de observación, no voto semántico de muro.

        El prior A1.2 conserva información útil de calidad/procedencia. Aquí solo
        modula la magnitud de la evidencia de identidad: no puede cambiar por sí
        mismo una hipótesis de no-muro a muro.
        """
        observed = max(
            float(candidate.evidence.vector_support),
            float(candidate.evidence.raster_line_support),
        )
        return max(0.72, min(1.0, 0.72 + 0.18 * candidate.prior_score + 0.10 * observed))

    @staticmethod
    def _context_risk(candidate: WallCandidate) -> float:
        state = candidate.metadata.get("context_state")
        if state == "PROTECTED" or not candidate.metadata.get("context_region_id"):
            return 0.0
        raw = candidate.metadata.get("context_risk", 0.0)
        try:
            risk = max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

        # axis_support no alivia riesgo por sí solo: el eje gráfico también puede
        # coincidir con el candidato falso. Solo semántica localizada o linaje
        # estructural fuerte e independiente reducen parcialmente el riesgo.
        semantic = max(0.0, min(1.0, float(candidate.evidence.semantic_support)))
        lineage = candidate.metadata.get("structural_lineage") or {}
        lineage_relief = 0.10 if bool(lineage.get("strong")) else 0.0
        relief = min(0.22, 0.12 * semantic + lineage_relief)
        return risk * (1.0 - relief)

    @classmethod
    def _context_penalty(cls, candidate: WallCandidate) -> float:
        # Misma escala log-odds usada por _unary. REVIEW no se excluye: requiere
        # evidencia suficiente para vencer el riesgo contextual.
        return 1.55 * cls._context_risk(candidate)

    @staticmethod
    def _metric_thickness_penalty(candidate: WallCandidate) -> float:
        """Penalización física continua separando espesor y esbeltez.

        En V1.1 una meseta de plausibilidad todavía permitía que una pareja de
        líneas de baldosa de ~0.30 m o una banda gráfica de ~0.03 m obtuviera casi
        el mismo trato que un muro. V1.2 conserva ambas señales por separado: la
        longitud NO puede rescatar por sí sola un espesor físicamente pobre y un
        espesor plausible tampoco legitima un fragmento extremadamente corto.
        """
        raw_thickness = candidate.metadata.get("metric_thickness_support")
        raw_slenderness = candidate.metadata.get("metric_slenderness_support")
        if raw_thickness is None and raw_slenderness is None:
            return 0.0
        try:
            thickness_support = (
                1.0
                if raw_thickness is None
                else max(0.0, min(1.0, float(raw_thickness)))
            )
        except (TypeError, ValueError):
            thickness_support = 1.0
        try:
            slenderness_support = (
                1.0
                if raw_slenderness is None
                else max(0.0, min(1.0, float(raw_slenderness)))
            )
        except (TypeError, ValueError):
            slenderness_support = 1.0

        penalty = (
            1.35 * (1.0 - thickness_support)
            + 0.35 * (1.0 - slenderness_support)
        )
        return min(1.65, max(0.0, penalty))

    @staticmethod
    def _reference_penalty(candidate: WallCandidate) -> float:
        """Penaliza referencias gráficas sin convertir AXIS en veto de muros."""
        dashed = max(0.0, min(1.0, float(candidate.evidence.dashed_penalty)))
        if dashed <= 0.0:
            return 0.0
        context_type = str(candidate.metadata.get("context_region_type") or "")
        context_bonus = 0.30 if context_type in {"AXIS_GRID_REGION", "DIMENSION_REGION"} else 0.0
        return min(0.85, 0.50 * dashed + context_bonus * dashed)

    @classmethod
    def _unary(cls, candidate: WallCandidate, topology_degree: int = 0) -> float:
        # La identidad de muro se calcula con geometría/arquitectura. La certeza de
        # observación solo escala la magnitud; vector/raster no deciden la clase.
        p = cls._wall_identity_probability(candidate)
        reference = 0.66
        identity_log_odds = (
            math.log(p / (1.0 - p))
            - math.log(reference / (1.0 - reference))
        )
        confidence = cls._observation_confidence(candidate)
        length_factor = min(1.30, max(0.78, math.sqrt(candidate.length_px / 140.0)))
        value = identity_log_odds * confidence * length_factor

        # La conectividad sigue siendo apoyo/penalización suave. No se modifica en
        # esta ronda para aislar la causa: scoring de evidencia + contexto REVIEW.
        if topology_degree <= 0:
            value -= 0.55
        elif topology_degree == 1:
            value -= 0.16
        else:
            value += 0.04

        value -= cls._context_penalty(candidate)
        value -= cls._pattern_penalty(candidate)
        value -= cls._reference_penalty(candidate)
        value -= cls._metric_thickness_penalty(candidate)
        return value

    @staticmethod
    def _incremental_relation_bonus(
        *,
        candidate_id: str,
        selected: frozenset[str],
        relations: dict[str, list[CandidateRelation]],
    ) -> float:
        """Máximo dos apoyos estructurales por candidato.

        La versión anterior sumaba cada JUNCTION; una retícula con muchas líneas
        podía ganar recompensa por densidad. Aquí todas las conexiones activas se
        convierten en apoyos y solo cuentan las dos más fuertes.
        """
        supports: list[float] = []
        for relation in relations.get(candidate_id, []):
            other = relation.b_id if relation.a_id == candidate_id else relation.a_id
            if other not in selected:
                continue
            if relation.relation_type == "JUNCTION":
                supports.append(0.16 * relation.strength)
            elif relation.relation_type == "CONTINUATION":
                supports.append(0.08 * relation.strength)
        supports.sort(reverse=True)
        return float(sum(supports[:2]))

    @staticmethod
    def _selected_relation_objective(
        selected_ids: list[str],
        relations: list[CandidateRelation],
    ) -> float:
        """Diagnóstico consistente con la regla de máximo dos apoyos por muro."""
        selected = set(selected_ids)
        support_by_candidate: dict[str, list[float]] = {
            candidate_id: [] for candidate_id in selected
        }
        for relation in relations:
            if relation.a_id not in selected or relation.b_id not in selected:
                continue
            if relation.relation_type == "JUNCTION":
                value = 0.16 * relation.strength
            elif relation.relation_type == "CONTINUATION":
                value = 0.08 * relation.strength
            else:
                continue
            support_by_candidate[relation.a_id].append(value)
            support_by_candidate[relation.b_id].append(value)

        # Cada relación alimentó los dos candidatos; dividir entre dos evita doble
        # conteo en el diagnóstico agregado.
        total = sum(
            sum(sorted(values, reverse=True)[:2])
            for values in support_by_candidate.values()
        )
        return float(total / 2.0)
