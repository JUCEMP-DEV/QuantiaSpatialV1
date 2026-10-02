from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from shapely.geometry import LineString, Point, box

from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel
from app.quantia_spatialV1.stages.walls.core.reconstruction_pipeline import ProposalCReconstructionPipeline
from app.quantia_spatialV1.stages.walls.core.candidate_models import WallCandidate
from app.quantia_spatialV1.core.scale.level_scale_normalizer import LevelScaleProfile
from app.quantia_spatialV1.stages.walls.core.wall_track_consolidator import WallTrackConsolidator

from .contracts import (
    AdaptiveReconstructionRuntime,
    ClassifiedWall,
    LogicalGap,
    PhysicalWallTrack,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
    SpaceTopology,
    VirtualBridge,
)
from .decision_engine import AdaptiveModuleDecisionEngine
from .decision_trace import AdaptiveDecisionTraceRecorder


class AdaptiveReconstructionEngine:
    """Reconstrucción híbrida modular de un LevelView.

    Responsabilidades:
      - usar F03 como seed estructural, no como verdad exhaustiva;
      - recuperar geometría desde DrawingModel/F01.5 sin depender de F02;
      - aplicar Context Gate como aislamiento negativo;
      - canonicalizar a una sola línea por muro;
      - construir continuidad lógica y topología de espacios;
      - decidir si el resultado necesita Call 2 multimodal.

    Ninguna decisión depende del nombre del proyecto/caso.
    """

    def __init__(
        self,
        *,
        pipeline: ProposalCReconstructionPipeline | None = None,
        decision_engine: AdaptiveModuleDecisionEngine | None = None,
        trace_recorder: AdaptiveDecisionTraceRecorder | None = None,
    ) -> None:
        self.pipeline = pipeline or ProposalCReconstructionPipeline()
        self.decision_engine = decision_engine or AdaptiveModuleDecisionEngine()
        self.trace_recorder = trace_recorder or AdaptiveDecisionTraceRecorder()

    def run(
        self,
        *,
        level_view: LevelView,
        perimeter: EditablePerimeterModel,
        evidence: Sequence[RawEvidence],
        scale_profile: LevelScaleProfile,
    ) -> AdaptiveReconstructionRuntime:
        if scale_profile.local_m_per_px is None or scale_profile.local_m_per_px <= 0.0:
            raise ValueError("AdaptiveReconstructionEngine requiere escala RESOLVED.")
        px_per_m = 1.0 / float(scale_profile.local_m_per_px)

        # 1) F03 actual: produce seed estructural y contexto ya probado.
        full = self.pipeline.run(
            level_view=level_view,
            perimeter=perimeter,
            evidence=list(evidence),
            scale_profile=scale_profile,
        )
        graph_by_id = {candidate.id: candidate for candidate in full.candidate_graph.candidates}
        seeds = [
            graph_by_id[candidate_id]
            for candidate_id in full.solution.selected_candidate_ids
            if candidate_id in graph_by_id
        ]

        # 2) Candidate Discovery independiente de F02.
        discovered = self._discover_without_f02_perimeter(full=full, scale_profile=scale_profile)
        regions = [
            region
            for region in full.context_gate.regions
            if region.region_type != "WALL_PROTECTED_REGION"
        ]
        gate = self.pipeline.context_gate.apply(candidates=discovered, regions=regions)

        pre_plan = self.decision_engine.plan_pre_topology(
            seed_count=len(seeds),
            discovered_count=len(discovered),
            quarantined_count=len(gate.quarantined_candidates),
        )

        # 3) Fusión adaptable: Context Gate elimina competencia, F03 puede rescatar
        # geometría que la clasificación contextual haya encapsulado.
        hybrid_candidates = self._merge_seed_and_context_cleaned(
            discovered=discovered,
            gate=gate,
            seeds=seeds,
            px_per_m=px_per_m,
            width=full.drawing_model.width_px,
            height=full.drawing_model.height_px,
            recovery_strength=pre_plan.recovery_strength,
        )
        if not hybrid_candidates:
            raise RuntimeError("Adaptive reconstruction no produjo candidatos híbridos.")

        physical = self._physical_walls(hybrid_candidates)
        seed_wall_ids = self._physical_seed_ids(physical, seeds=seeds, px_per_m=px_per_m)

        bridges_all = self._strict_virtual_bridges(
            physical,
            seed_ids=seed_wall_ids,
            px_per_m=px_per_m,
        )
        components = self._component_membership(physical, bridges_all, px_per_m=px_per_m)
        selected, component_report = self._select_hybrid_network(
            physical,
            components,
            seed_ids=seed_wall_ids,
            px_per_m=px_per_m,
            recovery_strength=pre_plan.recovery_strength,
        )
        if not selected:
            raise RuntimeError("Adaptive reconstruction dejó vacía la red de muros.")

        selected_ids = {wall.wall_id for wall in selected}
        bridges = [
            bridge
            for bridge in bridges_all
            if all(wall_id in selected_ids for wall_id in bridge.wall_ids)
        ]

        # 4) Topología. Los bridges cierran continuidad lógica, nunca muro sólido.
        wall_mask = self._render_wall_mask(selected, shape=(level_view.raster_height_px, level_view.raster_width_px), px_per_m=px_per_m)
        logical_mask = self._logical_seal_mask(
            selected,
            bridges,
            shape=wall_mask.shape,
            px_per_m=px_per_m,
        )
        topology = self._space_topology(logical_mask, px_per_m=px_per_m)
        roles = [self._classify_wall(wall, topology, px_per_m=px_per_m) for wall in selected]
        junctions = self._wall_junctions(selected, px_per_m=px_per_m)

        role_by_id = {item.wall_id: item for item in roles}
        perimeter_count = sum(1 for item in roles if item.role == "PERIMETER")
        divider_count = sum(1 for item in roles if item.role == "DIVIDER")
        review_count = sum(1 for item in roles if item.role == "REVIEW")
        seed_coverage = len(seed_wall_ids & selected_ids) / max(len(seed_wall_ids), 1)
        quarantine_ratio = len(gate.quarantined_candidates) / max(len(discovered), 1)
        review_ratio = review_count / max(len(selected), 1)

        diagnostics = ReconstructionDiagnostics(
            level_view_id=level_view.id,
            f03_seed_count=len(seeds),
            discovered_candidate_count=len(discovered),
            quarantined_candidate_count=len(gate.quarantined_candidates),
            hybrid_candidate_count=len(hybrid_candidates),
            selected_wall_count=len(selected),
            component_count=len(components),
            junction_count=len(junctions),
            virtual_bridge_count=len(bridges),
            interior_space_count=len(topology.interior_labels),
            review_wall_count=review_count,
            perimeter_wall_count=perimeter_count,
            divider_wall_count=divider_count,
            seed_coverage_ratio=max(0.0, min(1.0, seed_coverage)),
            quarantine_ratio=max(0.0, min(1.0, quarantine_ratio)),
            review_ratio=max(0.0, min(1.0, review_ratio)),
        )
        final_plan = self.decision_engine.plan_post_topology(current=pre_plan, diagnostics=diagnostics)

        single_walls: list[SingleLineWall] = []
        for wall in selected:
            role = role_by_id[wall.wall_id]
            single_walls.append(
                SingleLineWall(
                    id=wall.wall_id,
                    start_px=wall.start,
                    end_px=wall.end,
                    thickness_px=wall.thickness_px,
                    role=role.role,
                    confidence=max(0.0, min(1.0, max(wall.confidence, role.confidence))),
                    source_candidate_ids=list(wall.source_candidate_ids),
                    evidence_ids=list(wall.evidence_ids),
                    source_names=list(wall.source_names),
                    f03_seed_protected=wall.wall_id in seed_wall_ids,
                    context_state=wall.context_state,
                    topology_support={
                        "perimeter_votes": role.perimeter_votes,
                        "divider_votes": role.divider_votes,
                        "review_votes": role.review_votes,
                        "left_labels": list(role.left_labels),
                        "right_labels": list(role.right_labels),
                    },
                )
            )

        graph = SingleLineWallGraph(
            level_view_id=level_view.id,
            level_name=level_view.level_name,
            image_size_px=(level_view.raster_width_px, level_view.raster_height_px),
            px_per_m=px_per_m,
            walls=single_walls,
            logical_gaps=[
                LogicalGap(
                    id=bridge.bridge_id,
                    kind=bridge.kind,
                    start_px=bridge.start,
                    end_px=bridge.end,
                    gap_px=bridge.gap_px,
                    wall_ids=list(bridge.wall_ids),
                    protected_by_seed=bridge.protected_by_seed,
                )
                for bridge in bridges
            ],
            interior_space_count=len(topology.interior_labels),
            route_plan=final_plan,
            diagnostics=diagnostics,
        )

        self.trace_recorder.append_graph_decision(graph)

        return AdaptiveReconstructionRuntime(
            graph=graph,
            wall_tracks=selected,
            bridges=bridges,
            topology=topology,
            context_quarantined_count=len(gate.quarantined_candidates),
            artifacts={
                "wall_mask": wall_mask,
                "logical_mask": logical_mask,
                "junctions": junctions,
                "component_report": component_report,
                "f03_seed_candidates": seeds,
                "discovered_candidates": discovered,
                "quarantined_candidates": list(gate.quarantined_candidates),
                "context_decisions": list(gate.decisions),
                "context_regions": list(gate.regions),
                "hybrid_candidates": hybrid_candidates,
            },
        )

    # ------------------------------------------------------------------
    # Candidate Discovery / F03 seed fusion
    # ------------------------------------------------------------------

    def _discover_without_f02_perimeter(self, *, full, scale_profile) -> list[WallCandidate]:
        drawing = full.drawing_model
        generator = self.pipeline.candidate_generator
        constraints = full.reference_constraints
        mask = generator._mask(drawing)
        face_lines = [item for item in drawing.lines if item.kind == "LINE"]
        region_lines = [item for item in drawing.lines if item.kind == "REGION_CENTERLINE"]
        representatives = generator._representative_lines(face_lines, drawing=drawing)
        profiles = generator._thickness_profiles(representatives, drawing=drawing)
        neutral_polygon = box(-4.0, -4.0, drawing.width_px + 4.0, drawing.height_px + 4.0)

        candidates = generator._double_face_candidates(
            drawing=drawing,
            lines=representatives,
            thickness_profiles=profiles,
            perimeter_polygon_geom=neutral_polygon,
            wall_region_mask=mask,
            constraints=constraints,
        )
        candidates.extend(
            generator._region_candidates(
                drawing=drawing,
                region_lines=region_lines,
                face_lines=representatives,
                thickness_profiles=profiles,
                perimeter_polygon_geom=neutral_polygon,
                wall_region_mask=mask,
                constraints=constraints,
            )
        )
        candidates = generator._dedupe_candidates(candidates)
        candidates = WallTrackConsolidator().consolidate(
            candidates=candidates,
            width_px=drawing.width_px,
            height_px=drawing.height_px,
        )
        # axis_support NO se penaliza. Un muro real puede estar alineado al eje.
        return self.pipeline._annotate_metric_thickness(candidates=candidates, scale_profile=scale_profile)

    def _merge_seed_and_context_cleaned(
        self,
        *,
        discovered: Sequence[WallCandidate],
        gate,
        seeds: Sequence[WallCandidate],
        px_per_m: float,
        width: int,
        height: int,
        recovery_strength: float,
    ) -> list[WallCandidate]:
        decisions = {item.candidate_id: item for item in gate.decisions}
        seed_matches = {
            candidate.id
            for candidate in discovered
            if any(self._seed_match(candidate, seed, px_per_m=px_per_m) for seed in seeds)
        }

        kept: dict[str, WallCandidate] = {}
        for candidate in discovered:
            decision = decisions.get(candidate.id)
            state = decision.state if decision is not None else "ACTIVE"
            # DENSE_RECOVERY mantiene REVIEW; QUARANTINE requiere seed equivalente.
            keep = state != "QUARANTINE" or candidate.id in seed_matches
            if not keep:
                continue
            metadata = dict(candidate.metadata)
            metadata.update({
                "adaptive_context_state": state,
                "adaptive_context_region_type": decision.region_type if decision else None,
                "adaptive_structural_seed_match": candidate.id in seed_matches,
                "adaptive_recovery_strength": recovery_strength,
            })
            kept[candidate.id] = candidate.model_copy(update={"metadata": metadata})

        # Seed F03 jamás se pierde por un filtro posterior.
        for seed in seeds:
            if any(self._seed_match(candidate, seed, px_per_m=px_per_m) for candidate in kept.values()):
                continue
            metadata = dict(seed.metadata)
            metadata.update({
                "adaptive_context_state": "PROTECTED",
                "adaptive_context_region_type": "F03_STRUCTURAL_SEED",
                "adaptive_structural_seed_match": True,
                "adaptive_seed_injected": True,
            })
            kept[f"SEED::{seed.id}"] = seed.model_copy(update={"metadata": metadata})

        merged = self.pipeline.candidate_generator._dedupe_candidates(list(kept.values()))
        return WallTrackConsolidator().consolidate(
            candidates=merged,
            width_px=width,
            height_px=height,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_tuple(value) -> tuple[float, float]:
        if hasattr(value, "x"):
            return float(value.x), float(value.y)
        return float(value[0]), float(value[1])

    @classmethod
    def _line(cls, item) -> LineString:
        return LineString([cls._point_tuple(item.start), cls._point_tuple(item.end)])

    @classmethod
    def _vector(cls, item) -> tuple[float, float, float]:
        x1, y1 = cls._point_tuple(item.start)
        x2, y2 = cls._point_tuple(item.end)
        dx, dy = x2 - x1, y2 - y1
        length = max(math.hypot(dx, dy), 1e-9)
        ux, uy = dx / length, dy / length
        if ux < -1e-9 or (abs(ux) <= 1e-9 and uy < 0.0):
            ux, uy = -ux, -uy
        return ux, uy, length

    @classmethod
    def _angle_deg(cls, item) -> float:
        ux, uy, _ = cls._vector(item)
        return math.degrees(math.atan2(uy, ux)) % 180.0

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @classmethod
    def _interval(cls, item, ux: float, uy: float) -> tuple[float, float]:
        start, end = cls._point_tuple(item.start), cls._point_tuple(item.end)
        values = (start[0] * ux + start[1] * uy, end[0] * ux + end[1] * uy)
        return min(values), max(values)

    @classmethod
    def _normal_offset(cls, item, ux: float, uy: float) -> float:
        nx, ny = -uy, ux
        start, end = cls._point_tuple(item.start), cls._point_tuple(item.end)
        mx, my = (start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5
        return mx * nx + my * ny

    @classmethod
    def _candidate_overlap_ratio(cls, first, second) -> float:
        ux, uy, _ = cls._vector(first)
        a0, a1 = cls._interval(first, ux, uy)
        b0, b1 = cls._interval(second, ux, uy)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1e-6, min(a1 - a0, b1 - b0))
        return max(0.0, min(1.0, overlap / denom))

    @classmethod
    def _seed_match(cls, candidate, seed, *, px_per_m: float) -> bool:
        if cls._angle_diff(cls._angle_deg(candidate), cls._angle_deg(seed)) > 5.0:
            return False
        distance = float(cls._line(candidate).distance(cls._line(seed)))
        thickness = max(
            float(getattr(candidate, "thickness_px", 0.0) or 0.0),
            float(getattr(seed, "thickness_px", 0.0) or 0.0),
            0.10 * px_per_m,
        )
        if distance > max(0.16 * px_per_m, 1.2 * thickness):
            return False
        if cls._candidate_overlap_ratio(candidate, seed) >= 0.20:
            return True
        endpoint_distance = min(
            Point(cls._point_tuple(candidate.start)).distance(cls._line(seed)),
            Point(cls._point_tuple(candidate.end)).distance(cls._line(seed)),
            Point(cls._point_tuple(seed.start)).distance(cls._line(candidate)),
            Point(cls._point_tuple(seed.end)).distance(cls._line(candidate)),
        )
        return endpoint_distance <= 0.18 * px_per_m

    @staticmethod
    def _physical_walls(candidates: Sequence[WallCandidate]) -> list[PhysicalWallTrack]:
        output: list[PhysicalWallTrack] = []
        for index, item in enumerate(sorted(candidates, key=lambda c: (round(c.start.y, 3), round(c.start.x, 3), c.id)), 1):
            merged_from = item.metadata.get("consolidated_from")
            source_ids = tuple(sorted(str(v) for v in merged_from)) if isinstance(merged_from, list) else (item.id,)
            output.append(
                PhysicalWallTrack(
                    wall_id=f"PW_{index:03d}",
                    start=(float(item.start.x), float(item.start.y)),
                    end=(float(item.end.x), float(item.end.y)),
                    length_px=float(item.length_px),
                    thickness_px=float(item.thickness_px),
                    confidence=float(item.prior_score),
                    source_candidate_ids=source_ids,
                    evidence_ids=tuple(sorted(str(v) for v in item.evidence_ids)),
                    source_names=tuple(sorted(str(v) for v in item.source_names)),
                    region_support=float(item.evidence.region_support),
                    thickness_support=float(item.evidence.thickness_support),
                    vector_support=float(item.evidence.vector_support),
                    raster_line_support=float(item.evidence.raster_line_support),
                    source_consensus=float(item.evidence.source_consensus),
                    axis_support=float(item.evidence.axis_support),
                    dashed_penalty=float(item.evidence.dashed_penalty),
                    context_state=str(item.metadata.get("adaptive_context_state") or "ACTIVE"),
                )
            )
        return output

    @classmethod
    def _physical_seed_ids(cls, physical: Sequence[PhysicalWallTrack], *, seeds: Sequence[WallCandidate], px_per_m: float) -> set[str]:
        return {
            wall.wall_id
            for wall in physical
            if any(cls._seed_match(wall, seed, px_per_m=px_per_m) for seed in seeds)
        }

    # ------------------------------------------------------------------
    # Strict logical connectivity
    # ------------------------------------------------------------------

    @classmethod
    def _strict_collinear_bridge(cls, first, second, *, seed_ids: set[str], px_per_m: float) -> VirtualBridge | None:
        if cls._angle_diff(cls._angle_deg(first), cls._angle_deg(second)) > 3.0:
            return None
        thickness_ratio = max(first.thickness_px, second.thickness_px) / max(min(first.thickness_px, second.thickness_px), 1e-6)
        if thickness_ratio > 2.0:
            return None
        ux, uy, _ = cls._vector(first)
        first_offset = cls._normal_offset(first, ux, uy)
        second_offset = cls._normal_offset(second, ux, uy)
        thickness = float(np.median([first.thickness_px, second.thickness_px]))
        if abs(first_offset - second_offset) > max(0.06 * px_per_m, 0.90 * thickness):
            return None
        a0, a1 = cls._interval(first, ux, uy)
        b0, b1 = cls._interval(second, ux, uy)
        if a1 < b0:
            t0, t1 = a1, b0
        elif b1 < a0:
            t0, t1 = b1, a0
        else:
            return None
        gap = t1 - t0
        protected = first.wall_id in seed_ids or second.wall_id in seed_ids
        max_gap = (1.20 if protected else 0.70) * px_per_m
        if gap < 0.04 * px_per_m or gap > max_gap:
            return None
        if not protected and min(first.region_support, second.region_support) < 0.38:
            return None
        if min(first.thickness_support, second.thickness_support) < 0.28:
            return None
        nx, ny = -uy, ux
        offset = (first_offset + second_offset) * 0.5
        return VirtualBridge(
            bridge_id=f"HB_COL_{first.wall_id}_{second.wall_id}",
            kind="COLLINEAR_LOGICAL_CONTINUITY",
            start=(ux * t0 + nx * offset, uy * t0 + ny * offset),
            end=(ux * t1 + nx * offset, uy * t1 + ny * offset),
            gap_px=float(gap),
            wall_ids=(first.wall_id, second.wall_id),
            protected_by_seed=protected,
        )

    @classmethod
    def _strict_junction_bridge(cls, source, host, *, endpoint, seed_ids: set[str], px_per_m: float) -> VirtualBridge | None:
        angle = cls._angle_diff(cls._angle_deg(source), cls._angle_deg(host))
        if not (78.0 <= angle <= 102.0):
            return None
        host_line = cls._line(host)
        point = Point(endpoint)
        projected = host_line.interpolate(host_line.project(point))
        distance = float(point.distance(projected))
        protected = source.wall_id in seed_ids or host.wall_id in seed_ids
        max_distance = (0.18 if protected else 0.10) * px_per_m
        if distance <= 1e-6 or distance > max_distance:
            return None
        if not protected and min(source.region_support, host.region_support) < 0.42:
            return None
        return VirtualBridge(
            bridge_id=f"HB_JNC_{source.wall_id}_{host.wall_id}_{round(endpoint[0], 2)}_{round(endpoint[1], 2)}",
            kind="JUNCTION_LOGICAL_CONTINUITY",
            start=(float(endpoint[0]), float(endpoint[1])),
            end=(float(projected.x), float(projected.y)),
            gap_px=distance,
            wall_ids=(source.wall_id, host.wall_id),
            protected_by_seed=protected,
        )

    @classmethod
    def _strict_virtual_bridges(cls, walls: Sequence[PhysicalWallTrack], *, seed_ids: set[str], px_per_m: float) -> list[VirtualBridge]:
        bridges: list[VirtualBridge] = []
        seen: set[tuple] = set()
        for i, first in enumerate(walls):
            for second in walls[i + 1:]:
                bridge = cls._strict_collinear_bridge(first, second, seed_ids=seed_ids, px_per_m=px_per_m)
                if bridge is None:
                    continue
                key = (bridge.kind, tuple(sorted(bridge.wall_ids)), round(bridge.gap_px, 2))
                if key not in seen:
                    seen.add(key)
                    bridges.append(bridge)
        for source in walls:
            for endpoint in (source.start, source.end):
                best = None
                for host in walls:
                    if host.wall_id == source.wall_id:
                        continue
                    bridge = cls._strict_junction_bridge(source, host, endpoint=endpoint, seed_ids=seed_ids, px_per_m=px_per_m)
                    if bridge is not None and (best is None or bridge.gap_px < best.gap_px):
                        best = bridge
                if best is None:
                    continue
                key = (best.kind, best.wall_ids, round(best.start[0], 1), round(best.start[1], 1))
                if key not in seen:
                    seen.add(key)
                    bridges.append(best)
        return bridges

    @classmethod
    def _component_membership(cls, walls: Sequence[PhysicalWallTrack], bridges: Sequence[VirtualBridge], *, px_per_m: float) -> list[list[int]]:
        lines = [cls._line(item) for item in walls]
        adjacency = [set() for _ in walls]
        tolerance = 0.14 * px_per_m
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                if lines[i].distance(lines[j]) <= tolerance:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        index = {wall.wall_id: idx for idx, wall in enumerate(walls)}
        for bridge in bridges:
            a, b = index.get(bridge.wall_ids[0]), index.get(bridge.wall_ids[1])
            if a is not None and b is not None:
                adjacency[a].add(b)
                adjacency[b].add(a)
        remaining = set(range(len(walls)))
        components: list[list[int]] = []
        while remaining:
            seed = next(iter(remaining))
            stack, component = [seed], []
            while stack:
                current = stack.pop()
                if current not in remaining:
                    continue
                remaining.remove(current)
                component.append(current)
                stack.extend(adjacency[current])
            components.append(component)
        components.sort(key=lambda group: sum(walls[idx].length_px for idx in group), reverse=True)
        return components

    @staticmethod
    def _select_hybrid_network(
        walls: Sequence[PhysicalWallTrack],
        components: Sequence[Sequence[int]],
        *,
        seed_ids: set[str],
        px_per_m: float,
        recovery_strength: float,
    ) -> tuple[list[PhysicalWallTrack], list[dict]]:
        selected_indices: set[int] = set()
        report: list[dict] = []
        # Densificación alta acepta componentes geométricamente fuertes algo más
        # pequeños; SEED_FIRST exige más soporte si no hay seed.
        min_component = 2 if recovery_strength >= 0.75 else 3
        min_length_m = 2.2 if recovery_strength >= 0.75 else 3.0
        for component_index, component in enumerate(components, 1):
            total_length_px = sum(walls[index].length_px for index in component)
            total_length_m = total_length_px / px_per_m
            seed_count = sum(1 for index in component if walls[index].wall_id in seed_ids)
            strong_count = sum(
                1
                for index in component
                if walls[index].region_support >= 0.42 and walls[index].thickness_support >= 0.32
            )
            keep = (
                seed_count > 0
                or (len(component) >= min_component and strong_count >= 2)
                or (total_length_m >= min_length_m and strong_count >= 1)
            )
            if keep:
                selected_indices.update(component)
            report.append({
                "component_id": component_index,
                "wall_count": len(component),
                "total_length_m": total_length_m,
                "seed_count": seed_count,
                "strong_wall_count": strong_count,
                "selected": keep,
                "wall_ids": [walls[index].wall_id for index in component],
            })
        return [wall for idx, wall in enumerate(walls) if idx in selected_indices], report

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_segment(image, start, end, value, thickness: int) -> None:
        cv2.line(
            image,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            value,
            thickness,
            cv2.LINE_AA,
        )

    @classmethod
    def _render_wall_mask(cls, walls: Sequence[PhysicalWallTrack], *, shape: tuple[int, int], px_per_m: float) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for wall in walls:
            thickness = max(2, int(round(wall.thickness_px or 0.12 * px_per_m)))
            cls._draw_segment(mask, wall.start, wall.end, 255, thickness)
        return mask

    @classmethod
    def _logical_seal_mask(cls, walls: Sequence[PhysicalWallTrack], bridges: Sequence[VirtualBridge], *, shape: tuple[int, int], px_per_m: float) -> np.ndarray:
        mask = cls._render_wall_mask(walls, shape=shape, px_per_m=px_per_m)
        by_id = {wall.wall_id: wall for wall in walls}
        for bridge in bridges:
            widths = [by_id[wall_id].thickness_px for wall_id in bridge.wall_ids if wall_id in by_id]
            thickness = float(np.median(widths)) if widths else 0.12 * px_per_m
            cls._draw_segment(mask, bridge.start, bridge.end, 255, max(2, int(round(thickness))))
        kernel_size = max(3, int(round(0.05 * px_per_m)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    @staticmethod
    def _space_topology(logical_mask: np.ndarray, *, px_per_m: float) -> SpaceTopology:
        background = (logical_mask == 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
        height, width = logical_mask.shape
        exterior, interior, areas = set(), set(), {}
        min_room_area_px2 = 0.40 * px_per_m * px_per_m
        for label in range(1, count):
            x, y, w, h, area = [int(value) for value in stats[label]]
            areas[label] = area
            touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
            if touches_border:
                exterior.add(label)
            elif area >= min_room_area_px2:
                interior.add(label)
        return SpaceTopology(labels=labels, exterior_labels=frozenset(exterior), interior_labels=frozenset(interior), areas_px2=areas)

    @staticmethod
    def _sample_label(topology: SpaceTopology, point: tuple[float, float]) -> int:
        x, y = int(round(point[0])), int(round(point[1]))
        if x < 0 or y < 0 or y >= topology.labels.shape[0] or x >= topology.labels.shape[1]:
            return -1
        return int(topology.labels[y, x])

    @classmethod
    def _classify_wall(cls, wall: PhysicalWallTrack, topology: SpaceTopology, *, px_per_m: float) -> ClassifiedWall:
        ux, uy, length = cls._vector(wall)
        nx, ny = -uy, ux
        sample_offset = max(wall.thickness_px * 0.5 + 0.08 * px_per_m, 3.0)
        sample_ts = (0.15, 0.30, 0.50, 0.70, 0.85) if length >= 12.0 else (0.50,)
        perimeter_votes = divider_votes = review_votes = 0
        left_labels: list[int] = []
        right_labels: list[int] = []
        for t in sample_ts:
            start, end = wall.start, wall.end
            cx = start[0] + (end[0] - start[0]) * t
            cy = start[1] + (end[1] - start[1]) * t
            left = cls._sample_label(topology, (cx + nx * sample_offset, cy + ny * sample_offset))
            right = cls._sample_label(topology, (cx - nx * sample_offset, cy - ny * sample_offset))
            left_labels.append(left)
            right_labels.append(right)
            left_ext, right_ext = left in topology.exterior_labels, right in topology.exterior_labels
            left_int, right_int = left in topology.interior_labels, right in topology.interior_labels
            if (left_ext and right_int) or (right_ext and left_int):
                perimeter_votes += 1
            elif left_int and right_int and left != right:
                divider_votes += 1
            else:
                review_votes += 1
        total = len(sample_ts)
        minimum_votes = max(1, math.ceil(total * 0.60))
        if perimeter_votes >= minimum_votes and perimeter_votes > divider_votes:
            role, confidence = "PERIMETER", perimeter_votes / total
        elif divider_votes >= minimum_votes and divider_votes > perimeter_votes:
            role, confidence = "DIVIDER", divider_votes / total
        else:
            role, confidence = "REVIEW", max(perimeter_votes, divider_votes) / total
        return ClassifiedWall(
            wall_id=wall.wall_id,
            role=role,
            confidence=float(confidence),
            perimeter_votes=perimeter_votes,
            divider_votes=divider_votes,
            review_votes=review_votes,
            left_labels=tuple(left_labels),
            right_labels=tuple(right_labels),
        )

    @classmethod
    def _wall_junctions(cls, walls: Sequence[PhysicalWallTrack], *, px_per_m: float) -> list[tuple[float, float]]:
        tolerance = 0.12 * px_per_m
        lines = [cls._line(wall) for wall in walls]
        points: list[tuple[float, float]] = []
        for i, first in enumerate(lines):
            for second in lines[i + 1:]:
                if first.distance(second) > tolerance:
                    continue
                intersection = first.intersection(second)
                if intersection.geom_type == "Point":
                    points.append((float(intersection.x), float(intersection.y)))
        deduped: list[tuple[float, float]] = []
        for point in points:
            if not any(math.hypot(point[0] - other[0], point[1] - other[1]) <= 3.0 for other in deduped):
                deduped.append(point)
        return deduped
