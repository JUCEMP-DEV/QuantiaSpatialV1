from __future__ import annotations

import math
from collections.abc import Sequence

from app.quantia_spatialV1.stages.walls.adaptive.contracts import AdaptiveReconstructionRuntime, SingleLineWall

from .contracts import FilterModuleDecision, PatternEvidence, PostFilterPlan


class PostFilterSelector:
    """Activa filtros por evidencia observada, nunca por nombre de caso."""

    ARCHITECTURAL_REGION_TYPES = {
        "STAIR_FLIGHT_REGION",
        "DOOR_REGION",
        "WINDOW_REGION",
        "OPENING_REGION",
        "SYMBOL_FIXTURE_REGION",
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

    def plan(
        self,
        *,
        runtime: AdaptiveReconstructionRuntime,
        pattern_evidence: Sequence[PatternEvidence] = (),
    ) -> PostFilterPlan:
        region_types: set[str] = set()
        context_states: set[str] = set()
        for decision in runtime.artifacts.get("context_decisions", []) or []:
            region = getattr(decision, "region_type", None)
            state = getattr(decision, "state", None)
            if region:
                region_types.add(str(region))
            if state:
                context_states.add(str(state))
        for candidate in runtime.artifacts.get("hybrid_candidates", []) or []:
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            region = metadata.get("adaptive_context_region_type") or metadata.get("context_region_type")
            state = metadata.get("adaptive_context_state") or metadata.get("context_state")
            if region:
                region_types.add(str(region))
            if state:
                context_states.add(str(state))

        duplicate_pairs = self._count_duplicate_pairs(runtime.graph.walls, runtime.graph.px_per_m)
        face_pairs = self._count_wall_face_pairs(runtime.graph.walls, runtime.graph.px_per_m)
        near_coincident_pairs = self._count_near_coincident_pairs(runtime.graph.walls, runtime.graph.px_per_m)
        review_heavy = runtime.graph.diagnostics.review_ratio >= 0.15
        repetitive = any(item.kind in {"REPETITIVE_PARALLEL", "ORTHOGONAL_GRID"} for item in pattern_evidence)
        angular_hubs = any(item.kind == "ANGULAR_HUB" for item in pattern_evidence)

        decisions = [
            FilterModuleDecision(
                module="CONTEXT_CLASSIFIER",
                enabled=bool(region_types or context_states),
                reason="Existen decisiones/regiones contextuales trazables en los candidatos de entrada.",
            ),
            FilterModuleDecision(
                module="TOPOLOGY_GUARD",
                enabled=True,
                reason="La topologia ya validada se usa como evidencia de conservacion, nunca como borrado.",
            ),
            FilterModuleDecision(
                module="GRAPHIC_EVIDENCE_FILTER",
                enabled=bool(region_types & self.GRAPHIC_REGION_TYPES) or review_heavy,
                reason="Hay regiones graficas conocidas o una fraccion relevante de muros REVIEW.",
            ),
            FilterModuleDecision(
                module="ARCHITECTURAL_ELEMENT_FILTER",
                enabled=bool(region_types & self.ARCHITECTURAL_REGION_TYPES),
                reason="Hay regiones de elementos arquitectonicos ya detectadas por capas anteriores.",
            ),
            FilterModuleDecision(
                module="REPETITIVE_PATTERN_FILTER",
                enabled=repetitive,
                reason="La geometria post-reconstruccion contiene familias repetitivas regulares que requieren encapsulamiento.",
            ),
            FilterModuleDecision(
                module="ANGULAR_HUB_FILTER",
                enabled=angular_hubs,
                reason="Se detectaron nodos con convergencia angular compatibles con cubierta/proyeccion arquitectonica.",
            ),
            FilterModuleDecision(
                module="RAILING_GUARDRAIL_FILTER",
                enabled=repetitive or angular_hubs or "STAIR_FLIGHT_REGION" in region_types,
                reason="Existe evidencia de bordes abiertos/escaleras o patrones repetitivos que pueden corresponder a barandales/barandales de proteccion.",
            ),
            FilterModuleDecision(
                module="STAIR_HANDRAIL_FILTER",
                enabled="STAIR_FLIGHT_REGION" in region_types or repetitive,
                reason="Existe evidencia de escalera o familias repetitivas asociables a pasamanos/barandal de escalera.",
            ),
            FilterModuleDecision(
                module="SLAB_EDGE_LEVEL_CHANGE_FILTER",
                enabled=review_heavy or angular_hubs,
                reason="Hay REVIEWs sobre bordes abiertos/proyecciones donde pueden existir bordes de losa o cambios de nivel.",
            ),
            FilterModuleDecision(
                module="OPENING_CLOSURE_FILTER",
                enabled=repetitive or bool(region_types & {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"}),
                reason="Hay patrones repetitivos o regiones de opening/door/window compatibles con cierres lineales o cortinas.",
            ),
            FilterModuleDecision(
                module="ALUMINUM_GLAZING_FILTER",
                enabled=bool(region_types & {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"}) or repetitive,
                reason="Hay openings o modulos lineales repetitivos compatibles con canceleria/aluminio de piso a techo.",
            ),
            FilterModuleDecision(
                module="ORPHAN_GRAPHIC_FILTER",
                enabled=review_heavy,
                reason="Los REVIEW aislados se mantienen trazables y pueden excluirse solo con evidencia grafica adicional.",
            ),
            FilterModuleDecision(
                module="DUPLICATE_WALL_CANONICALIZER",
                enabled=duplicate_pairs > 0,
                reason=f"Se detectaron {duplicate_pairs} pares de hipotesis paralelas/solapadas compatibles con un mismo muro fisico.",
            ),
            FilterModuleDecision(
                module="WALL_FACE_CANONICALIZER",
                enabled=face_pairs > 0,
                reason=f"Se detectaron {face_pairs} pares de caras paralelas compatibles con una sola centerline fisica.",
            ),
            FilterModuleDecision(
                module="NEAR_COINCIDENT_CANONICALIZER",
                enabled=near_coincident_pairs > 0,
                reason=f"Se detectaron {near_coincident_pairs} pares casi coincidentes que pueden representar la misma centerline aunque difiera su rol topologico.",
            ),
        ]
        reasons = [item.reason for item in decisions if item.enabled]
        return PostFilterPlan(modules=decisions, reasons=reasons)

    @classmethod
    def _count_duplicate_pairs(cls, walls: Sequence[SingleLineWall], px_per_m: float) -> int:
        total = 0
        for i, first in enumerate(walls):
            for second in walls[i + 1:]:
                if cls._same_wall_hypothesis(first, second, px_per_m=px_per_m):
                    total += 1
        return total

    @classmethod
    def _count_wall_face_pairs(cls, walls: Sequence[SingleLineWall], px_per_m: float) -> int:
        total = 0
        for i, first in enumerate(walls):
            for second in walls[i + 1:]:
                if cls._same_wall_face_pair(first, second, px_per_m=px_per_m):
                    total += 1
        return total


    @classmethod
    def _count_near_coincident_pairs(cls, walls: Sequence[SingleLineWall], px_per_m: float) -> int:
        total = 0
        for i, first in enumerate(walls):
            for second in walls[i + 1:]:
                metrics = cls._pair_geometry(first, second)
                if metrics is None:
                    continue
                overlap_ratio, distance, length_similarity = metrics
                if overlap_ratio < 0.75 or length_similarity < 0.55:
                    continue
                distance_m = distance / max(px_per_m, 1e-6)
                if distance_m <= 0.09:
                    total += 1
        return total

    @staticmethod
    def _angle(wall: SingleLineWall) -> float:
        dx = wall.end_px[0] - wall.start_px[0]
        dy = wall.end_px[1] - wall.start_px[1]
        return math.degrees(math.atan2(dy, dx)) % 180.0

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @classmethod
    def _pair_geometry(cls, first: SingleLineWall, second: SingleLineWall) -> tuple[float, float, float] | None:
        if cls._angle_diff(cls._angle(first), cls._angle(second)) > 3.0:
            return None
        angle = math.radians(cls._angle(first))
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux

        def interval(wall: SingleLineWall) -> tuple[float, float]:
            values = [
                wall.start_px[0] * ux + wall.start_px[1] * uy,
                wall.end_px[0] * ux + wall.end_px[1] * uy,
            ]
            return min(values), max(values)

        a0, a1 = interval(first)
        b0, b1 = interval(second)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1e-6, min(a1 - a0, b1 - b0))
        overlap_ratio = overlap / denom
        length_similarity = min(a1 - a0, b1 - b0) / max(max(a1 - a0, b1 - b0), 1e-6)

        def offset(wall: SingleLineWall) -> float:
            mx = (wall.start_px[0] + wall.end_px[0]) * 0.5
            my = (wall.start_px[1] + wall.end_px[1]) * 0.5
            return mx * nx + my * ny

        distance = abs(offset(first) - offset(second))
        return overlap_ratio, distance, length_similarity

    @classmethod
    def _same_wall_hypothesis(cls, first: SingleLineWall, second: SingleLineWall, *, px_per_m: float) -> bool:
        metrics = cls._pair_geometry(first, second)
        if metrics is None:
            return False
        overlap_ratio, distance, _ = metrics
        if overlap_ratio < 0.60:
            return False
        thickness = max(1.0, min(first.thickness_px, second.thickness_px))
        return distance <= max(0.06 * px_per_m, 0.65 * thickness)

    @classmethod
    def _same_wall_face_pair(cls, first: SingleLineWall, second: SingleLineWall, *, px_per_m: float) -> bool:
        metrics = cls._pair_geometry(first, second)
        if metrics is None:
            return False
        overlap_ratio, distance, length_similarity = metrics
        if overlap_ratio < 0.82 or length_similarity < 0.78:
            return False
        distance_m = distance / max(px_per_m, 1e-6)
        if not 0.065 <= distance_m <= 0.30:
            return False
        thin_a = first.thickness_px / px_per_m <= 0.10
        thin_b = second.thickness_px / px_per_m <= 0.10
        return thin_a or thin_b
