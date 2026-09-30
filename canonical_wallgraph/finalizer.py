from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.quantia_spatialV1.adaptive_reconstruction.contracts import (
    AdaptiveReconstructionRuntime,
    LogicalGap,
    PhysicalWallTrack,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
)
from app.quantia_spatialV1.adaptive_reconstruction.decision_engine import AdaptiveModuleDecisionEngine
from app.quantia_spatialV1.adaptive_reconstruction.hybrid_engine import AdaptiveReconstructionEngine
from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.post_reconstruction_filters.contracts import PostReconstructionFilterResult

from .contracts import CanonicalWallGraphIntegrity, CanonicalWallGraphResult
from .evidence_bundle import ReconstructionEvidenceBundleBuilder


class CanonicalWallGraphFinalizer:
    """Reconstruye las relaciones del WallGraph DESPUÉS de PostFilter/canonicalización.

    No selecciona nuevos muros ni cambia su geometría. Solo vuelve a calcular
    conectividad lógica, espacios, roles topológicos, diagnósticos y route plan
    usando exclusivamente los muros canónicos publicados.
    """

    def __init__(
        self,
        *,
        topology_engine: AdaptiveReconstructionEngine | None = None,
        decision_engine: AdaptiveModuleDecisionEngine | None = None,
        evidence_builder: ReconstructionEvidenceBundleBuilder | None = None,
    ) -> None:
        self.topology_engine = topology_engine or AdaptiveReconstructionEngine()
        self.decision_engine = decision_engine or AdaptiveModuleDecisionEngine()
        self.evidence_builder = evidence_builder or ReconstructionEvidenceBundleBuilder()

    def finalize(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
        runtime: AdaptiveReconstructionRuntime,
        post_filter: PostReconstructionFilterResult,
        walls_override: Sequence[SingleLineWall] | None = None,
        source_mapping_override: dict[str, list[str]] | None = None,
        correction_review: dict | None = None,
    ) -> CanonicalWallGraphResult:
        canonical_walls = list(walls_override if walls_override is not None else post_filter.filtered_wall_graph.walls)
        if not canonical_walls:
            raise RuntimeError("CanonicalWallGraphFinalizer requiere al menos un muro canónico.")

        self._validate_partition(runtime, post_filter)
        source_mapping = source_mapping_override if source_mapping_override is not None else {
            item.canonical_wall_id: list(item.source_wall_ids) for item in post_filter.canonical_mapping
        }
        if walls_override is not None and source_mapping_override is None:
            raise ValueError("Corrected walls require explicit source mapping")
        self._validate_walls(canonical_walls, source_mapping, runtime)
        tracks = self._canonical_tracks(runtime=runtime, source_map=source_mapping, walls=canonical_walls)
        seed_ids = {wall.id for wall in canonical_walls if wall.f03_seed_protected}
        px_per_m = runtime.graph.px_per_m

        bridges = self.topology_engine._strict_virtual_bridges(tracks, seed_ids=seed_ids, px_per_m=px_per_m)
        wall_mask = self.topology_engine._render_wall_mask(
            tracks,
            shape=(level_view.raster_height_px, level_view.raster_width_px),
            px_per_m=px_per_m,
        )
        logical_mask = self.topology_engine._logical_seal_mask(
            tracks,
            bridges,
            shape=wall_mask.shape,
            px_per_m=px_per_m,
        )
        topology = self.topology_engine._space_topology(logical_mask, px_per_m=px_per_m)
        roles = [self.topology_engine._classify_wall(track, topology, px_per_m=px_per_m) for track in tracks]
        role_by_id = {item.wall_id: item for item in roles}
        junctions = self.topology_engine._wall_junctions(tracks, px_per_m=px_per_m)
        components = self.topology_engine._component_membership(tracks, bridges, px_per_m=px_per_m)

        finalized_walls: list[SingleLineWall] = []
        for wall in canonical_walls:
            role = role_by_id[wall.id]
            finalized_walls.append(
                wall.model_copy(
                    update={
                        "role": role.role,
                        "confidence": max(wall.confidence, role.confidence),
                        "topology_support": {
                            **dict(wall.topology_support or {}),
                            "finalized_topology": True,
                            "perimeter_votes": role.perimeter_votes,
                            "divider_votes": role.divider_votes,
                            "review_votes": role.review_votes,
                            "left_labels": list(role.left_labels),
                            "right_labels": list(role.right_labels),
                        },
                    },
                    deep=True,
                )
            )

        final_wall_ids = {wall.id for wall in finalized_walls}
        final_gaps = [
            LogicalGap(
                id=bridge.bridge_id,
                kind=bridge.kind,
                start_px=bridge.start,
                end_px=bridge.end,
                gap_px=bridge.gap_px,
                wall_ids=list(bridge.wall_ids),
                protected_by_seed=bridge.protected_by_seed,
                physical_wall_present=False,
                logical_continuity_only=True,
            )
            for bridge in bridges
            if all(wall_id in final_wall_ids for wall_id in bridge.wall_ids)
        ]

        perimeter_count = sum(wall.role == "PERIMETER" for wall in finalized_walls)
        divider_count = sum(wall.role == "DIVIDER" for wall in finalized_walls)
        review_count = sum(wall.role == "REVIEW" for wall in finalized_walls)
        source_seed_ids = {wall.id for wall in runtime.graph.walls if wall.f03_seed_protected}
        represented_source_ids = {
            source_id
            for sources in source_mapping.values()
            for source_id in sources
        }
        represented_seed_count = len(source_seed_ids & represented_source_ids)
        seed_coverage = represented_seed_count / max(len(source_seed_ids), 1)

        upstream = runtime.graph.diagnostics
        diagnostics = ReconstructionDiagnostics(
            level_view_id=runtime.graph.level_view_id,
            f03_seed_count=upstream.f03_seed_count,
            discovered_candidate_count=upstream.discovered_candidate_count,
            quarantined_candidate_count=upstream.quarantined_candidate_count,
            hybrid_candidate_count=upstream.hybrid_candidate_count,
            selected_wall_count=len(finalized_walls),
            component_count=len(components),
            junction_count=len(junctions),
            virtual_bridge_count=len(final_gaps),
            interior_space_count=len(topology.interior_labels),
            review_wall_count=review_count,
            perimeter_wall_count=perimeter_count,
            divider_wall_count=divider_count,
            seed_coverage_ratio=max(0.0, min(1.0, seed_coverage)),
            quarantine_ratio=upstream.quarantine_ratio,
            review_ratio=review_count / max(len(finalized_walls), 1),
        )

        pre_plan = self.decision_engine.plan_pre_topology(
            seed_count=diagnostics.f03_seed_count,
            discovered_count=diagnostics.discovered_candidate_count,
            quarantined_count=diagnostics.quarantined_candidate_count,
        )
        final_plan = self.decision_engine.plan_post_topology(current=pre_plan, diagnostics=diagnostics)

        graph = SingleLineWallGraph(
            level_view_id=runtime.graph.level_view_id,
            level_name=runtime.graph.level_name,
            image_size_px=runtime.graph.image_size_px,
            px_per_m=px_per_m,
            walls=finalized_walls,
            logical_gaps=final_gaps,
            interior_space_count=len(topology.interior_labels),
            route_plan=final_plan,
            diagnostics=diagnostics,
        )

        integrity = self._integrity(graph)
        if not integrity.valid:
            raise ValueError(f"Invalid canonical WallGraph: {integrity.model_dump()}")
        if not self._same_wall_geometry(canonical_walls, finalized_walls):
            raise ValueError("Finalizer changed wall geometry")
        bundle = self.evidence_builder.build(
            level_view=level_view,
            evidence=evidence,
            runtime=runtime,
            post_filter=post_filter,
            final_graph=graph,
            source_mapping=source_mapping,
            correction_review=correction_review,
        )
        return CanonicalWallGraphResult(
            graph=graph,
            integrity=integrity,
            evidence_bundle=bundle,
            audit={
                "post_filter_wall_geometry_unchanged": self._same_wall_geometry(canonical_walls, finalized_walls),
                "relationships_rebuilt_from_final_walls": True,
                "legacy_gap_count": len(post_filter.filtered_wall_graph.logical_gaps),
                "final_gap_count": len(final_gaps),
                "legacy_interior_space_count": post_filter.filtered_wall_graph.interior_space_count,
                "final_interior_space_count": graph.interior_space_count,
            },
        )

    def _canonical_tracks(
        self,
        *,
        runtime: AdaptiveReconstructionRuntime,
        source_map: dict[str, list[str]],
        walls: Sequence[SingleLineWall],
    ) -> list[PhysicalWallTrack]:
        source_track = {track.wall_id: track for track in runtime.wall_tracks}
        output: list[PhysicalWallTrack] = []
        for wall in walls:
            sources = [source_track[item] for item in source_map.get(wall.id, [wall.id]) if item in source_track]
            output.append(
                PhysicalWallTrack(
                    wall_id=wall.id,
                    start=wall.start_px,
                    end=wall.end_px,
                    length_px=math.hypot(wall.end_px[0] - wall.start_px[0], wall.end_px[1] - wall.start_px[1]),
                    thickness_px=wall.thickness_px,
                    confidence=wall.confidence,
                    source_candidate_ids=tuple(wall.source_candidate_ids),
                    evidence_ids=tuple(wall.evidence_ids),
                    source_names=tuple(wall.source_names),
                    region_support=self._aggregate(sources, "region_support", 0.50),
                    thickness_support=self._aggregate(sources, "thickness_support", 0.50),
                    vector_support=self._aggregate(sources, "vector_support", 0.0),
                    raster_line_support=self._aggregate(sources, "raster_line_support", 0.0),
                    source_consensus=self._aggregate(sources, "source_consensus", 0.50),
                    axis_support=self._aggregate(sources, "axis_support", 0.0),
                    dashed_penalty=self._aggregate(sources, "dashed_penalty", 0.0),
                    context_state=wall.context_state,
                )
            )
        return output

    @staticmethod
    def _aggregate(items: Sequence[PhysicalWallTrack], field: str, fallback: float) -> float:
        if not items:
            return fallback
        values = [float(getattr(item, field)) for item in items]
        return float(np.median(values))

    @staticmethod
    def _same_wall_geometry(before: Sequence[SingleLineWall], after: Sequence[SingleLineWall]) -> bool:
        if len(before) != len(after):
            return False
        by_id = {wall.id: wall for wall in after}
        for wall in before:
            other = by_id.get(wall.id)
            if other is None:
                return False
            if wall.start_px != other.start_px or wall.end_px != other.end_px or wall.thickness_px != other.thickness_px:
                return False
        return True

    @staticmethod
    def _integrity(graph: SingleLineWallGraph) -> CanonicalWallGraphIntegrity:
        wall_ids = [wall.id for wall in graph.walls]
        wall_id_set = set(wall_ids)
        invalid_gap_refs = sum(
            any(wall_id not in wall_id_set for wall_id in gap.wall_ids)
            for gap in graph.logical_gaps
        )
        duplicate_wall_ids = len(wall_ids) - len(wall_id_set)
        selected_matches = graph.diagnostics.selected_wall_count == len(graph.walls)
        role_matches = (
            graph.diagnostics.review_wall_count == sum(wall.role == "REVIEW" for wall in graph.walls)
            and graph.diagnostics.perimeter_wall_count == sum(wall.role == "PERIMETER" for wall in graph.walls)
            and graph.diagnostics.divider_wall_count == sum(wall.role == "DIVIDER" for wall in graph.walls)
        )
        interior_matches = graph.diagnostics.interior_space_count == graph.interior_space_count
        valid = invalid_gap_refs == 0 and duplicate_wall_ids == 0 and selected_matches and role_matches and interior_matches
        return CanonicalWallGraphIntegrity(
            wall_count=len(graph.walls),
            logical_gap_count=len(graph.logical_gaps),
            invalid_gap_reference_count=invalid_gap_refs,
            duplicate_wall_id_count=duplicate_wall_ids,
            selected_wall_count_matches=selected_matches,
            role_counts_match=role_matches,
            interior_space_count_matches=interior_matches,
            valid=valid,
        )

    @staticmethod
    def _validate_partition(runtime, post_filter):
        ids = [wall.id for wall in runtime.graph.walls]
        decisions = post_filter.decisions
        decision_ids = [item.source_wall_id for item in decisions]
        if len(ids) != len(set(ids)) or len(decision_ids) != len(set(decision_ids)) or set(ids) != set(decision_ids):
            raise ValueError("PostFilter decisions must partition all Adaptive walls")
        accepted = {item.source_wall_id for item in decisions if item.output_class == "WALL"}
        if any(item.included_in_wallgraph != (item.output_class == "WALL") or not item.preserved for item in decisions):
            raise ValueError("Invalid PostFilter publication or preservation decision")
        mappings = post_filter.canonical_mapping
        sources = [sid for item in mappings for sid in item.source_wall_ids]
        targets = [item.canonical_wall_id for item in mappings]
        if len(sources) != len(set(sources)) or set(sources) != accepted:
            raise ValueError("Canonical mappings must partition accepted source walls")
        if len(targets) != len(set(targets)) or set(targets) != {w.id for w in post_filter.filtered_wall_graph.walls}:
            raise ValueError("Canonical mapping targets do not match published walls")
        originals = {wall.id: wall for wall in runtime.graph.walls}
        for label, bucket in (("ARCHITECTURAL_ELEMENT", post_filter.architectural_elements),
                              ("EXCLUDED_GRAPHIC", post_filter.excluded_graphics), ("UNRESOLVED", post_filter.unresolved)):
            expected = {item.source_wall_id for item in decisions if item.output_class == label}
            actual = [item.source_wall.id for item in bucket]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError(f"Incomplete evidence bucket: {label}")
            by_id = {item.source_wall_id: item for item in decisions}
            if any(item.source_wall != originals[item.source_wall.id] or item.decision != by_id[item.source_wall.id] for item in bucket):
                raise ValueError(f"Modified evidence bucket: {label}")

    @staticmethod
    def _validate_walls(walls, source_mapping, runtime):
        ids = [wall.id for wall in walls]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate canonical wall IDs")
        if set(source_mapping) != set(ids):
            raise ValueError("Final source mapping does not match final wall IDs")
        source_ids = {wall.id for wall in runtime.graph.walls}
        if any(set(sources) - source_ids for sources in source_mapping.values()):
            raise ValueError("Unknown Adaptive source wall")
        geometries = set()
        for wall in walls:
            values = (*wall.start_px, *wall.end_px, wall.thickness_px)
            if not all(math.isfinite(value) for value in values) or wall.thickness_px <= 0 or wall.start_px == wall.end_px:
                raise ValueError("Invalid canonical wall geometry")
            geometry = tuple(sorted((wall.start_px, wall.end_px)))
            if geometry in geometries:
                raise ValueError("Duplicate canonical centerline geometry")
            geometries.add(geometry)
