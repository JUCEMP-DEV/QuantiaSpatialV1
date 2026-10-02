from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import median
from typing import Iterable, Sequence

from shapely.geometry import LineString, Point, box

from app.quantia_spatialV1.core.models.parametric import ParametricWall
from app.quantia_spatialV1.stages.walls.core.context_models import ContextRegion
from app.quantia_spatialV1.stages.walls.core.drawing_model import (
    DrawingBBox,
    DrawingCurve,
    DrawingLine,
    DrawingModel,
)
from app.quantia_spatialV1.core.scale.level_scale_normalizer import LevelScaleProfile


class OpeningHypothesisGenerator:
    """Detector independiente de openings para Architectural Elements V2.

    Fuente de verdad:
    - primitivas F01.5 normalizadas en ``DrawingModel``;
    - semántica Gemini ya persistida en ``semantic_observations``;
    - muros finales publicados por Spatial, usados solo como contexto/host.

    No invoca ``ElementContextDetector`` ni consume regiones F03. La salida son
    hipótesis amplias; aceptar/rechazar sigue siendo responsabilidad del mini-motor.
    """

    VERSION = "ARCHITECTURAL_ELEMENTS_HYPOTHESES_V2"

    def generate(
        self,
        *,
        drawing: DrawingModel,
        walls: Sequence[ParametricWall] = (),
        scale_profile: LevelScaleProfile | None = None,
        architectural_polygon=None,
    ) -> list[ContextRegion]:
        m_per_px = self._m_per_px(scale_profile)
        door_constraints = self._semantic_opening_constraints(
            drawing=drawing,
            family="DOOR",
            m_per_px=m_per_px,
        )
        window_constraints = self._semantic_opening_constraints(
            drawing=drawing,
            family="WINDOW",
            m_per_px=m_per_px,
        )

        regions: list[ContextRegion] = []
        regions.extend(
            self._door_regions(
                drawing=drawing,
                walls=walls,
                m_per_px=m_per_px,
                architectural_polygon=architectural_polygon,
                semantic_constraints=door_constraints,
            )
        )
        regions.extend(
            self._wall_gap_door_regions(
                drawing=drawing,
                walls=walls,
                m_per_px=m_per_px,
                architectural_polygon=architectural_polygon,
                semantic_constraints=door_constraints,
                existing=regions,
            )
        )
        regions.extend(
            self._window_regions(
                drawing=drawing,
                walls=walls,
                m_per_px=m_per_px,
                architectural_polygon=architectural_polygon,
                semantic_constraints=window_constraints,
            )
        )
        regions.extend(
            self._direct_semantic_bbox_regions(
                drawing=drawing,
                existing=regions,
                m_per_px=m_per_px,
            )
        )
        return self._deduplicate(regions)

    # ------------------------------------------------------------------
    # DOOR DISCOVERY
    # ------------------------------------------------------------------
    def _door_regions(
        self,
        *,
        drawing: DrawingModel,
        walls: Sequence[ParametricWall],
        m_per_px: float | None,
        architectural_polygon,
        semantic_constraints: Sequence[dict],
    ) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        for curve in drawing.curves:
            arc_support, radius_px = self._door_arc_support(curve=curve, m_per_px=m_per_px)
            if arc_support < 0.42:
                continue

            architecture_support = self._architecture_support(
                bbox=curve.bbox,
                polygon=architectural_polygon,
            )
            if architectural_polygon is not None and architecture_support < 0.08:
                continue

            semantic_support, semantic_ids = self._semantic_bbox_support(
                bbox=curve.bbox,
                constraints=semantic_constraints,
            )
            leaf_ids, leaf_support = self._near_curve_leaf_support(
                curve=curve,
                lines=drawing.lines,
                radius_px=radius_px,
            )
            host_wall_id, host_wall_support, host_offset = self._nearest_wall_support(
                bbox=curve.bbox,
                walls=walls,
                search_px=max(
                    10.0,
                    radius_px * 0.72,
                    (0.32 / m_per_px) if m_per_px else 0.0,
                ),
            )
            raw_host_ids, raw_host_support = self._near_curve_raw_host_support(
                curve=curve,
                lines=drawing.lines,
                radius_px=radius_px,
            )
            host_support = max(host_wall_support, raw_host_support)

            # V2 es deliberadamente de alto recall: una curva plausible sobre un
            # muro final no se descarta solo porque la hoja vectorial esté fragmentada.
            # Los casos débiles pasan a REVIEW en el resolver, no se publican aquí.
            local_support = max(leaf_support, host_support, semantic_support)
            if local_support < 0.18:
                continue
            if host_wall_support < 0.12 and semantic_support < 0.40 and leaf_support < 0.32:
                continue

            confidence = max(
                0.25,
                min(
                    0.91,
                    0.34 * arc_support
                    + 0.20 * leaf_support
                    + 0.24 * host_support
                    + 0.10 * architecture_support
                    + 0.12 * semantic_support,
                ),
            )
            pad = max(3.0, 0.08 * radius_px)
            bbox = self._pad_bbox(curve.bbox, pad, drawing)
            digest = self._stable_id("DOOR", curve.id, bbox)
            host_tokens = set(raw_host_ids)
            if host_wall_id:
                host_tokens.add(host_wall_id)
                for wall in walls:
                    if wall.id == host_wall_id:
                        host_tokens.update(wall.evidence_ids)
                        if wall.source_candidate_id:
                            host_tokens.add(wall.source_candidate_id)
                        if wall.source_f02_wall_id:
                            host_tokens.add(wall.source_f02_wall_id)
                        break

            output.append(
                ContextRegion(
                    id=f"QAE2__DOOR__{digest}",
                    region_type="DOOR_REGION",
                    bbox=bbox,
                    confidence=confidence,
                    quarantine_enabled=False,
                    semantic_support=semantic_support,
                    member_line_ids=sorted(set(leaf_ids)),
                    metadata={
                        "detector": self.VERSION,
                        "source": "INDEPENDENT_CUBIC_DOOR_DISCOVERY_V2",
                        "curve_id": curve.id,
                        "curve_evidence_ids": list(curve.evidence_ids),
                        "door_arc_score": arc_support,
                        "estimated_swing_radius_px": radius_px,
                        "door_leaf_support": leaf_support,
                        "door_host_support": host_support,
                        "host_wall_hint_id": host_wall_id,
                        "host_wall_hint_score": host_wall_support,
                        "host_wall_hint_offset_px": host_offset,
                        "host_line_ids": sorted(host_tokens),
                        "semantic_search_support": semantic_support,
                        "semantic_constraint_ids": semantic_ids,
                        "architecture_support": architecture_support,
                        "opening_verified": bool(
                            arc_support >= 0.62
                            and host_wall_support >= 0.42
                            and (leaf_support >= 0.28 or semantic_support >= 0.50)
                        ),
                        "element_role": "OPENING",
                        "negative_mask": False,
                    },
                )
            )
        return output


    def _wall_gap_door_regions(
        self,
        *,
        drawing: DrawingModel,
        walls: Sequence[ParametricWall],
        m_per_px: float | None,
        architectural_polygon,
        semantic_constraints: Sequence[dict],
        existing: Sequence[ContextRegion],
    ) -> list[ContextRegion]:
        """Descubre interrupciones alineadas en caras de muro.

        Esta ruta cubre puertas raster/vectoriales cuyo arco no sobrevivió como
        CUBIC_BEZIER. Una interrupción por sí sola no se publica: produce REVIEW
        salvo que exista consenso geométrico/semántico suficiente.
        """
        output: list[ContextRegion] = []
        for wall in walls:
            if len(wall.reference_path) < 2 or wall.length_px <= 8.0:
                continue
            a, b = wall.reference_path[0], wall.reference_path[-1]
            wall_angle = self._angle(a.x_px, a.y_px, b.x_px, b.y_px)
            theta = math.radians(wall_angle)
            ux, uy = math.cos(theta), math.sin(theta)
            nx, ny = -uy, ux
            ax_proj = a.x_px * ux + a.y_px * uy
            wall_len = float(wall.length_px)
            thickness = float(wall.thickness_px or 0.0)
            normal_tol = max(thickness * 1.8, (0.24 / m_per_px) if m_per_px else 12.0, 6.0)
            track_tol = max(1.5, thickness * 0.28, normal_tol * 0.10)
            merge_tol = max(2.0, (0.08 / m_per_px) if m_per_px else 4.0)

            track_items: list[tuple[float, float, float, DrawingLine]] = []
            wall_geom = LineString([(a.x_px, a.y_px), (b.x_px, b.y_px)])
            for line in drawing.lines:
                if line.kind != "LINE" or line.dashed or line.length_px <= 2.0:
                    continue
                if self._angle_diff(line.angle_deg, wall_angle) > 4.0:
                    continue
                geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
                if geom.distance(wall_geom) > normal_tol:
                    continue
                values = [
                    line.start.x * ux + line.start.y * uy - ax_proj,
                    line.end.x * ux + line.end.y * uy - ax_proj,
                ]
                lo, hi = max(0.0, min(values)), min(wall_len, max(values))
                if hi - lo <= 2.0:
                    continue
                mx = (line.start.x + line.end.x) * 0.5
                my = (line.start.y + line.end.y) * 0.5
                cross = (mx - a.x_px) * nx + (my - a.y_px) * ny
                track_items.append((cross, lo, hi, line))
            if len(track_items) < 2:
                continue

            track_items.sort(key=lambda item: item[0])
            tracks: list[dict] = []
            for cross, lo, hi, line in track_items:
                if tracks and abs(cross - tracks[-1]["cross"]) <= track_tol:
                    tracks[-1]["items"].append((lo, hi, line))
                    tracks[-1]["cross"] = (tracks[-1]["cross"] + cross) * 0.5
                else:
                    tracks.append({"cross": cross, "items": [(lo, hi, line)]})

            gap_candidates: list[dict] = []
            for track in tracks:
                intervals = sorted((lo, hi) for lo, hi, _ in track["items"])
                merged: list[list[float]] = []
                for lo, hi in intervals:
                    if not merged or lo - merged[-1][1] > merge_tol:
                        merged.append([lo, hi])
                    else:
                        merged[-1][1] = max(merged[-1][1], hi)
                coverage = sum(hi - lo for lo, hi in merged) / max(wall_len, 1e-6)
                if coverage < 0.22:
                    continue
                for left, right in zip(merged, merged[1:]):
                    gap_lo, gap_hi = left[1], right[0]
                    gap_width = gap_hi - gap_lo
                    if gap_width <= merge_tol:
                        continue
                    if m_per_px is not None:
                        gap_m = gap_width * m_per_px
                        if not 0.42 <= gap_m <= 1.80:
                            continue
                        size_support = 1.0 if 0.65 <= gap_m <= 1.25 else 0.66
                    else:
                        ratio = gap_width / max(wall_len, 1e-6)
                        if not 0.025 <= ratio <= 0.30:
                            continue
                        size_support = max(0.45, 1.0 - abs(ratio - 0.10) / 0.20)
                    center = (gap_lo + gap_hi) * 0.5
                    gap_candidates.append({
                        "center": center,
                        "lo": gap_lo,
                        "hi": gap_hi,
                        "width": gap_width,
                        "size_support": size_support,
                        "cross": track["cross"],
                    })

            if not gap_candidates:
                continue

            clusters: list[list[dict]] = []
            for gap in sorted(gap_candidates, key=lambda item: item["center"]):
                matched = None
                for cluster in clusters:
                    c = median(item["center"] for item in cluster)
                    tol = max(6.0, 0.35 * max(gap["width"], median(item["width"] for item in cluster)))
                    if abs(gap["center"] - c) <= tol:
                        matched = cluster
                        break
                if matched is None:
                    clusters.append([gap])
                else:
                    matched.append(gap)

            for cluster in clusters:
                center_offset = float(median(item["center"] for item in cluster))
                width = float(median(item["width"] for item in cluster))
                aligned_tracks = len(cluster)
                cx = a.x_px + ux * center_offset
                cy = a.y_px + uy * center_offset
                half = width * 0.5
                pad_normal = max(4.0, thickness * 1.5, normal_tol * 0.55)
                x_values = [
                    cx - ux * half - nx * pad_normal,
                    cx - ux * half + nx * pad_normal,
                    cx + ux * half - nx * pad_normal,
                    cx + ux * half + nx * pad_normal,
                ]
                y_values = [
                    cy - uy * half - ny * pad_normal,
                    cy - uy * half + ny * pad_normal,
                    cy + uy * half - ny * pad_normal,
                    cy + uy * half + ny * pad_normal,
                ]
                bbox = DrawingBBox(
                    x_min=max(0.0, min(x_values)),
                    y_min=max(0.0, min(y_values)),
                    x_max=min(float(drawing.width_px), max(x_values)),
                    y_max=min(float(drawing.height_px), max(y_values)),
                )
                if self._overlaps_existing(bbox, "DOOR_REGION", [*existing, *output]):
                    continue

                semantic_support, semantic_ids = self._semantic_bbox_support(
                    bbox=bbox,
                    constraints=semantic_constraints,
                )
                curve_support = 0.0
                curve_evidence: list[str] = []
                expanded = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max).buffer(max(3.0, 0.20 * width))
                for curve in drawing.curves:
                    curve_geom = box(curve.bbox.x_min, curve.bbox.y_min, curve.bbox.x_max, curve.bbox.y_max)
                    if not expanded.intersects(curve_geom):
                        continue
                    score, _ = self._door_arc_support(curve=curve, m_per_px=m_per_px)
                    if score > curve_support:
                        curve_support = score
                        curve_evidence = [curve.id, *curve.evidence_ids]

                leaf_support = 0.0
                leaf_ids: list[str] = []
                gap_center = Point(cx, cy)
                for line in drawing.lines:
                    if line.dashed or line.kind != "LINE":
                        continue
                    angle_diff = self._angle_diff(line.angle_deg, wall_angle)
                    if angle_diff < 18.0 or angle_diff > 162.0:
                        continue
                    geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
                    distance = geom.distance(gap_center)
                    if distance > max(8.0, width * 0.55):
                        continue
                    ratio = line.length_px / max(width, 1e-6)
                    if not 0.25 <= ratio <= 1.60:
                        continue
                    score = 0.60 * max(0.0, 1.0 - distance / max(8.0, width * 0.55)) + 0.40 * max(0.0, 1.0 - abs(ratio - 0.90) / 0.90)
                    if score > leaf_support:
                        leaf_support = score
                        leaf_ids = [line.id, *line.evidence_ids]

                architecture_support = self._architecture_support(bbox=bbox, polygon=architectural_polygon)
                aligned_support = min(1.0, aligned_tracks / 2.0)
                size_support = max(item["size_support"] for item in cluster)
                confidence = max(
                    0.24,
                    min(
                        0.90,
                        0.28 * aligned_support
                        + 0.18 * size_support
                        + 0.18 * curve_support
                        + 0.14 * leaf_support
                        + 0.14 * semantic_support
                        + 0.08 * architecture_support,
                    ),
                )
                if aligned_tracks < 2 and max(curve_support, leaf_support, semantic_support) < 0.42:
                    continue
                opening_verified = bool(
                    aligned_tracks >= 2
                    and size_support >= 0.62
                    and (
                        curve_support >= 0.52
                        or semantic_support >= 0.55
                        or leaf_support >= 0.52
                    )
                )
                host_tokens = set(wall.evidence_ids)
                host_tokens.add(wall.id)
                if wall.source_candidate_id:
                    host_tokens.add(wall.source_candidate_id)
                if wall.source_f02_wall_id:
                    host_tokens.add(wall.source_f02_wall_id)
                digest = self._stable_id("DOOR_GAP", f"{wall.id}|{center_offset:.2f}", bbox)
                output.append(
                    ContextRegion(
                        id=f"QAE2__DOOR_GAP__{digest}",
                        region_type="DOOR_REGION",
                        bbox=bbox,
                        confidence=confidence,
                        quarantine_enabled=False,
                        member_line_ids=sorted(set(leaf_ids)),
                        semantic_support=semantic_support,
                        metadata={
                            "detector": self.VERSION,
                            "source": "INDEPENDENT_WALL_GAP_DOOR_DISCOVERY_V2",
                            "door_arc_score": curve_support,
                            "curve_evidence_ids": sorted(set(curve_evidence)),
                            "door_leaf_support": leaf_support,
                            "door_host_support": aligned_support,
                            "wall_gap_support": aligned_support,
                            "gap_track_count": aligned_tracks,
                            "estimated_swing_radius_px": width,
                            "host_wall_hint_id": wall.id,
                            "host_wall_hint_score": max(0.65, aligned_support),
                            "host_wall_hint_offset_px": center_offset,
                            "host_line_ids": sorted(host_tokens),
                            "semantic_search_support": semantic_support,
                            "semantic_constraint_ids": semantic_ids,
                            "architecture_support": architecture_support,
                            "opening_verified": opening_verified,
                            "element_role": "OPENING",
                            "negative_mask": False,
                        },
                    )
                )
        return output

    # ------------------------------------------------------------------
    # WINDOW DISCOVERY
    # ------------------------------------------------------------------
    def _window_regions(
        self,
        *,
        drawing: DrawingModel,
        walls: Sequence[ParametricWall],
        m_per_px: float | None,
        architectural_polygon,
        semantic_constraints: Sequence[dict],
    ) -> list[ContextRegion]:
        if not walls:
            return []

        output: list[ContextRegion] = []
        seen_member_sets: set[tuple[str, ...]] = set()
        for wall in walls:
            if len(wall.reference_path) < 2 or wall.length_px <= 1.0:
                continue
            start = wall.reference_path[0]
            end = wall.reference_path[-1]
            wall_geom = LineString([(start.x_px, start.y_px), (end.x_px, end.y_px)])
            wall_angle = self._angle(start.x_px, start.y_px, end.x_px, end.y_px)
            wall_thickness = float(wall.thickness_px or 0.0)
            normal_tol = max(
                5.0,
                wall_thickness * 1.8,
                (0.24 / m_per_px) if m_per_px else 12.0,
            )

            candidates: list[DrawingLine] = []
            for line in drawing.lines:
                if line.dashed or line.kind != "LINE" or line.length_px <= 2.0:
                    continue
                if self._angle_diff(line.angle_deg, wall_angle) > 4.0:
                    continue
                geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
                if geom.distance(wall_geom) > normal_tol:
                    continue
                length_m = line.length_px * m_per_px if m_per_px else None
                if length_m is not None:
                    if not 0.18 <= length_m <= 4.50:
                        continue
                else:
                    ratio = line.length_px / max(wall.length_px, 1e-6)
                    if not 0.025 <= ratio <= 0.82:
                        continue
                candidates.append(line)

            if len(candidates) < 2:
                continue

            groups = self._parallel_overlap_groups(candidates, max_band=normal_tol)
            for group in groups:
                if len(group) < 2:
                    continue
                tracks = self._distinct_tracks(group, target_angle=wall_angle, min_gap=max(1.0, normal_tol * 0.10))
                if len(tracks) < 2:
                    continue
                representatives = [item[1] for item in tracks]
                member_key = tuple(sorted(line.id for line in representatives))
                if member_key in seen_member_sets:
                    continue

                bbox = self._bbox_for_lines(representatives, drawing=drawing, pad=max(2.0, wall_thickness * 0.35))
                if bbox is None:
                    continue
                long_span = max(bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min)
                if long_span > 0.88 * wall.length_px:
                    continue

                overlap_support = self._group_overlap_support(representatives)
                length_similarity = self._length_similarity(representatives)
                left_support, right_support, through_support = self._wall_interval_support(
                    wall=wall,
                    frame_lines=representatives,
                    all_lines=drawing.lines,
                    normal_tol=normal_tol,
                    m_per_px=m_per_px,
                )
                embedding_support = min(left_support, right_support) * (1.0 - 0.60 * through_support)
                semantic_support, semantic_ids = self._semantic_bbox_support(
                    bbox=bbox,
                    constraints=semantic_constraints,
                )
                architecture_support = self._architecture_support(bbox=bbox, polygon=architectural_polygon)
                if architectural_polygon is not None and architecture_support < 0.05:
                    continue

                frame_support = min(
                    1.0,
                    0.42 * min(1.0, len(tracks) / 3.0)
                    + 0.30 * overlap_support
                    + 0.28 * length_similarity,
                )
                opening_verified = (
                    embedding_support >= 0.42 and through_support <= 0.48
                ) or (
                    semantic_support >= 0.62 and frame_support >= 0.55 and through_support <= 0.62
                )
                confidence = max(
                    0.22,
                    min(
                        0.94,
                        0.28 * frame_support
                        + 0.28 * embedding_support
                        + 0.16 * max(left_support, right_support)
                        + 0.12 * semantic_support
                        + 0.08 * architecture_support
                        + 0.08 * (1.0 - through_support),
                    ),
                )
                # Retener hipótesis ambiguas para RAG/REVIEW, pero no ruido puro.
                if confidence < 0.35 and semantic_support < 0.45:
                    continue

                host_tokens = set(wall.evidence_ids)
                if wall.source_candidate_id:
                    host_tokens.add(wall.source_candidate_id)
                if wall.source_f02_wall_id:
                    host_tokens.add(wall.source_f02_wall_id)
                host_tokens.add(wall.id)
                member_ids = sorted({token for line in representatives for token in (line.id, *line.evidence_ids)})
                digest = self._stable_id("WINDOW", "|".join(member_ids), bbox)
                output.append(
                    ContextRegion(
                        id=f"QAE2__WINDOW__{digest}",
                        region_type="WINDOW_REGION",
                        bbox=bbox,
                        confidence=confidence,
                        quarantine_enabled=False,
                        member_line_ids=member_ids,
                        semantic_support=semantic_support,
                        metadata={
                            "detector": self.VERSION,
                            "source": "INDEPENDENT_WALL_GUIDED_WINDOW_DISCOVERY_V2",
                            "track_count": len(tracks),
                            "parallel_overlap_support": overlap_support,
                            "length_similarity": length_similarity,
                            "host_wall_support": embedding_support,
                            "host_embedding_support": embedding_support,
                            "host_left_support": left_support,
                            "host_right_support": right_support,
                            "through_wall_support": through_support,
                            "host_wall_hint_id": wall.id,
                            "host_wall_hint_score": max(0.55, embedding_support),
                            "host_line_ids": sorted(host_tokens),
                            "semantic_search_support": semantic_support,
                            "semantic_constraint_ids": semantic_ids,
                            "architecture_support": architecture_support,
                            "opening_verified": opening_verified,
                            "element_role": "OPENING",
                            "negative_mask": False,
                        },
                    )
                )
                seen_member_sets.add(member_key)
        return output

    # ------------------------------------------------------------------
    # PERSISTED GEMINI SEMANTICS
    # ------------------------------------------------------------------
    def _direct_semantic_bbox_regions(
        self,
        *,
        drawing: DrawingModel,
        existing: list[ContextRegion],
        m_per_px: float | None,
    ) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        pad = (0.08 / m_per_px) if m_per_px else 5.0
        for obs in drawing.semantic_observations:
            family = str(obs.family or "").upper().strip()
            if family not in {"DOOR", "WINDOW"} or obs.bbox is None:
                continue
            region_type = "DOOR_REGION" if family == "DOOR" else "WINDOW_REGION"
            bbox = self._pad_bbox(obs.bbox, pad, drawing)
            if self._overlaps_existing(bbox, region_type, existing):
                continue
            confidence = max(0.35, min(0.95, float(obs.confidence or 0.65)))
            digest = self._stable_id(f"SEMANTIC_{family}", obs.id, bbox)
            output.append(
                ContextRegion(
                    id=f"QAE2__SEMANTIC__{family}__{digest}",
                    region_type=region_type,
                    bbox=bbox,
                    confidence=confidence,
                    quarantine_enabled=False,
                    member_line_ids=[],
                    semantic_support=confidence,
                    metadata={
                        "detector": self.VERSION,
                        "source": "PERSISTED_GEMINI_DIRECT_BBOX_V2",
                        "semantic_observation_id": obs.id,
                        "semantic_search_support": confidence,
                        "opening_verified": False,
                        "architecture_support": 0.5,
                        "element_role": "OPENING",
                        "negative_mask": False,
                    },
                )
            )
        return output

    def _semantic_opening_constraints(
        self,
        *,
        drawing: DrawingModel,
        family: str,
        m_per_px: float | None,
    ) -> list[dict]:
        wanted = family.upper().strip()
        spaces: dict[str, DrawingBBox] = {}
        for obs in drawing.semantic_observations:
            if str(obs.family or "").upper().strip() != "SPACE" or obs.bbox is None:
                continue
            raw = obs.payload.get("raw_observation") if isinstance(obs.payload, dict) else None
            names = {
                str(obs.payload.get("semantic_name") or "").strip() if isinstance(obs.payload, dict) else "",
                str(raw.get("name") or "").strip() if isinstance(raw, dict) else "",
            }
            for name in names:
                if name:
                    spaces[name] = obs.bbox

        pad = (0.22 / m_per_px) if m_per_px else 18.0
        constraints: list[dict] = []
        for obs in drawing.semantic_observations:
            if str(obs.family or "").upper().strip() != wanted:
                continue
            confidence = max(0.0, min(1.0, float(obs.confidence or 0.70)))
            if obs.bbox is not None:
                constraints.append({
                    "id": obs.id,
                    "bbox": self._pad_bbox(obs.bbox, pad, drawing),
                    "confidence": confidence,
                    "source": "DIRECT_SEMANTIC_BBOX",
                })
                continue
            relations = obs.payload.get("relations") if isinstance(obs.payload, dict) else None
            raw = obs.payload.get("raw_observation") if isinstance(obs.payload, dict) else None
            if not isinstance(relations, list) and isinstance(raw, dict):
                relations = raw.get("relations")
            if not isinstance(relations, list):
                continue
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                if str(relation.get("type") or "").upper() != "LOCATED_IN":
                    continue
                if str(relation.get("target_category") or "").upper() != "SPACE":
                    continue
                target = str(relation.get("target") or "").strip()
                if not target or target not in spaces:
                    continue
                constraints.append({
                    "id": obs.id,
                    "bbox": self._pad_bbox(spaces[target], pad, drawing),
                    "confidence": confidence,
                    "source": "HOST_SPACE_BBOX",
                    "host_space": target,
                })
        return constraints

    # ------------------------------------------------------------------
    # GEOMETRY HELPERS
    # ------------------------------------------------------------------
    def _door_arc_support(self, *, curve: DrawingCurve, m_per_px: float | None) -> tuple[float, float]:
        width = max(1e-6, curve.bbox.x_max - curve.bbox.x_min)
        height = max(1e-6, curve.bbox.y_max - curve.bbox.y_min)
        radius = max(width, height)
        aspect = min(width, height) / radius
        if aspect < 0.16:
            return 0.0, radius
        if m_per_px is not None:
            span_m = radius * m_per_px
            if not 0.30 <= span_m <= 2.20:
                return 0.0, radius
            size = 1.0 if 0.55 <= span_m <= 1.35 else 0.62
        else:
            size = 0.70
        p0, p1, p2, p3 = curve.control_points
        chord = math.hypot(p3.x - p0.x, p3.y - p0.y)
        chord_support = min(1.0, chord / max(radius * 0.70, 1e-6))
        shape = min(1.0, aspect / 0.75)
        tangent_change = self._angle_between(
            (p1.x - p0.x, p1.y - p0.y),
            (p3.x - p2.x, p3.y - p2.y),
        )
        tangent_support = max(0.0, 1.0 - abs(tangent_change - 90.0) / 75.0)
        return max(0.0, min(1.0, 0.34 * size + 0.28 * shape + 0.20 * chord_support + 0.18 * tangent_support)), radius

    def _near_curve_leaf_support(
        self,
        *,
        curve: DrawingCurve,
        lines: Iterable[DrawingLine],
        radius_px: float,
    ) -> tuple[list[str], float]:
        curve_points = [curve.control_points[0], curve.control_points[-1]]
        ids: list[str] = []
        best = 0.0
        tol = max(5.0, 0.32 * radius_px)
        for line in lines:
            if line.dashed or line.kind != "LINE" or line.length_px <= 1.0:
                continue
            ratio = line.length_px / max(radius_px, 1e-6)
            if not 0.18 <= ratio <= 1.55:
                continue
            d = min(
                math.hypot(endpoint.x - cp.x, endpoint.y - cp.y)
                for endpoint in (line.start, line.end)
                for cp in curve_points
            )
            if d > tol:
                continue
            proximity = max(0.0, 1.0 - d / tol)
            length = max(0.0, 1.0 - abs(ratio - 0.82) / 0.82)
            score = 0.64 * proximity + 0.36 * length
            if score >= 0.20:
                ids.extend([line.id, *line.evidence_ids])
                best = max(best, score)
        return sorted(set(ids)), max(0.0, min(1.0, best))

    def _near_curve_raw_host_support(
        self,
        *,
        curve: DrawingCurve,
        lines: Iterable[DrawingLine],
        radius_px: float,
    ) -> tuple[list[str], float]:
        center = Point((curve.bbox.x_min + curve.bbox.x_max) * 0.5, (curve.bbox.y_min + curve.bbox.y_max) * 0.5)
        ids: list[str] = []
        best = 0.0
        tol = max(7.0, 0.55 * radius_px)
        for line in lines:
            if line.dashed or line.kind != "LINE" or line.length_px < 0.45 * radius_px:
                continue
            geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
            distance = geom.distance(center)
            if distance > tol:
                continue
            proximity = max(0.0, 1.0 - distance / tol)
            length = min(1.0, line.length_px / max(1.25 * radius_px, 1e-6))
            score = 0.62 * proximity + 0.38 * length
            if score >= 0.20:
                ids.extend([line.id, *line.evidence_ids])
                best = max(best, score)
        return sorted(set(ids)), max(0.0, min(1.0, best))

    def _nearest_wall_support(
        self,
        *,
        bbox: DrawingBBox,
        walls: Sequence[ParametricWall],
        search_px: float,
    ) -> tuple[str | None, float, float | None]:
        center = Point((bbox.x_min + bbox.x_max) * 0.5, (bbox.y_min + bbox.y_max) * 0.5)
        best_id = None
        best_score = 0.0
        best_offset = None
        for wall in walls:
            if len(wall.reference_path) < 2:
                continue
            a, b = wall.reference_path[0], wall.reference_path[-1]
            geom = LineString([(a.x_px, a.y_px), (b.x_px, b.y_px)])
            distance = float(geom.distance(center))
            support = max(0.0, 1.0 - distance / max(search_px, 1e-6))
            if support > best_score:
                best_score = support
                best_id = wall.id
                best_offset = float(geom.project(center))
        return best_id, max(0.0, min(1.0, best_score)), best_offset

    def _parallel_overlap_groups(self, lines: Sequence[DrawingLine], *, max_band: float) -> list[list[DrawingLine]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        by_id = {line.id: line for line in lines}
        for i, first in enumerate(lines):
            for second in lines[i + 1:]:
                if self._angle_diff(first.angle_deg, second.angle_deg) > 4.0:
                    continue
                ratio = min(first.length_px, second.length_px) / max(first.length_px, second.length_px)
                if ratio < 0.55:
                    continue
                if self._line_overlap_ratio(first, second) < 0.58:
                    continue
                distance = self._normal_distance(first, second)
                if not 0.8 <= distance <= max_band:
                    continue
                adjacency[first.id].add(second.id)
                adjacency[second.id].add(first.id)
        groups: list[list[DrawingLine]] = []
        visited: set[str] = set()
        for line in lines:
            if line.id in visited or line.id not in adjacency:
                continue
            stack = [line.id]
            component: list[DrawingLine] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(by_id[current])
                stack.extend(adjacency.get(current, ()))
            if len(component) >= 2:
                groups.append(component)
        return groups

    def _distinct_tracks(self, lines: Sequence[DrawingLine], *, target_angle: float, min_gap: float) -> list[tuple[float, DrawingLine]]:
        angle = math.radians(target_angle)
        nx, ny = -math.sin(angle), math.cos(angle)
        observed = []
        for line in lines:
            mx = (line.start.x + line.end.x) * 0.5
            my = (line.start.y + line.end.y) * 0.5
            observed.append((mx * nx + my * ny, line))
        observed.sort(key=lambda item: item[0])
        tracks: list[tuple[float, DrawingLine]] = []
        for cross, line in observed:
            if tracks and abs(cross - tracks[-1][0]) < min_gap:
                if line.length_px > tracks[-1][1].length_px:
                    tracks[-1] = (0.5 * (tracks[-1][0] + cross), line)
                continue
            tracks.append((cross, line))
        return tracks

    def _wall_interval_support(
        self,
        *,
        wall: ParametricWall,
        frame_lines: Sequence[DrawingLine],
        all_lines: Sequence[DrawingLine],
        normal_tol: float,
        m_per_px: float | None,
    ) -> tuple[float, float, float]:
        a, b = wall.reference_path[0], wall.reference_path[-1]
        wall_angle = self._angle(a.x_px, a.y_px, b.x_px, b.y_px)
        theta = math.radians(wall_angle)
        ux, uy = math.cos(theta), math.sin(theta)
        nx, ny = -uy, ux

        def interval(line: DrawingLine) -> tuple[float, float]:
            values = [line.start.x * ux + line.start.y * uy, line.end.x * ux + line.end.y * uy]
            return min(values), max(values)

        def cross(line: DrawingLine) -> float:
            mx = (line.start.x + line.end.x) * 0.5
            my = (line.start.y + line.end.y) * 0.5
            return mx * nx + my * ny

        frame_intervals = [interval(line) for line in frame_lines]
        frame_lo = float(median(item[0] for item in frame_intervals))
        frame_hi = float(median(item[1] for item in frame_intervals))
        frame_len = max(1e-6, frame_hi - frame_lo)
        frame_crosses = [cross(line) for line in frame_lines]
        frame_ids = {token for line in frame_lines for token in (line.id, *line.evidence_ids)}
        endpoint_tol = (0.30 / m_per_px) if m_per_px else max(8.0, normal_tol * 0.95)

        left = right = through = 0.0
        for line in all_lines:
            lineage = {line.id, *line.evidence_ids}
            if lineage & frame_ids or line.dashed or line.kind != "LINE":
                continue
            if self._angle_diff(line.angle_deg, wall_angle) > 4.0:
                continue
            normal_distance = min(abs(cross(line) - value) for value in frame_crosses)
            if normal_distance > 1.6 * normal_tol:
                continue
            lo, hi = interval(line)
            if lo <= frame_lo - 0.10 * frame_len and hi >= frame_hi + 0.10 * frame_len:
                through = max(through, min(1.0, (hi - lo) / max(frame_len * 1.8, 1e-6)))
                continue
            gap = frame_lo - hi
            if -0.10 * frame_len <= gap <= endpoint_tol:
                gap_score = max(0.0, 1.0 - max(0.0, gap) / max(endpoint_tol, 1e-6))
                left = max(left, 0.62 * gap_score + 0.38 * min(1.0, line.length_px / max(0.5 * frame_len, 1e-6)))
            gap = lo - frame_hi
            if -0.10 * frame_len <= gap <= endpoint_tol:
                gap_score = max(0.0, 1.0 - max(0.0, gap) / max(endpoint_tol, 1e-6))
                right = max(right, 0.62 * gap_score + 0.38 * min(1.0, line.length_px / max(0.5 * frame_len, 1e-6)))
        return min(1.0, left), min(1.0, right), min(1.0, through)

    @staticmethod
    def _semantic_bbox_support(*, bbox: DrawingBBox, constraints: Sequence[dict]) -> tuple[float, list[str]]:
        if not constraints:
            return 0.0, []
        geom = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        if geom.area <= 1e-6:
            return 0.0, []
        center = geom.centroid
        best = 0.0
        ids: list[str] = []
        for item in constraints:
            target_bbox = item.get("bbox")
            if not isinstance(target_bbox, DrawingBBox):
                continue
            target = box(target_bbox.x_min, target_bbox.y_min, target_bbox.x_max, target_bbox.y_max)
            overlap = target.intersection(geom).area / max(geom.area, 1e-6)
            center_support = 1.0 if target.contains(center) or target.touches(center) else 0.0
            spatial = max(overlap, center_support)
            if spatial <= 0.0:
                continue
            score = spatial * max(0.0, min(1.0, float(item.get("confidence", 0.70))))
            if score > 0.0:
                ids.append(str(item.get("id") or ""))
                best = max(best, score)
        return max(0.0, min(1.0, best)), sorted({item for item in ids if item})

    @staticmethod
    def _architecture_support(*, bbox: DrawingBBox, polygon) -> float:
        if polygon is None:
            return 0.5
        geom = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        if geom.area <= 0.0:
            return 0.0
        return max(0.0, min(1.0, geom.intersection(polygon.buffer(2.0)).area / geom.area))

    @staticmethod
    def _bbox_for_lines(lines: Sequence[DrawingLine], *, drawing: DrawingModel, pad: float) -> DrawingBBox | None:
        if not lines:
            return None
        xs = [p.x for line in lines for p in (line.start, line.end)]
        ys = [p.y for line in lines for p in (line.start, line.end)]
        return DrawingBBox(
            x_min=max(0.0, min(xs) - pad),
            y_min=max(0.0, min(ys) - pad),
            x_max=min(float(drawing.width_px), max(xs) + pad),
            y_max=min(float(drawing.height_px), max(ys) + pad),
        )

    @staticmethod
    def _pad_bbox(bbox: DrawingBBox, pad: float, drawing: DrawingModel) -> DrawingBBox:
        return DrawingBBox(
            x_min=max(0.0, bbox.x_min - pad),
            y_min=max(0.0, bbox.y_min - pad),
            x_max=min(float(drawing.width_px), bbox.x_max + pad),
            y_max=min(float(drawing.height_px), bbox.y_max + pad),
        )

    @staticmethod
    def _angle(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0

    @staticmethod
    def _angle_diff(first: float, second: float) -> float:
        diff = abs(first - second) % 180.0
        return min(diff, 180.0 - diff)

    @staticmethod
    def _angle_between(first: tuple[float, float], second: tuple[float, float]) -> float:
        n1 = math.hypot(*first)
        n2 = math.hypot(*second)
        if n1 <= 1e-9 or n2 <= 1e-9:
            return 0.0
        dot = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / (n1 * n2)))
        return math.degrees(math.acos(dot))

    @staticmethod
    def _normal_distance(first: DrawingLine, second: DrawingLine) -> float:
        theta = math.radians(first.angle_deg)
        nx, ny = -math.sin(theta), math.cos(theta)
        mx1 = (first.start.x + first.end.x) * 0.5
        my1 = (first.start.y + first.end.y) * 0.5
        mx2 = (second.start.x + second.end.x) * 0.5
        my2 = (second.start.y + second.end.y) * 0.5
        return abs((mx2 - mx1) * nx + (my2 - my1) * ny)

    @staticmethod
    def _line_overlap_ratio(first: DrawingLine, second: DrawingLine) -> float:
        theta = math.radians(first.angle_deg)
        ux, uy = math.cos(theta), math.sin(theta)
        a = sorted([first.start.x * ux + first.start.y * uy, first.end.x * ux + first.end.y * uy])
        b = sorted([second.start.x * ux + second.start.y * uy, second.end.x * ux + second.end.y * uy])
        overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        return overlap / max(1e-6, min(a[1] - a[0], b[1] - b[0]))

    @staticmethod
    def _length_similarity(lines: Sequence[DrawingLine]) -> float:
        lengths = [line.length_px for line in lines if line.length_px > 0.0]
        if not lengths:
            return 0.0
        return max(0.0, min(1.0, min(lengths) / max(lengths)))

    def _group_overlap_support(self, lines: Sequence[DrawingLine]) -> float:
        if len(lines) < 2:
            return 0.0
        scores = []
        for i, first in enumerate(lines):
            for second in lines[i + 1:]:
                scores.append(self._line_overlap_ratio(first, second))
        return max(0.0, min(1.0, sum(scores) / len(scores))) if scores else 0.0

    @staticmethod
    def _m_per_px(profile: LevelScaleProfile | None) -> float | None:
        if profile is None or profile.state != "RESOLVED" or profile.local_m_per_px is None:
            return None
        value = float(profile.local_m_per_px)
        return value if value > 0.0 else None

    @staticmethod
    def _stable_id(prefix: str, seed: str, bbox: DrawingBBox) -> str:
        payload = f"{prefix}|{seed}|{bbox.x_min:.2f}|{bbox.y_min:.2f}|{bbox.x_max:.2f}|{bbox.y_max:.2f}"
        return hashlib.sha1(payload.encode()).hexdigest()[:14]

    @staticmethod
    def _overlaps_existing(bbox: DrawingBBox, region_type: str, existing: Sequence[ContextRegion]) -> bool:
        geom = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        for item in existing:
            if item.region_type != region_type:
                continue
            other = box(item.bbox.x_min, item.bbox.y_min, item.bbox.x_max, item.bbox.y_max)
            denom = max(1e-6, min(geom.area, other.area))
            if geom.intersection(other).area / denom >= 0.55:
                return True
        return False

    @classmethod
    def _deduplicate(cls, regions: Sequence[ContextRegion]) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        for region in sorted(regions, key=lambda item: (-item.confidence, item.id)):
            if cls._overlaps_existing(region.bbox, region.region_type, output):
                continue
            output.append(region)
        return sorted(output, key=lambda item: item.id)
