from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Sequence

from shapely.geometry import LineString, Point, box

from .context_models import ContextRegion
from .drawing_model import DrawingBBox, DrawingCurve, DrawingLine, DrawingModel
from .level_scale_normalizer import LevelScaleProfile


class ElementContextDetector:
    """Clasifica elementos funcionales antes del CandidateGraph.

    V4 cierra fallas verificadas en la corrida real de V3:
    1) activa correctamente la semántica legacy DOOR/WINDOW ya persistida;
    2) usa el SPACE anfitrión semántico como ventana de búsqueda, no como máscara;
    3) puertas admiten geometría parcial solo si el contexto semántico/arquitectónico
       y el host físico son compatibles;
    4) ventanas no pueden aislar muros por paralelismo únicamente: sin verificación
       semántica o geométrica fuerte permanecen REVIEW;
    5) ninguna bbox de elemento funciona como máscara negativa global.
    """

    VERSION = "F03_ELEMENT_CONTEXT_ISOLATION_V5"

    SEMANTIC_REGION_MAP = {
        "DOOR": "DOOR_REGION",
        "WINDOW": "WINDOW_REGION",
        "OPENING": "OPENING_REGION",
        "COLUMN": "COLUMN_REGION",
        "FURNITURE": "FURNITURE_MODULE_REGION",
        "FIXTURE": "SYMBOL_FIXTURE_REGION",
        "EQUIPMENT": "SYMBOL_FIXTURE_REGION",
        "PLUMBING": "SYMBOL_FIXTURE_REGION",
        "ELECTRICAL": "SYMBOL_FIXTURE_REGION",
        "SYMBOL": "TEXT_SYMBOL_REGION",
        "TEXT": "TEXT_SYMBOL_REGION",
        "LABEL": "TEXT_SYMBOL_REGION",
        "NOTE": "TEXT_SYMBOL_REGION",
    }

    def detect(
        self,
        *,
        drawing: DrawingModel,
        scale_profile: LevelScaleProfile | None = None,
        architectural_polygon=None,
    ) -> list[ContextRegion]:
        m_per_px = self._m_per_px(scale_profile)
        door_constraints = self._semantic_opening_constraints(
            drawing=drawing, family="DOOR", m_per_px=m_per_px
        )
        window_constraints = self._semantic_opening_constraints(
            drawing=drawing, family="WINDOW", m_per_px=m_per_px
        )
        regions: list[ContextRegion] = []
        regions.extend(
            self._door_regions_from_curves(
                drawing=drawing,
                m_per_px=m_per_px,
                architectural_polygon=architectural_polygon,
                semantic_constraints=door_constraints,
            )
        )
        regions.extend(
            self._window_regions_from_geometry(
                drawing=drawing,
                m_per_px=m_per_px,
                architectural_polygon=architectural_polygon,
                semantic_constraints=window_constraints,
            )
        )
        regions.extend(self._semantic_regions(drawing=drawing, m_per_px=m_per_px))
        regions.extend(self._text_regions(drawing=drawing, m_per_px=m_per_px))
        return self._merge_regions(regions, drawing=drawing)

    # ------------------------------------------------------------------
    # DOORS
    # ------------------------------------------------------------------
    def _door_regions_from_curves(
        self,
        *,
        drawing: DrawingModel,
        m_per_px: float | None,
        architectural_polygon,
        semantic_constraints: Sequence[dict[str, Any]] = (),
    ) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        for curve in drawing.curves:
            arc_score, radius_px = self._door_arc_score(curve=curve, m_per_px=m_per_px)
            if arc_score < 0.58:
                continue

            architecture_support = self._architecture_support(
                bbox=curve.bbox,
                architectural_polygon=architectural_polygon,
                m_per_px=m_per_px,
            )
            semantic_search_support, semantic_constraint_ids = self._semantic_bbox_support(
                bbox=curve.bbox,
                constraints=semantic_constraints,
                m_per_px=m_per_px,
            )
            if architectural_polygon is not None and architecture_support < 0.18:
                # Burbujas de eje y símbolos de referencia suelen quedar fuera del
                # polígono arquitectónico. Una puerta real debe vivir dentro o en
                # contacto inmediato con la planta.
                continue
            # V5: la semántica limita/bloquea falsos positivos solo como evidencia
            # adicional; nunca actúa como veto duro. En los planos reales, Gemini
            # puede localizar correctamente la puerta a nivel de SPACE sin que el
            # bbox del SPACE coincida con el arco vectorial después de normalizar.
            # La geometría física (arco + hoja + host) conserva autoridad local.

            best_hinge: Point | None = None
            best_leaf_ids: list[str] = []
            best_leaf_support = 0.0
            best_host_ids: list[str] = []
            best_host_support = 0.0
            best_hinge_score = -1.0
            for hinge in self._door_hinge_candidates(curve=curve):
                leaf_ids, leaf_support = self._door_leaf_lines(
                    curve=curve,
                    hinge=hinge,
                    lines=drawing.lines,
                    radius_px=radius_px,
                )
                host_ids, host_support = self._door_host_support(
                    curve=curve,
                    hinge=hinge,
                    leaf_ids=set(leaf_ids),
                    lines=drawing.lines,
                    radius_px=radius_px,
                )
                hinge_score = 0.62 * leaf_support + 0.38 * host_support
                if hinge_score > best_hinge_score:
                    best_hinge_score = hinge_score
                    best_hinge = hinge
                    best_leaf_ids = leaf_ids
                    best_leaf_support = leaf_support
                    best_host_ids = host_ids
                    best_host_support = host_support

            hinge = best_hinge
            leaf_ids = best_leaf_ids
            leaf_support = best_leaf_support
            host_ids = best_host_ids
            host_support = best_host_support
            if hinge is None:
                continue
            # V4: dos rutas válidas. La ruta geométrica sigue siendo estricta; la
            # ruta asistida por semántica permite hoja/host fragmentados, pero solo
            # dentro del SPACE donde Gemini reportó una puerta. Nunca se acepta una
            # curva aislada.
            strong_geometry = (
                (leaf_support >= 0.34 and host_support >= 0.12)
                or (leaf_support >= 0.24 and host_support >= 0.24)
            )
            semantic_assisted = (
                semantic_search_support >= 0.45
                and leaf_support >= 0.18
                and host_support >= 0.08
                and architecture_support >= 0.30
            )
            # Si existe semántica pero la curva no cae dentro de su zona de búsqueda,
            # exigimos una geometría local más fuerte en lugar de descartarla. Esto
            # recupera puertas reales sin volver a admitir burbujas de eje.
            semantic_miss_geometry = (
                bool(semantic_constraints)
                and semantic_search_support <= 0.0
                and leaf_support >= 0.40
                and host_support >= 0.18
                and architecture_support >= 0.45
            )
            door_contract_ok = strong_geometry or semantic_assisted or semantic_miss_geometry
            if not door_contract_ok:
                continue

            confidence = max(
                0.0,
                min(
                    0.99,
                    0.34 * arc_score
                    + 0.22 * leaf_support
                    + 0.20 * host_support
                    + 0.08 * architecture_support
                    + 0.16 * semantic_search_support,
                ),
            )
            minimum_confidence = 0.58 if semantic_assisted else (0.60 if semantic_miss_geometry else 0.62)
            if confidence < minimum_confidence:
                continue

            pad_px = max(3.0, (0.07 / m_per_px) if m_per_px else radius_px * 0.08)
            symbol_bbox = self._bbox_from_points(
                [*curve.control_points, hinge],
                drawing=drawing,
                pad=pad_px,
            )
            ident = self._stable_id("DOOR", curve.id, symbol_bbox)
            output.append(
                ContextRegion(
                    id=f"CTX__DOOR_REGION__{ident}",
                    region_type="DOOR_REGION",
                    bbox=symbol_bbox,
                    confidence=confidence,
                    quarantine_enabled=True,
                    # Solo geometría del símbolo. Los host walls se guardan en
                    # metadata y jamás se convierten en miembros negativos.
                    member_line_ids=sorted(set(leaf_ids)),
                    semantic_support=0.0,
                    metadata={
                        "detector": self.VERSION,
                        "source": "CUBIC_BEZIER_DOOR_SWING",
                        "validation_contract": "ARC_LEAF_HOST_OPTIONAL_SEMANTIC_SPACE_V5",
                        "curve_id": curve.id,
                        "curve_evidence_ids": list(curve.evidence_ids),
                        "door_arc_score": arc_score,
                        "door_leaf_support": leaf_support,
                        "door_host_support": host_support,
                        "architecture_support": architecture_support,
                        "semantic_search_support": semantic_search_support,
                        "semantic_constraint_ids": semantic_constraint_ids,
                        "semantic_constraints_present": bool(semantic_constraints),
                        "semantic_miss_geometry": bool(semantic_miss_geometry),
                        "opening_verified": bool(strong_geometry or semantic_assisted or semantic_miss_geometry),
                        "estimated_swing_radius_px": radius_px,
                        "estimated_hinge": [hinge.x, hinge.y],
                        "host_line_ids": sorted(host_ids),
                        "element_role": "OPENING",
                        "negative_mask": False,
                    },
                )
            )
        return output

    def _door_arc_score(
        self,
        *,
        curve: DrawingCurve,
        m_per_px: float | None,
    ) -> tuple[float, float]:
        pts = curve.control_points
        p0, p1, p2, p3 = pts
        width = max(1e-6, curve.bbox.x_max - curve.bbox.x_min)
        height = max(1e-6, curve.bbox.y_max - curve.bbox.y_min)
        radius_px = max(width, height)
        minor = min(width, height)
        aspect = minor / radius_px

        if m_per_px is not None:
            span_m = radius_px * m_per_px
            if not 0.45 <= span_m <= 1.80:
                return 0.0, radius_px
            if 0.60 <= span_m <= 1.30:
                size_support = 1.0
            elif 0.50 <= span_m <= 1.55:
                size_support = 0.82
            else:
                size_support = 0.58
        else:
            size_support = 0.70

        t0 = (p1.x - p0.x, p1.y - p0.y)
        t1 = (p3.x - p2.x, p3.y - p2.y)
        tangent_angle = self._vector_angle(t0, t1)
        endpoint_distance = math.hypot(p3.x - p0.x, p3.y - p0.y)
        chord_ratio = endpoint_distance / max(radius_px, 1e-6)

        aspect_support = max(0.0, min(1.0, (aspect - 0.35) / 0.45))
        tangent_support = max(0.0, 1.0 - abs(tangent_angle - 90.0) / 65.0)
        chord_support = max(0.0, 1.0 - abs(chord_ratio - math.sqrt(2.0)) / 0.75)
        score = 0.42 * size_support + 0.18 * aspect_support + 0.22 * tangent_support + 0.18 * chord_support
        return max(0.0, min(1.0, score)), radius_px

    def _door_hinge_candidates(self, *, curve: DrawingCurve) -> list[Point]:
        p0, p1, p2, p3 = curve.control_points
        raw = [
            Point(curve.bbox.x_min, curve.bbox.y_min),
            Point(curve.bbox.x_min, curve.bbox.y_max),
            Point(curve.bbox.x_max, curve.bbox.y_min),
            Point(curve.bbox.x_max, curve.bbox.y_max),
        ]
        normal = self._normal_intersection(
            p0,
            (p1.x - p0.x, p1.y - p0.y),
            p3,
            (p3.x - p2.x, p3.y - p2.y),
        )
        if normal is not None:
            raw.append(normal)

        candidates: list[tuple[float, Point]] = []
        for hinge in raw:
            r0 = math.hypot(p0.x - hinge.x, p0.y - hinge.y)
            r1 = math.hypot(p3.x - hinge.x, p3.y - hinge.y)
            radius = max(r0, r1, 1e-6)
            similarity = 1.0 - min(1.0, abs(r0 - r1) / radius)
            radial_angle = self._vector_angle(
                (p0.x - hinge.x, p0.y - hinge.y),
                (p3.x - hinge.x, p3.y - hinge.y),
            )
            angle_support = max(0.0, 1.0 - abs(radial_angle - 90.0) / 50.0)
            score = 0.60 * similarity + 0.40 * angle_support
            if score >= 0.62:
                candidates.append((score, hinge))

        deduped: list[Point] = []
        for _, hinge in sorted(candidates, key=lambda item: item[0], reverse=True):
            if any(hinge.distance(existing) <= 2.0 for existing in deduped):
                continue
            deduped.append(hinge)
        return deduped

    def _door_leaf_lines(
        self,
        *,
        curve: DrawingCurve,
        hinge: Point,
        lines: Iterable[DrawingLine],
        radius_px: float,
    ) -> tuple[list[str], float]:
        arc_endpoints = [curve.control_points[0], curve.control_points[-1]]
        output: list[str] = []
        best = 0.0
        hinge_tol = max(4.0, radius_px * 0.23)
        tip_tol = max(4.0, radius_px * 0.26)

        for line in lines:
            if line.kind != "LINE" or line.dashed or line.length_px <= 1e-6:
                continue
            length_ratio = line.length_px / max(radius_px, 1e-6)
            if not 0.34 <= length_ratio <= 1.36:
                continue

            endpoints = [line.start, line.end]
            hinge_distance = min(math.hypot(p.x - hinge.x, p.y - hinge.y) for p in endpoints)
            if hinge_distance > hinge_tol:
                continue
            other = endpoints[1] if math.hypot(endpoints[0].x - hinge.x, endpoints[0].y - hinge.y) <= math.hypot(endpoints[1].x - hinge.x, endpoints[1].y - hinge.y) else endpoints[0]
            tip_distance = min(
                math.hypot(other.x - p.x, other.y - p.y)
                for p in arc_endpoints
            )
            radial = (other.x - hinge.x, other.y - hinge.y)
            radial_support = max(
                max(0.0, 1.0 - self._vector_angle(
                    radial,
                    (p.x - hinge.x, p.y - hinge.y),
                ) / 18.0)
                for p in arc_endpoints
            )
            # Una hoja puede estar fragmentada antes de alcanzar el arco. Si sale
            # claramente desde la bisagra en una de las direcciones radiales del
            # swing, no exigimos que el fragmento llegue al extremo de la curva.
            if tip_distance > tip_tol and radial_support < 0.82:
                continue

            hinge_score = max(0.0, 1.0 - hinge_distance / max(hinge_tol, 1e-6))
            tip_proximity = max(0.0, 1.0 - tip_distance / max(tip_tol, 1e-6))
            tip_score = max(tip_proximity, radial_support * min(1.0, length_ratio / 0.58))
            length_score = max(0.0, 1.0 - abs(length_ratio - 0.82) / 0.62)
            score = 0.42 * hinge_score + 0.38 * tip_score + 0.20 * length_score
            if score >= 0.40:
                output.extend(self._line_lineage_ids(line))
                best = max(best, score)

        return sorted(set(output)), max(0.0, min(1.0, best))

    def _door_host_support(
        self,
        *,
        curve: DrawingCurve,
        hinge: Point,
        leaf_ids: set[str],
        lines: Iterable[DrawingLine],
        radius_px: float,
    ) -> tuple[list[str], float]:
        bbox_geom = box(curve.bbox.x_min, curve.bbox.y_min, curve.bbox.x_max, curve.bbox.y_max)
        hinge_point = Point(hinge.x, hinge.y)
        host_ids: list[str] = []
        best = 0.0
        for line in lines:
            if line.kind != "LINE" or line.dashed or line.length_px < radius_px * 0.62:
                continue
            lineage = self._line_lineage_ids(line)
            if lineage & leaf_ids:
                continue
            geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
            distance = geom.distance(hinge_point)
            if distance > max(5.0, radius_px * 0.30):
                continue
            inside = geom.intersection(bbox_geom).length / max(geom.length, 1e-6)
            # Host wall debe extenderse fuera del símbolo de giro. Esto excluye
            # radios, texto y trazos internos de burbujas.
            outside = max(0.0, 1.0 - inside)
            if outside < 0.12:
                continue
            distance_score = max(0.0, 1.0 - distance / max(radius_px * 0.30, 1e-6))
            length_score = min(1.0, line.length_px / max(radius_px * 1.8, 1e-6))
            score = 0.58 * distance_score + 0.42 * length_score
            best = max(best, score)
            host_ids.extend(lineage)
        return sorted(set(host_ids)), max(0.0, min(1.0, best))

    # ------------------------------------------------------------------
    # WINDOWS
    # ------------------------------------------------------------------
    def _window_regions_from_geometry(
        self,
        *,
        drawing: DrawingModel,
        m_per_px: float | None,
        architectural_polygon,
        semantic_constraints: Sequence[dict[str, Any]] = (),
    ) -> list[ContextRegion]:
        """Detecta ventanas por frame local + host wall flanqueando el opening.

        V2 aceptaba tres líneas paralelas cerca del perímetro, firma que también
        describe una banda de muro. V4 exige que el frame esté embebido entre
        tramos host a izquierda/derecha y rechaza una línea continua que atraviese
        el supuesto opening.
        """
        lines = [
            line
            for line in drawing.lines
            if line.kind == "LINE" and not line.dashed and line.length_px > 1e-6
        ]
        if len(lines) < 3:
            return []

        min_len = (0.30 / m_per_px) if m_per_px else max(22.0, min(drawing.width_px, drawing.height_px) * 0.022)
        max_len = (2.40 / m_per_px) if m_per_px else max(240.0, min(drawing.width_px, drawing.height_px) * 0.30)
        max_band = (0.32 / m_per_px) if m_per_px else max(22.0, min(drawing.width_px, drawing.height_px) * 0.028)
        min_track_gap = (0.018 / m_per_px) if m_per_px else 1.5

        usable = [line for line in lines if min_len <= line.length_px <= max_len]
        components: list[list[DrawingLine]] = []
        remaining = set(range(len(usable)))
        while remaining:
            seed_idx = min(remaining)
            remaining.remove(seed_idx)
            component = [usable[seed_idx]]
            changed = True
            while changed:
                changed = False
                for idx in list(remaining):
                    line = usable[idx]
                    if any(self._window_neighbor(line, ref, max_band=max_band) for ref in component):
                        component.append(line)
                        remaining.remove(idx)
                        changed = True
            if len(component) >= 3:
                components.append(component)

        output: list[ContextRegion] = []
        seen_members: set[tuple[str, ...]] = set()
        for component in components:
            ref = max(component, key=lambda item: item.length_px)
            tracks = self._distinct_parallel_tracks(
                component,
                target_angle=ref.angle_deg,
                min_gap=min_track_gap,
            )
            if not 3 <= len(tracks) <= 5:
                continue
            cross_values = [item[0] for item in tracks]
            band_width = max(cross_values) - min(cross_values)
            if band_width <= min_track_gap or band_width > max_band:
                continue

            representatives = [item[1] for item in tracks]
            lengths = [line.length_px for line in representatives]
            length_similarity = min(lengths) / max(max(lengths), 1e-6)
            if length_similarity < 0.70:
                continue
            overlap_support = self._multi_overlap_support(representatives)
            if overlap_support < 0.72:
                continue

            bbox = self._lines_bbox(representatives, drawing=drawing, pad=max(2.0, 0.06 * band_width))
            architecture_support = self._architecture_support(
                bbox=bbox,
                architectural_polygon=architectural_polygon,
                m_per_px=m_per_px,
            )
            if architectural_polygon is not None and architecture_support < 0.34:
                continue

            semantic_search_support, semantic_constraint_ids = self._semantic_bbox_support(
                bbox=bbox,
                constraints=semantic_constraints,
                m_per_px=m_per_px,
            )
            if semantic_constraints and semantic_search_support <= 0.0 and architecture_support < 0.82:
                continue

            perimeter_support = self._window_perimeter_support(
                representatives=representatives,
                architectural_polygon=architectural_polygon,
                m_per_px=m_per_px,
            )
            embedding_support, host_ids, through_support, left_support, right_support = self._window_host_embedding_support(
                representatives=representatives,
                all_lines=lines,
                max_band=max_band,
                m_per_px=m_per_px,
            )
            # La evidencia principal es la interrupción embebida en host wall.
            # V4 distingue una apertura verificada de una simple banda paralela.
            semantic_verified = semantic_search_support >= 0.55
            strong_geometry_verified = (
                embedding_support >= 0.76
                and through_support < 0.22
                and len(tracks) >= 4
                and (perimeter_support >= 0.45 or architecture_support >= 0.90)
                and (not semantic_constraints or semantic_search_support > 0.0)
            )
            opening_verified = semantic_verified or strong_geometry_verified
            minimum_embedding = 0.50 if semantic_verified else 0.62
            maximum_through = 0.42 if semantic_verified else 0.30
            if embedding_support < minimum_embedding or through_support >= maximum_through:
                continue

            member_ids = sorted({
                lineage_id
                for line in representatives
                for lineage_id in self._line_lineage_ids(line)
            })
            member_key = tuple(member_ids)
            if not member_ids or member_key in seen_members:
                continue
            seen_members.add(member_key)

            confidence = max(
                0.0,
                min(
                    0.97,
                    0.16 * min(1.0, len(tracks) / 4.0)
                    + 0.16 * length_similarity
                    + 0.16 * overlap_support
                    + 0.28 * embedding_support
                    + 0.06 * perimeter_support
                    + 0.06 * architecture_support
                    + 0.12 * semantic_search_support,
                ),
            )
            if confidence < 0.68:
                continue
            digest = self._stable_id("WINDOW", "|".join(member_ids), bbox)
            output.append(
                ContextRegion(
                    id=f"CTX__WINDOW_REGION__{digest}",
                    region_type="WINDOW_REGION",
                    bbox=bbox,
                    confidence=confidence,
                    quarantine_enabled=opening_verified,
                    member_line_ids=member_ids,
                    semantic_support=0.0,
                    metadata={
                        "detector": self.VERSION,
                        "source": "EMBEDDED_PARALLEL_WINDOW_FRAME_V4",
                        "validation_contract": "FRAME_HOST_GAP_SEMANTIC_SPACE_V4",
                        "track_count": len(tracks),
                        "band_width_px": band_width,
                        "length_similarity": length_similarity,
                        "parallel_overlap_support": overlap_support,
                        "perimeter_support": perimeter_support,
                        "host_wall_support": embedding_support,
                        "host_embedding_support": embedding_support,
                        "host_left_support": left_support,
                        "host_right_support": right_support,
                        "through_wall_support": through_support,
                        "host_line_ids": sorted(host_ids),
                        "architecture_support": architecture_support,
                        "semantic_search_support": semantic_search_support,
                        "semantic_constraint_ids": semantic_constraint_ids,
                        "opening_verified": opening_verified,
                        "element_role": "OPENING",
                        "negative_mask": False,
                    },
                )
            )
        return output

    def _window_neighbor(self, first: DrawingLine, second: DrawingLine, *, max_band: float) -> bool:
        if self._angle_diff(first.angle_deg, second.angle_deg) > 3.0:
            return False
        ratio = min(first.length_px, second.length_px) / max(first.length_px, second.length_px)
        if ratio < 0.66:
            return False
        if self._line_overlap_ratio(first, second) < 0.70:
            return False
        distance = self._normal_distance(first, second)
        return 1.0 <= distance <= max_band

    def _distinct_parallel_tracks(
        self,
        lines: Sequence[DrawingLine],
        *,
        target_angle: float,
        min_gap: float,
    ) -> list[tuple[float, DrawingLine]]:
        angle = math.radians(target_angle)
        nx, ny = -math.sin(angle), math.cos(angle)
        observed: list[tuple[float, DrawingLine]] = []
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

    def _window_perimeter_support(self, *, representatives, architectural_polygon, m_per_px) -> float:
        if architectural_polygon is None:
            return 0.0
        tolerance = (0.24 / m_per_px) if m_per_px else 22.0
        best = 0.0
        for line in representatives:
            geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
            distance = geom.distance(architectural_polygon.boundary)
            best = max(best, max(0.0, 1.0 - distance / max(tolerance, 1e-6)))
        return max(0.0, min(1.0, best))

    def _window_host_embedding_support(
        self,
        *,
        representatives: Sequence[DrawingLine],
        all_lines: Sequence[DrawingLine],
        max_band: float,
        m_per_px: float | None,
    ) -> tuple[float, set[str], float, float, float]:
        ref = max(representatives, key=lambda item: item.length_px)
        member_lineage = {
            lineage_id
            for line in representatives
            for lineage_id in self._line_lineage_ids(line)
        }
        angle = math.radians(ref.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux

        def interval(line: DrawingLine) -> tuple[float, float]:
            values = [
                line.start.x * ux + line.start.y * uy,
                line.end.x * ux + line.end.y * uy,
            ]
            return float(min(values)), float(max(values))

        def cross(line: DrawingLine) -> float:
            mx = (line.start.x + line.end.x) * 0.5
            my = (line.start.y + line.end.y) * 0.5
            return float(mx * nx + my * ny)

        rep_intervals = [interval(line) for line in representatives]
        frame_lo = float(median([item[0] for item in rep_intervals]))
        frame_hi = float(median([item[1] for item in rep_intervals]))
        frame_len = max(1e-6, frame_hi - frame_lo)
        rep_crosses = [cross(line) for line in representatives]
        endpoint_tol = (0.28 / m_per_px) if m_per_px else max(8.0, 0.80 * max_band)
        normal_tol = 1.55 * max_band

        left_best = 0.0
        right_best = 0.0
        through_best = 0.0
        left_ids: set[str] = set()
        right_ids: set[str] = set()

        for line in all_lines:
            lineage = self._line_lineage_ids(line)
            if lineage & member_lineage or line.dashed:
                continue
            if self._angle_diff(line.angle_deg, ref.angle_deg) > 3.0:
                continue
            c = cross(line)
            normal_distance = min(abs(c - value) for value in rep_crosses)
            if normal_distance > normal_tol:
                continue
            lo, hi = interval(line)
            if hi <= lo:
                continue

            # Una línea que cruza el frame y continúa por ambos lados es evidencia
            # de muro continuo, no de opening.
            if lo <= frame_lo - 0.12 * frame_len and hi >= frame_hi + 0.12 * frame_len:
                crossing = min(1.0, (hi - lo) / max(frame_len * 1.8, 1e-6))
                through_best = max(through_best, crossing)
                continue

            left_gap = frame_lo - hi
            if -0.10 * frame_len <= left_gap <= endpoint_tol:
                gap_score = max(0.0, 1.0 - max(0.0, left_gap) / max(endpoint_tol, 1e-6))
                length_score = min(1.0, line.length_px / max(0.55 * frame_len, 1e-6))
                normal_score = max(0.0, 1.0 - normal_distance / max(normal_tol, 1e-6))
                score = 0.46 * gap_score + 0.34 * length_score + 0.20 * normal_score
                if score > left_best:
                    left_best = score
                    left_ids = set(lineage)

            right_gap = lo - frame_hi
            if -0.10 * frame_len <= right_gap <= endpoint_tol:
                gap_score = max(0.0, 1.0 - max(0.0, right_gap) / max(endpoint_tol, 1e-6))
                length_score = min(1.0, line.length_px / max(0.55 * frame_len, 1e-6))
                normal_score = max(0.0, 1.0 - normal_distance / max(normal_tol, 1e-6))
                score = 0.46 * gap_score + 0.34 * length_score + 0.20 * normal_score
                if score > right_best:
                    right_best = score
                    right_ids = set(lineage)

        if left_best <= 0.0 or right_best <= 0.0:
            embedding = 0.0
        else:
            embedding = min(left_best, right_best) * (1.0 - 0.55 * min(1.0, through_best))
        return (
            max(0.0, min(1.0, embedding)),
            left_ids | right_ids,
            max(0.0, min(1.0, through_best)),
            max(0.0, min(1.0, left_best)),
            max(0.0, min(1.0, right_best)),
        )

    # ------------------------------------------------------------------
    # SEMANTIC SEARCH CONSTRAINTS
    # ------------------------------------------------------------------
    def _semantic_opening_constraints(
        self,
        *,
        drawing: DrawingModel,
        family: str,
        m_per_px: float | None,
    ) -> list[dict[str, Any]]:
        """Convierte semántica legacy en zonas de búsqueda, nunca en máscaras.

        DOOR/WINDOW del contrato legacy normalmente no traen bbox propio, pero sí
        una relación LOCATED_IN al SPACE anfitrión. Ese SPACE sí conserva
        bbox_normalized y puede limitar falsos positivos (p. ej. burbujas de eje)
        sin afirmar una ubicación exacta del opening.
        """
        wanted = str(family or "").upper().strip()
        if wanted not in {"DOOR", "WINDOW", "OPENING"}:
            return []

        spaces: dict[str, DrawingBBox] = {}
        for obs in drawing.semantic_observations:
            if str(obs.family or "").upper().strip() != "SPACE" or obs.bbox is None:
                continue
            names = {
                str(obs.payload.get("semantic_name") or "").strip(),
                str((obs.payload.get("raw_observation") or {}).get("name") or "").strip()
                if isinstance(obs.payload.get("raw_observation"), dict) else "",
            }
            for name in names:
                if name:
                    spaces[name] = obs.bbox

        pad_px = (0.22 / m_per_px) if m_per_px else 18.0
        constraints: list[dict[str, Any]] = []
        for obs in drawing.semantic_observations:
            if str(obs.family or "").upper().strip() != wanted:
                continue
            confidence = float(obs.confidence if obs.confidence is not None else 0.70)
            if obs.bbox is not None:
                constraints.append({
                    "id": obs.id,
                    "bbox": self._pad_bbox(obs.bbox, pad_px, drawing=drawing),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "source": "DIRECT_SEMANTIC_BBOX",
                })
                continue

            relations = obs.payload.get("relations")
            if not isinstance(relations, list):
                raw = obs.payload.get("raw_observation")
                relations = raw.get("relations") if isinstance(raw, dict) else []
            targets: list[str] = []
            if isinstance(relations, list):
                for relation in relations:
                    if not isinstance(relation, dict):
                        continue
                    if str(relation.get("type") or "").upper() != "LOCATED_IN":
                        continue
                    if str(relation.get("target_category") or "").upper() != "SPACE":
                        continue
                    target = str(relation.get("target") or "").strip()
                    if target:
                        targets.append(target)
            for target in targets:
                bbox = spaces.get(target)
                if bbox is None:
                    continue
                constraints.append({
                    "id": obs.id,
                    "bbox": self._pad_bbox(bbox, pad_px, drawing=drawing),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "source": "HOST_SPACE_BBOX",
                    "host_space": target,
                })

        # Deduplicar por observación+bbox; varias relaciones equivalentes no deben
        # multiplicar peso semántico.
        unique: dict[tuple, dict[str, Any]] = {}
        for item in constraints:
            bbox = item["bbox"]
            key = (
                str(item.get("id") or ""),
                round(bbox.x_min, 2), round(bbox.y_min, 2),
                round(bbox.x_max, 2), round(bbox.y_max, 2),
            )
            unique[key] = item
        return list(unique.values())

    @staticmethod
    def _semantic_bbox_support(
        *,
        bbox: DrawingBBox,
        constraints: Sequence[dict[str, Any]],
        m_per_px: float | None,
    ) -> tuple[float, list[str]]:
        if not constraints:
            return 0.0, []
        geom = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        if geom.area <= 1e-6:
            return 0.0, []
        center = geom.centroid
        best = 0.0
        matched: list[str] = []
        for item in constraints:
            raw_bbox = item.get("bbox")
            if not isinstance(raw_bbox, DrawingBBox):
                continue
            target = box(raw_bbox.x_min, raw_bbox.y_min, raw_bbox.x_max, raw_bbox.y_max)
            overlap = target.intersection(geom).area / max(geom.area, 1e-6)
            center_support = 1.0 if target.contains(center) or target.touches(center) else 0.0
            spatial = max(overlap, center_support)
            if spatial <= 0.0:
                continue
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.70) or 0.70)))
            score = spatial * confidence
            if score > 0.0:
                matched.append(str(item.get("id") or ""))
                best = max(best, score)
        return max(0.0, min(1.0, best)), sorted({item for item in matched if item})

    # ------------------------------------------------------------------
    # SEMANTICS / TEXT
    # ------------------------------------------------------------------
    def _semantic_regions(self, *, drawing: DrawingModel, m_per_px: float | None) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        drawing_area = max(1.0, float(drawing.width_px * drawing.height_px))
        min_dim = float(min(drawing.width_px, drawing.height_px))
        for obs in drawing.semantic_observations:
            family = str(obs.family or "").upper().strip()
            region_type = self.SEMANTIC_REGION_MAP.get(family)
            if region_type is None or obs.bbox is None:
                continue

            width = max(0.0, obs.bbox.x_max - obs.bbox.x_min)
            height = max(0.0, obs.bbox.y_max - obs.bbox.y_min)
            area_ratio = (width * height) / drawing_area
            # Un bbox semántico genérico de texto/símbolo que cubre gran parte de
            # la planta no debe convertirse en contexto de candidatos de muro.
            if region_type == "TEXT_SYMBOL_REGION" and (
                area_ratio > 0.018 or max(width, height) > 0.22 * min_dim
            ):
                continue

            confidence = float(obs.confidence if obs.confidence is not None else 0.70)
            pad_m = 0.08 if region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"} else 0.04
            pad_px = (pad_m / m_per_px) if m_per_px else 4.0
            bbox = self._pad_bbox(obs.bbox, pad_px, drawing=drawing)
            members = self._semantic_member_lines(
                bbox=bbox,
                drawing=drawing,
                region_type=region_type,
            )
            output.append(
                ContextRegion(
                    id=f"CTX__{region_type}__{self._stable_id(region_type, obs.id, bbox)}",
                    region_type=region_type,
                    bbox=bbox,
                    confidence=max(0.0, min(1.0, confidence)),
                    quarantine_enabled=region_type not in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"},
                    member_line_ids=members,
                    semantic_support=max(0.0, min(1.0, confidence)),
                    metadata={
                        "detector": self.VERSION,
                        "source": "GEMINI_SEMANTIC",
                        "semantic_observation_id": obs.id,
                        "semantic_family": family,
                        "payload": dict(obs.payload),
                        "element_role": "OPENING" if region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"} else family,
                        "negative_mask": False if region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"} else True,
                    },
                )
            )
        return output

    def _text_regions(self, *, drawing: DrawingModel, m_per_px: float | None) -> list[ContextRegion]:
        output: list[ContextRegion] = []
        pad_px = (0.015 / m_per_px) if m_per_px else 1.5
        for text in drawing.texts:
            if text.bbox is None:
                continue
            width = text.bbox.x_max - text.bbox.x_min
            height = text.bbox.y_max - text.bbox.y_min
            if width <= 0.0 or height <= 0.0:
                continue
            bbox = self._pad_bbox(text.bbox, pad_px, drawing=drawing)
            output.append(
                ContextRegion(
                    id=f"CTX__TEXT_SYMBOL_REGION__{self._stable_id('TEXT', text.id, bbox)}",
                    region_type="TEXT_SYMBOL_REGION",
                    bbox=bbox,
                    confidence=max(0.72, float(text.confidence or 0.0)),
                    quarantine_enabled=True,
                    member_line_ids=[],
                    semantic_support=0.0,
                    metadata={
                        "detector": self.VERSION,
                        "source": "TEXT_BBOX",
                        "text_id": text.id,
                        "text": text.text,
                        "negative_mask": False,
                    },
                )
            )
        return output

    def _semantic_member_lines(self, *, bbox: DrawingBBox, drawing: DrawingModel, region_type: str) -> list[str]:
        region = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        output: list[str] = []
        for line in drawing.lines:
            if line.kind != "LINE" or line.length_px <= 1e-6:
                continue
            geom = LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
            inside = geom.intersection(region).length / max(geom.length, 1e-6)
            if inside < 0.82:
                continue
            if region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"}:
                region_span = max(bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min)
                if line.length_px > region_span * 1.20:
                    continue
            output.extend(self._line_lineage_ids(line))
        return sorted(set(output))

    def _merge_regions(self, regions: list[ContextRegion], *, drawing: DrawingModel) -> list[ContextRegion]:
        if not regions:
            return []

        # TEXT debe permanecer local. Fusionar bboxes de rótulos adyacentes creó
        # regiones gigantes que cruzaban decenas de candidatos en Planta Alta.
        no_merge_types = {"TEXT_SYMBOL_REGION"}
        used = [False] * len(regions)
        output: list[ContextRegion] = []
        for i, base in enumerate(regions):
            if used[i]:
                continue
            used[i] = True
            if base.region_type in no_merge_types:
                output.append(base)
                continue

            group = [base]
            for j in range(i + 1, len(regions)):
                if used[j] or regions[j].region_type != base.region_type:
                    continue
                overlap = self._bbox_overlap(base.bbox, regions[j].bbox)
                threshold = 0.62 if base.region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"} else 0.42
                if overlap < threshold:
                    continue
                used[j] = True
                group.append(regions[j])
            if len(group) == 1:
                output.append(base)
                continue
            bbox = DrawingBBox(
                x_min=min(x.bbox.x_min for x in group),
                y_min=min(x.bbox.y_min for x in group),
                x_max=max(x.bbox.x_max for x in group),
                y_max=max(x.bbox.y_max for x in group),
            )
            members = sorted({line_id for x in group for line_id in x.member_line_ids})
            metadata = {
                "detector": self.VERSION,
                "merged_region_ids": [x.id for x in group],
                "sources": sorted({str(x.metadata.get("source")) for x in group}),
                "negative_mask": all(bool(x.metadata.get("negative_mask", False)) for x in group),
            }
            output.append(
                ContextRegion(
                    id=f"CTX__{base.region_type}__{self._stable_id(base.region_type, '|'.join(x.id for x in group), bbox)}",
                    region_type=base.region_type,
                    bbox=bbox,
                    confidence=max(x.confidence for x in group),
                    quarantine_enabled=any(x.quarantine_enabled for x in group),
                    member_line_ids=members,
                    semantic_support=max(x.semantic_support for x in group),
                    metadata=metadata,
                )
            )
        return output

    # ------------------------------------------------------------------
    # GEOMETRY HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _m_per_px(profile: LevelScaleProfile | None) -> float | None:
        if profile is None or profile.state != "RESOLVED":
            return None
        value = profile.canonical_m_per_px or profile.local_m_per_px
        if value is None or value <= 0.0:
            return None
        return float(value)

    @staticmethod
    def _normal_intersection(p0, t0, p1, t1) -> Point | None:
        # Normal a cada tangente: n=(-ty, tx). Intersección de líneas infinitas.
        n0 = (-t0[1], t0[0])
        n1 = (-t1[1], t1[0])
        det = n0[0] * (-n1[1]) - (-n1[0]) * n0[1]
        if abs(det) <= 1e-8:
            return None
        bx = p1.x - p0.x
        by = p1.y - p0.y
        # [n0x -n1x; n0y -n1y] [a;b] = [bx;by]
        a = (bx * (-n1[1]) - (-n1[0]) * by) / det
        return Point(p0.x + a * n0[0], p0.y + a * n0[1])

    @staticmethod
    def _vector_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
        na = math.hypot(*a)
        nb = math.hypot(*b)
        if na <= 1e-6 or nb <= 1e-6:
            return 0.0
        dot = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
        angle = math.degrees(math.acos(dot))
        return min(angle, 180.0 - angle)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _normal_distance(first: DrawingLine, second: DrawingLine) -> float:
        angle = math.radians(first.angle_deg)
        nx, ny = -math.sin(angle), math.cos(angle)
        a = ((first.start.x + first.end.x) * 0.5, (first.start.y + first.end.y) * 0.5)
        b = ((second.start.x + second.end.x) * 0.5, (second.start.y + second.end.y) * 0.5)
        return abs((a[0] - b[0]) * nx + (a[1] - b[1]) * ny)

    @staticmethod
    def _line_overlap_ratio(first: DrawingLine, second: DrawingLine) -> float:
        angle = math.radians(first.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        a = sorted((first.start.x * ux + first.start.y * uy, first.end.x * ux + first.end.y * uy))
        b = sorted((second.start.x * ux + second.start.y * uy, second.end.x * ux + second.end.y * uy))
        overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        return max(0.0, min(1.0, overlap / max(1e-6, min(a[1] - a[0], b[1] - b[0]))))

    def _multi_overlap_support(self, lines: Sequence[DrawingLine]) -> float:
        if len(lines) < 2:
            return 0.0
        values = []
        for i, first in enumerate(lines):
            for second in lines[i + 1 :]:
                values.append(self._line_overlap_ratio(first, second))
        return float(median(values)) if values else 0.0

    @staticmethod
    def _line_lineage_ids(line: DrawingLine) -> set[str]:
        ids: set[str] = {str(line.id)}
        ids.update(str(item) for item in line.evidence_ids if item)
        for key in (
            "merged_line_ids",
            "physical_stroke_member_line_ids",
            "context_equivalent_line_ids",
        ):
            raw = line.metadata.get(key)
            if isinstance(raw, (list, tuple, set)):
                ids.update(str(item) for item in raw if item)
        return ids

    @staticmethod
    def _architecture_support(*, bbox: DrawingBBox, architectural_polygon, m_per_px: float | None) -> float:
        if architectural_polygon is None:
            return 1.0
        pad = (0.12 / m_per_px) if m_per_px else 10.0
        geom = box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        if geom.area <= 1e-6:
            return 0.0
        expanded = architectural_polygon.buffer(pad, cap_style=2, join_style=2)
        return max(0.0, min(1.0, expanded.intersection(geom).area / geom.area))

    @staticmethod
    def _bbox_from_points(points, *, drawing: DrawingModel, pad: float) -> DrawingBBox:
        xs = [float(p.x) for p in points]
        ys = [float(p.y) for p in points]
        return DrawingBBox(
            x_min=max(0.0, min(xs) - pad),
            y_min=max(0.0, min(ys) - pad),
            x_max=min(float(drawing.width_px), max(xs) + pad),
            y_max=min(float(drawing.height_px), max(ys) + pad),
        )

    @staticmethod
    def _lines_bbox(lines: Sequence[DrawingLine], *, drawing: DrawingModel, pad: float) -> DrawingBBox:
        xs = [value for line in lines for value in (line.start.x, line.end.x)]
        ys = [value for line in lines for value in (line.start.y, line.end.y)]
        return DrawingBBox(
            x_min=max(0.0, min(xs) - pad),
            y_min=max(0.0, min(ys) - pad),
            x_max=min(float(drawing.width_px), max(xs) + pad),
            y_max=min(float(drawing.height_px), max(ys) + pad),
        )

    @staticmethod
    def _pad_bbox(bbox: DrawingBBox, pad: float, *, drawing: DrawingModel) -> DrawingBBox:
        return DrawingBBox(
            x_min=max(0.0, bbox.x_min - pad),
            y_min=max(0.0, bbox.y_min - pad),
            x_max=min(float(drawing.width_px), bbox.x_max + pad),
            y_max=min(float(drawing.height_px), bbox.y_max + pad),
        )

    @staticmethod
    def _bbox_overlap(a: DrawingBBox, b: DrawingBBox) -> float:
        ax1, ay1, ax2, ay2 = a.x_min, a.y_min, a.x_max, a.y_max
        bx1, by1, bx2, by2 = b.x_min, b.y_min, b.x_max, b.y_max
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        amin = min(
            max((ax2 - ax1) * (ay2 - ay1), 1e-6),
            max((bx2 - bx1) * (by2 - by1), 1e-6),
        )
        return inter / amin

    @staticmethod
    def _stable_id(prefix: str, payload: str, bbox: DrawingBBox) -> str:
        raw = f"{prefix}|{payload}|{bbox.x_min:.2f}|{bbox.y_min:.2f}|{bbox.x_max:.2f}|{bbox.y_max:.2f}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]
