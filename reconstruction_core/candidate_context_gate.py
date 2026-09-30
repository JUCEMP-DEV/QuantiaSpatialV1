from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from shapely.geometry import LineString, box

from .candidate_models import WallCandidate
from .context_models import (
    CandidateContextDecision,
    CandidateContextGateResult,
    ContextRegion,
    RepetitiveAxisProfile,
)
from .structural_lineage import StructuralLineage, StructuralLineageAnalyzer


@dataclass(frozen=True)
class _RegionPolicy:
    quarantine_confidence: float
    quarantine_containment: float
    quarantine_orientation: float
    review_confidence: float
    review_containment: float
    review_orientation: float
    review_only: bool = False


class CandidateContextGate:
    """Gate contextual V6 con política específica por clase gráfica.

    Las clases se detectan sin nombres de caso. Cada tipo usa umbrales propios
    porque una escalera, un hatch, una retícula de acabado y una cota no tienen la
    misma firma geométrica. QUARANTINE continúa fuera del CandidateGraph; REVIEW
    permanece como hipótesis con riesgo continuo; WALL_PROTECTED nunca se penaliza.
    """

    VERSION = "F03_SELECTION_CONTRACTS_V1_6"
    ORIENTATION_TOLERANCE_DEG = 6.0
    STRUCTURAL_LINEAGE_RESCUE_TYPES = {
        "FLOOR_FINISH_GRID_REGION",
        "FURNITURE_MODULE_REGION",
        "HATCH_FILL_REGION",
        "STAIR_FLIGHT_REGION",
        "UNKNOWN_REPETITIVE_REGION",
        "REPETITIVE_GRID_REGION",
        "REPETITIVE_PARALLEL_REGION",
    }

    def __init__(self, *, lineage_analyzer: StructuralLineageAnalyzer | None = None) -> None:
        self.lineage_analyzer = lineage_analyzer or StructuralLineageAnalyzer()


    REGION_DECISION_PRIORITY: dict[str, int] = {
        "WALL_PROTECTED_REGION": 100,
        "DOOR_REGION": 96,
        "WINDOW_REGION": 95,
        "OPENING_REGION": 94,
        "COLUMN_REGION": 92,
        "TEXT_SYMBOL_REGION": 91,
        "FLOOR_FINISH_GRID_REGION": 90,
        "STAIR_FLIGHT_REGION": 88,
        "HATCH_FILL_REGION": 86,
        "FURNITURE_MODULE_REGION": 82,
        "SYMBOL_FIXTURE_REGION": 78,
        "AXIS_GRID_REGION": 65,
        "DIMENSION_REGION": 60,
        "UNKNOWN_REPETITIVE_REGION": 40,
        "REPETITIVE_GRID_REGION": 30,
        "REPETITIVE_PARALLEL_REGION": 20,
    }
    REGION_POLICIES: dict[str, _RegionPolicy] = {
        "DOOR_REGION": _RegionPolicy(1.01, 1.01, 1.01, 0.55, 0.35, 0.00, review_only=True),
        "WINDOW_REGION": _RegionPolicy(1.01, 1.01, 1.01, 0.55, 0.35, 0.00, review_only=True),
        "OPENING_REGION": _RegionPolicy(1.01, 1.01, 1.01, 0.55, 0.35, 0.00, review_only=True),
        "COLUMN_REGION": _RegionPolicy(0.82, 0.78, 0.00, 0.64, 0.55, 0.00),
        "TEXT_SYMBOL_REGION": _RegionPolicy(1.01, 1.01, 1.01, 0.70, 0.65, 0.00, review_only=True),
        "STAIR_FLIGHT_REGION": _RegionPolicy(0.74, 0.74, 0.66, 0.60, 0.58, 0.52),
        "FLOOR_FINISH_GRID_REGION": _RegionPolicy(0.72, 0.72, 0.62, 0.58, 0.56, 0.48),
        "HATCH_FILL_REGION": _RegionPolicy(0.76, 0.76, 0.68, 0.62, 0.60, 0.54),
        "AXIS_GRID_REGION": _RegionPolicy(1.01, 1.01, 1.01, 0.62, 0.50, 0.00, review_only=True),
        "DIMENSION_REGION": _RegionPolicy(0.78, 0.68, 0.00, 0.62, 0.52, 0.00),
        "FURNITURE_MODULE_REGION": _RegionPolicy(0.84, 0.80, 0.62, 0.62, 0.60, 0.50),
        "SYMBOL_FIXTURE_REGION": _RegionPolicy(1.01, 1.01, 0.00, 0.64, 0.62, 0.00, review_only=True),
        "UNKNOWN_REPETITIVE_REGION": _RegionPolicy(1.01, 1.01, 1.01, 0.64, 0.62, 0.56, review_only=True),
        # Compatibilidad con evidencia V5 no reclasificada.
        "REPETITIVE_PARALLEL_REGION": _RegionPolicy(1.01, 1.01, 1.01, 0.66, 0.64, 0.58, review_only=True),
        "REPETITIVE_GRID_REGION": _RegionPolicy(0.78, 0.78, 0.68, 0.64, 0.62, 0.56),
    }

    def apply(
        self,
        *,
        candidates: Sequence[WallCandidate],
        regions: Sequence[ContextRegion],
    ) -> CandidateContextGateResult:
        """Clasifica contexto primero y calcula linaje después.

        Selection Contracts V1.2 evita el ciclo de autoconfirmación de V6.2:
        una línea de una escalera/retícula ya no puede convertirse en ancla de otra
        línea del mismo patrón antes de conocer su propia clasificación contextual.
        """
        candidates = list(candidates)
        base_decisions: dict[str, CandidateContextDecision] = {}

        # 1) Verdad contextual local SIN rescate estructural.
        for candidate in candidates:
            evaluations = [
                self._evaluate(candidate=candidate, region=region)
                for region in regions
            ]
            evaluations = [item for item in evaluations if item is not None]
            if evaluations:
                best = max(evaluations, key=self._decision_rank)
            else:
                best = CandidateContextDecision(
                    candidate_id=candidate.id,
                    state="ACTIVE",
                    reason="NO_CONTEXT_REGION_MATCH",
                )

            # El riesgo contextual no desaparece solo porque una regla local deje
            # provisionalmente ACTIVE al candidato. PROTECTED sí queda en cero.
            if best.state != "PROTECTED" and best.region_id is not None:
                best = best.model_copy(update={"context_risk": self._context_risk(best)})
            base_decisions[candidate.id] = best

        # 2) El linaje solo puede usar anclas cuya clasificación base ya es conocida.
        lineage_map = self.lineage_analyzer.analyze(
            candidates,
            decisions=base_decisions,
        )

        active: list[WallCandidate] = []
        quarantined: list[WallCandidate] = []
        review: list[WallCandidate] = []
        decisions: list[CandidateContextDecision] = []

        for candidate in candidates:
            best = self._apply_structural_lineage_guard(
                decision=base_decisions[candidate.id],
                lineage=lineage_map.get(candidate.id),
            )
            if best.state != "PROTECTED" and best.region_id is not None:
                best = best.model_copy(update={"context_risk": self._context_risk(best)})
            decisions.append(best)

            if best.state == "QUARANTINE":
                quarantined.append(candidate)
            elif best.state == "REVIEW":
                review.append(candidate)
            else:  # ACTIVE o PROTECTED
                active.append(candidate)

        return CandidateContextGateResult(
            active_candidates=active,
            quarantined_candidates=quarantined,
            review_candidates=review,
            decisions=decisions,
            regions=list(regions),
        )

    def _evaluate(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> CandidateContextDecision | None:
        if region.region_type == "WALL_PROTECTED_REGION":
            return self._evaluate_wall_protected(candidate=candidate, region=region)

        candidate_line = self._line(candidate)
        region_geom = box(
            region.bbox.x_min,
            region.bbox.y_min,
            region.bbox.x_max,
            region.bbox.y_max,
        )
        if not candidate_line.intersects(region_geom):
            return None

        containment = candidate_line.intersection(region_geom).length / max(candidate_line.length, 1e-6)
        if containment < 0.30:
            return None

        if region.region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"}:
            return self._evaluate_opening_region(
                candidate=candidate,
                region=region,
                containment=containment,
            )
        if region.region_type in {"COLUMN_REGION", "SYMBOL_FIXTURE_REGION", "TEXT_SYMBOL_REGION"}:
            return self._evaluate_element_symbol_region(
                candidate=candidate,
                region=region,
                containment=containment,
            )

        member_support = self._direct_member_support(candidate=candidate, region=region)
        geometry_support = 0.0
        if region.region_type in {
            "FLOOR_FINISH_GRID_REGION",
            "FURNITURE_MODULE_REGION",
            "HATCH_FILL_REGION",
        }:
            geometry_support = self._pattern_geometry_support(candidate=candidate, region=region)
        # `face_pattern_support` sigue publicando proximidad total para diagnóstico,
        # pero V1.4 distingue pertenencia por identidad de una mera coincidencia
        # geométrica. Solo `member_support` puede probar provenance directa.
        face_support = max(member_support, geometry_support)

        # AXIS/DIMENSION pueden no tener periodic profile. Se clasifican por
        # pertenencia a la región + evidencia de línea de referencia.
        if not region.profiles:
            return self._evaluate_non_profile_region(
                candidate=candidate,
                region=region,
                containment=containment,
                face_support=face_support,
                member_support=member_support,
                geometry_support=geometry_support,
            )

        profile_metrics = [
            self._profile_match(candidate=candidate, profile=profile)
            for profile in region.profiles
        ]
        profile_metrics = [item for item in profile_metrics if item is not None]
        if not profile_metrics:
            metric_support = self._metric_band_support(candidate)
            pair_support = self._pattern_pair_support(candidate=candidate, region=region)
            pair_provenance = self._pattern_pair_provenance(candidate=candidate, region=region)
            boundary_support = self._pattern_boundary_support(candidate=candidate, region=region)
            # Para una región de celdas, una hipótesis perpendicular puede ser el
            # otro borde de la celda. La pertenencia geométrica sigue teniendo
            # prioridad sobre la orientación del perfil.
            if member_support >= 0.50 and region.region_type in {
                "FLOOR_FINISH_GRID_REGION",
                "FURNITURE_MODULE_REGION",
            }:
                # V1.4: el detector de superficies puede incluir una cara real de
                # muro. Pertenecer a member_line_ids tampoco basta para aislarla: si
                # la repetición vive a un solo lado y la banda es físicamente
                # plausible, se conserva como REVIEW incluso cuando pair_support sea
                # alto. La sidedness es evidencia independiente del emparejamiento.
                boundary_wall = (
                    boundary_support >= 0.60
                    and metric_support >= 0.58
                )
                state = (
                    "REVIEW"
                    if boundary_wall or region.region_type == "FURNITURE_MODULE_REGION"
                    else "QUARANTINE"
                )
                return CandidateContextDecision(
                    candidate_id=candidate.id,
                    state=state,
                    reason=(
                        "REVIEW_PATTERN_MEMBER_BOUNDARY_WALL_CANDIDATE"
                        if boundary_wall
                        else "CANDIDATE_USES_CELL_REGION_MEMBERS"
                    ),
                    region_id=region.id,
                    region_type=region.region_type,
                    region_confidence=region.confidence,
                    spatial_containment=max(0.0, min(1.0, containment)),
                    face_pattern_support=max(0.0, min(1.0, face_support)),
                    metadata={
                        "metric_wall_band_support": metric_support,
                        "pattern_pair_support": pair_support,
                        "pattern_pair_provenance": pair_provenance,
                        "member_pattern_support": member_support,
                        "region_boundary_support": boundary_support,
                    },
                )

            # Selection Contracts V1.4: estar dentro de una región 2D NO demuestra
            # pertenencia al patrón. Para aislar una hipótesis perpendicular se exige
            # evidencia directa del patrón. En escaleras, múltiples peldaños que
            # cruzan materialmente el candidato sí constituyen esa evidencia; un muro
            # lateral donde los peldaños TERMINAN queda en REVIEW.
            strong_surface_types = {
                "STAIR_FLIGHT_REGION",
                "FLOOR_FINISH_GRID_REGION",
                "HATCH_FILL_REGION",
            }
            crossing_support = self._pattern_crossing_support(
                candidate=candidate,
                region=region,
            )
            direct_pattern_member = (
                member_support >= 0.50
                or pair_provenance >= 0.72
                or (
                    region.region_type == "STAIR_FLIGHT_REGION"
                    and crossing_support >= 0.65
                )
            )
            if (
                region.region_type in strong_surface_types
                and region.confidence >= 0.78
                and containment >= 0.72
            ):
                boundary_wall = boundary_support >= 0.58 and metric_support >= 0.55
                if boundary_wall:
                    state = "REVIEW"
                    reason = "REVIEW_CONTEXT_BOUNDARY_WALL_CANDIDATE"
                elif direct_pattern_member:
                    state = "QUARANTINE"
                    reason = "QUARANTINE_DIRECT_PATTERN_MEMBER_ORIENTATION_MISMATCH"
                else:
                    # La caja contextual solo expresa riesgo espacial. La hipótesis
                    # conserva acceso al solver para que geometría/topología decidan.
                    state = "REVIEW"
                    reason = "REVIEW_CONTEXT_CONTAINMENT_WITHOUT_PATTERN_MEMBERSHIP"
                return CandidateContextDecision(
                    candidate_id=candidate.id,
                    state=state,
                    reason=reason,
                    region_id=region.id,
                    region_type=region.region_type,
                    region_confidence=region.confidence,
                    spatial_containment=max(0.0, min(1.0, containment)),
                    face_pattern_support=max(0.0, min(1.0, face_support)),
                    metadata={
                        "metric_wall_band_support": metric_support,
                        "pattern_pair_support": pair_support,
                        "pattern_pair_provenance": pair_provenance,
                        "pattern_crossing_support": crossing_support,
                        "member_pattern_support": member_support,
                        "geometry_pattern_support": geometry_support,
                        "direct_pattern_member": direct_pattern_member,
                        "region_boundary_support": boundary_support,
                    },
                )

            risky_2d_types = {
                "STAIR_FLIGHT_REGION",
                "FLOOR_FINISH_GRID_REGION",
                "FURNITURE_MODULE_REGION",
                "HATCH_FILL_REGION",
                "UNKNOWN_REPETITIVE_REGION",
                "REPETITIVE_GRID_REGION",
                "REPETITIVE_PARALLEL_REGION",
            }
            state = (
                "REVIEW"
                if region.region_type in risky_2d_types
                and region.confidence >= 0.58
                and containment >= 0.52
                else "ACTIVE"
            )
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state=state,
                reason=f"{state}_CONTEXT_REGION_ORIENTATION_MISMATCH",
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=max(0.0, min(1.0, containment)),
                face_pattern_support=max(0.0, min(1.0, face_support)),
                metadata={
                    "metric_wall_band_support": metric_support,
                    "pattern_pair_support": pair_support,
                    "pattern_pair_provenance": pair_provenance,
                    "region_boundary_support": boundary_support,
                },
            )

        profile, orientation_match, spacing_match = max(
            profile_metrics,
            key=lambda item: item[1] * 0.45 + item[2] * 0.55,
        )

        length_ratio = candidate.length_px / max(profile.median_length_px, 1e-6)
        pair_support = self._pattern_pair_support(candidate=candidate, region=region)
        pair_provenance = self._pattern_pair_provenance(candidate=candidate, region=region)
        metric_support = self._metric_band_support(candidate)
        boundary_support = self._pattern_boundary_support(candidate=candidate, region=region)
        if (
            pair_support >= 0.72
            and region.region_type in self.STRUCTURAL_LINEAGE_RESCUE_TYPES
            and region.confidence >= 0.68
            and containment >= 0.55
        ):
            # V1.4: estar bracketed por dos pistas es evidencia fuerte de patrón,
            # pero ya no anula evidencia física de que el patrón termina a un solo
            # lado. Ese caso conserva REVIEW para no encapsular un muro de borde.
            boundary_wall = boundary_support >= 0.58 and metric_support >= 0.55
            requires_provenance = region.region_type in {
                "STAIR_FLIGHT_REGION",
                "FLOOR_FINISH_GRID_REGION",
                "FURNITURE_MODULE_REGION",
                "HATCH_FILL_REGION",
            }
            # V1.6: pair_support puede provenir de un perfil inferido. En patrones
            # funcionales capaces de solaparse con muros reales, QUARANTINE exige
            # pertenencia o bracket trazable a segmentos reales del patrón.
            proven_pattern = (
                member_support >= 0.45
                or pair_provenance >= 0.72
                or not requires_provenance
            )
            state = "REVIEW" if boundary_wall or not proven_pattern else "QUARANTINE"
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state=state,
                reason=(
                    f"REVIEW_BRACKETED_BOUNDARY_WALL_OVER_{region.region_type}"
                    if boundary_wall
                    else (
                        f"REVIEW_BRACKETED_WITHOUT_DIRECT_MEMBERSHIP_{region.region_type}"
                        if not proven_pattern
                        else f"CANDIDATE_BRACKETED_BY_{region.region_type}_TRACKS"
                    )
                ),
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                orientation_match=orientation_match,
                spacing_match=spacing_match,
                metadata={
                    "length_ratio": length_ratio,
                    "pattern_pair_support": pair_support,
                    "pattern_pair_provenance": pair_provenance,
                    "metric_wall_band_support": metric_support,
                    "region_boundary_support": boundary_support,
                    "member_pattern_support": member_support,
                    "proven_pattern_membership": proven_pattern,
                },
            )

        continuation_outside = max(0.0, 1.0 - containment)
        clear_continuation = length_ratio >= 1.55 and continuation_outside >= 0.14
        if clear_continuation:
            metric_support = self._metric_band_support(candidate)
            pair_support = self._pattern_pair_support(candidate=candidate, region=region)
            pair_provenance = self._pattern_pair_provenance(candidate=candidate, region=region)
            boundary_support = self._pattern_boundary_support(candidate=candidate, region=region)
            risky_types = self.STRUCTURAL_LINEAGE_RESCUE_TYPES
            if region.region_type in risky_types:
                # Salir del bbox no basta: una línea de escalón/baldosa puede
                # extenderse fuera de una micro-región fragmentada. ACTIVE requiere
                # banda física plausible + salida material + apoyo sobre el borde.
                trustworthy = (
                    continuation_outside >= 0.25
                    and metric_support >= 0.60
                    and boundary_support >= 0.35
                    and pair_support < 0.45
                )
                state = "ACTIVE" if trustworthy else "REVIEW"
            else:
                state = "ACTIVE"
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state=state,
                reason=(
                    "CANDIDATE_CONTINUES_OUTSIDE_CONTEXT_REGION"
                    if state == "ACTIVE"
                    else "REVIEW_WEAK_EXTERNAL_CONTINUATION"
                ),
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                orientation_match=orientation_match,
                spacing_match=spacing_match,
                metadata={
                    "length_ratio": length_ratio,
                    "metric_wall_band_support": metric_support,
                    "pattern_pair_support": pair_support,
                    "pattern_pair_provenance": pair_provenance,
                    "region_boundary_support": boundary_support,
                    "continuation_outside": continuation_outside,
                },
            )

        policy = self.REGION_POLICIES.get(
            region.region_type,
            self.REGION_POLICIES["UNKNOWN_REPETITIVE_REGION"],
        )
        pattern_derived = member_support >= 0.45
        spatially_derived = (
            spacing_match >= 0.68
            and orientation_match >= 0.72
            and containment >= 0.74
            and length_ratio <= 1.60
        )

        # `metric_support` y `boundary_support` ya fueron calculados antes de la
        # rama de pair_support. La pertenencia espacial sola no puede producir
        # QUARANTINE: solo pertenencia directa al patrón.
        boundary_wall = (
            region.region_type in self.STRUCTURAL_LINEAGE_RESCUE_TYPES
            and boundary_support >= 0.62
            and metric_support >= 0.58
        )
        requires_identity = region.region_type in {
            "FLOOR_FINISH_GRID_REGION",
            "FURNITURE_MODULE_REGION",
            "HATCH_FILL_REGION",
        }
        direct_pattern_member = (
            pattern_derived
            if requires_identity
            else (pattern_derived or pair_provenance >= 0.72)
        )

        can_quarantine = (
            not policy.review_only
            and region.quarantine_enabled
            and region.confidence >= policy.quarantine_confidence
            and containment >= policy.quarantine_containment
            and orientation_match >= policy.quarantine_orientation
            and direct_pattern_member
        )
        if can_quarantine and boundary_wall:
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state="REVIEW",
                reason=f"REVIEW_PATTERN_BOUNDARY_WALL_OVER_{region.region_type}",
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                orientation_match=orientation_match,
                spacing_match=spacing_match,
                metadata={
                    "pattern_derived_from_face_ids": pattern_derived,
                    "direct_pattern_member": direct_pattern_member,
                    "spatially_derived": spatially_derived,
                    "length_ratio": length_ratio,
                    "metric_wall_band_support": metric_support,
                    "pattern_pair_support": pair_support,
                    "pattern_pair_provenance": pair_provenance,
                    "region_boundary_support": boundary_support,
                },
            )
        if can_quarantine:
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state="QUARANTINE",
                reason=f"CANDIDATE_DERIVED_FROM_{region.region_type}",
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                orientation_match=orientation_match,
                spacing_match=spacing_match,
                metadata={
                    "pattern_derived_from_face_ids": pattern_derived,
                    "direct_pattern_member": direct_pattern_member,
                    "spatially_derived": spatially_derived,
                    "length_ratio": length_ratio,
                },
            )

        can_review = (
            region.confidence >= policy.review_confidence
            and containment >= policy.review_containment
            and orientation_match >= policy.review_orientation
            and (face_support >= 0.28 or spacing_match >= 0.52 or spatially_derived)
        )
        if can_review:
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state="REVIEW",
                reason=f"AMBIGUOUS_{region.region_type}",
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                orientation_match=orientation_match,
                spacing_match=spacing_match,
                metadata={"length_ratio": length_ratio},
            )

        return CandidateContextDecision(
            candidate_id=candidate.id,
            state="ACTIVE",
            reason="CONTEXT_EVIDENCE_INSUFFICIENT_FOR_ISOLATION",
            region_id=region.id,
            region_type=region.region_type,
            region_confidence=region.confidence,
            spatial_containment=containment,
            face_pattern_support=face_support,
            orientation_match=orientation_match,
            spacing_match=spacing_match,
            metadata={"length_ratio": length_ratio},
        )

    def _evaluate_opening_region(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
        containment: float,
    ) -> CandidateContextDecision:
        """Opening no es ruido: aísla símbolo sin destruir el host wall."""
        direct_member_support = self._direct_member_support(candidate=candidate, region=region)
        host_member_support = self._candidate_lineage_support(
            candidate=candidate,
            target_ids=region.metadata.get("host_line_ids") or [],
        )
        metric_support = self._metric_band_support(candidate)
        candidate_line = self._line(candidate)
        region_geom = box(region.bbox.x_min, region.bbox.y_min, region.bbox.x_max, region.bbox.y_max)
        boundary_overlap = candidate_line.intersection(
            region_geom.boundary.buffer(max(2.0, candidate.thickness_px * 0.8))
        ).length / max(candidate_line.length, 1e-6)
        extends_outside = max(0.0, 1.0 - containment)
        region_span = max(
            1e-6,
            region.bbox.x_max - region.bbox.x_min,
            region.bbox.y_max - region.bbox.y_min,
        )
        strong_structural_span = (
            metric_support >= 0.84
            and (extends_outside >= 0.10 or candidate.length_px >= region_span * 1.18)
        )

        # Host explícito publicado por el detector: nunca se interpreta como
        # símbolo aunque comparta geometría cercana al opening.
        if host_member_support >= 0.50:
            state = "ACTIVE"
            reason = f"ACTIVE_EXPLICIT_HOST_WALL_{region.region_type}"
            host_wall_like = True
        else:
            fully_local_member = (
                direct_member_support >= 0.50
                and containment >= 0.72
                and extends_outside <= 0.28
            )
            detector_verified = bool(region.metadata.get("opening_verified", True))
            if region.region_type == "WINDOW_REGION":
                embedding = float(region.metadata.get("host_embedding_support", 0.0) or 0.0)
                through_support = float(region.metadata.get("through_wall_support", 0.0) or 0.0)
                semantic_support = float(region.metadata.get("semantic_search_support", 0.0) or 0.0)
                detector_verified = detector_verified and (
                    (semantic_support >= 0.55 and embedding >= 0.50 and through_support < 0.42)
                    or (embedding >= 0.76 and through_support < 0.22)
                )

            symbol_like = fully_local_member and detector_verified and not strong_structural_span
            host_wall_like = (
                direct_member_support < 0.50
                and metric_support >= 0.58
                and (extends_outside >= 0.12 or boundary_overlap >= 0.20)
            )

            if strong_structural_span and direct_member_support >= 0.50:
                # Salvaguarda V3: una pertenencia directa no basta para destruir
                # una banda con continuidad física fuerte. El solver decide.
                state = "REVIEW"
                reason = f"REVIEW_DIRECT_{region.region_type}_STRONG_WALL_CONTINUITY"
            elif symbol_like:
                state = "QUARANTINE"
                reason = f"QUARANTINE_DIRECT_{region.region_type}_SYMBOL_MEMBER"
            elif host_wall_like:
                state = "ACTIVE"
                reason = f"ACTIVE_HOST_WALL_THROUGH_{region.region_type}"
            else:
                state = "REVIEW"
                reason = f"REVIEW_AMBIGUOUS_{region.region_type}_CONTEXT"

        return CandidateContextDecision(
            candidate_id=candidate.id,
            state=state,
            reason=reason,
            region_id=region.id,
            region_type=region.region_type,
            region_confidence=region.confidence,
            spatial_containment=containment,
            face_pattern_support=direct_member_support,
            metadata={
                "element_context": True,
                "element_role": "OPENING",
                "direct_element_member_support": direct_member_support,
                "explicit_host_member_support": host_member_support,
                "metric_wall_band_support": metric_support,
                "opening_boundary_overlap": max(0.0, min(1.0, boundary_overlap)),
                "opening_extends_outside": extends_outside,
                "opening_bridge_candidate": host_wall_like,
                "strong_structural_span": strong_structural_span,
                "negative_mask": False,
                **dict(region.metadata),
            },
        )

    def _evaluate_element_symbol_region(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
        containment: float,
    ) -> CandidateContextDecision:
        direct_member_support = self._direct_member_support(candidate=candidate, region=region)
        metric_support = self._metric_band_support(candidate)

        if direct_member_support >= 0.50 and region.quarantine_enabled and metric_support < 0.72:
            state = "QUARANTINE"
            reason = f"QUARANTINE_DIRECT_{region.region_type}_MEMBER"
        elif region.region_type == "TEXT_SYMBOL_REGION":
            # El bbox de texto es localización, no una máscara negativa. Sin
            # provenance directa de la geometría del glifo no se degrada un muro.
            state = "ACTIVE"
            reason = "ACTIVE_TEXT_SYMBOL_CONTEXT_ONLY"
        elif containment >= 0.70 and region.confidence >= 0.70:
            state = "REVIEW"
            reason = f"REVIEW_{region.region_type}_CONTAINMENT"
        else:
            state = "ACTIVE"
            reason = f"ACTIVE_{region.region_type}_CONTEXT"
        return CandidateContextDecision(
            candidate_id=candidate.id,
            state=state,
            reason=reason,
            region_id=region.id,
            region_type=region.region_type,
            region_confidence=region.confidence,
            spatial_containment=containment,
            face_pattern_support=direct_member_support,
            metadata={
                "element_context": True,
                "direct_element_member_support": direct_member_support,
                "metric_wall_band_support": metric_support,
                "negative_mask": False,
                **dict(region.metadata),
            },
        )

    def _evaluate_non_profile_region(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
        containment: float,
        face_support: float,
        member_support: float,
        geometry_support: float,
    ) -> CandidateContextDecision:
        e = candidate.evidence
        if region.region_type == "FLOOR_FINISH_GRID_REGION":
            metric_support = self._metric_band_support(candidate)
            pair_support = self._pattern_pair_support(candidate=candidate, region=region)
            pair_provenance = self._pattern_pair_provenance(candidate=candidate, region=region)
            boundary_support = self._pattern_boundary_support(candidate=candidate, region=region)
            # V5/V1.6: pair_support inferido no demuestra provenance, pero un
            # bracket formado por segmentos publicados del patrón sí aumenta el
            # riesgo contextual sin convertirlo automáticamente en QUARANTINE.
            # provenance. Sin identidad directa del patrón, como máximo REVIEW.
            direct_pattern_member = member_support >= 0.45
            boundary_wall = boundary_support >= 0.58 and metric_support >= 0.55

            # V1.4: la bbox de un acabado es contexto, no una máscara negativa.
            # QUARANTINE requiere demostrar que la hipótesis deriva del patrón.
            # Una hipótesis solamente contenida conserva REVIEW para el solver.
            if region.confidence >= 0.85 and containment >= 0.75:
                if boundary_wall:
                    state = "REVIEW"
                    reason = "REVIEW_FLOOR_REGION_BOUNDARY_WALL_CANDIDATE"
                elif direct_pattern_member and region.quarantine_enabled:
                    state = "QUARANTINE"
                    reason = "QUARANTINE_DIRECT_FLOOR_PATTERN_MEMBER"
                else:
                    state = "REVIEW"
                    reason = "REVIEW_FLOOR_CONTAINMENT_WITHOUT_PATTERN_MEMBERSHIP"
            elif (
                region.quarantine_enabled
                and region.confidence >= 0.70
                and containment >= 0.68
                and direct_pattern_member
                and not boundary_wall
            ):
                state = "QUARANTINE"
                reason = "QUARANTINE_SEGMENTED_FLOOR_PATTERN"
            elif region.confidence >= 0.58 and containment >= 0.56:
                state = "REVIEW"
                reason = "REVIEW_SEGMENTED_FLOOR_PATTERN"
            else:
                state = "ACTIVE"
                reason = "ACTIVE_SEGMENTED_FLOOR_PATTERN"
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state=state,
                reason=reason,
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                metadata={
                    "surface_pattern_support": face_support,
                    "member_pattern_support": member_support,
                    "geometry_pattern_support": geometry_support,
                    "metric_wall_band_support": metric_support,
                    "pattern_pair_support": pair_support,
                    "pattern_pair_provenance": pair_provenance,
                    "direct_pattern_member": direct_pattern_member,
                    "region_boundary_support": boundary_support,
                },
            )

        if region.region_type == "AXIS_GRID_REGION":
            metric_support = self._metric_band_support(candidate)
            dashed_support = max(0.0, min(1.0, float(e.dashed_penalty)))
            direct_reference_support = max(face_support, dashed_support)

            # V1.4: AXIS_GRID_REGION NO es una máscara negativa 2D. axis_support es
            # evidencia positiva de localización del centro de muro y nunca puede
            # usarse para expulsar una banda plausible. Solo se aísla la propia
            # primitiva de eje cuando existe pertenencia directa/dashed + banda poco
            # plausible. El resto permanece ACTIVE o REVIEW sin penalización por bbox.
            exact_reference = (
                direct_reference_support >= 0.60
                and metric_support < 0.55
            ) or (
                dashed_support >= 0.78
                and metric_support < 0.72
            )
            ambiguous_reference = (
                direct_reference_support >= 0.30
                and metric_support < 0.72
            )
            if exact_reference:
                state = "QUARANTINE"
                reason = "QUARANTINE_DIRECT_AXIS_REFERENCE_PRIMITIVE"
            elif ambiguous_reference:
                state = "REVIEW"
                reason = "REVIEW_DIRECT_AXIS_REFERENCE_PRIMITIVE"
            else:
                state = "ACTIVE"
                reason = "ACTIVE_AXIS_LOCATOR_NOT_NEGATIVE_MASK"

            reference_support = direct_reference_support if state != "ACTIVE" else 0.0
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state=state,
                reason=reason,
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                metadata={
                    "reference_support": reference_support,
                    "direct_reference_support": direct_reference_support,
                    "metric_wall_band_support": metric_support,
                    "axis_locator_support": float(e.axis_support),
                    "negative_mask": False,
                },
            )

        if region.region_type == "DIMENSION_REGION":
            reference_support = max(face_support, float(e.dashed_penalty), 1.0 - float(e.region_support))
            if (
                region.quarantine_enabled
                and region.confidence >= 0.78
                and containment >= 0.68
                and reference_support >= 0.50
            ):
                state = "QUARANTINE"
            elif region.confidence >= 0.62 and containment >= 0.52:
                state = "REVIEW"
            else:
                state = "ACTIVE"
            return CandidateContextDecision(
                candidate_id=candidate.id,
                state=state,
                reason=f"{state}_DIMENSION_REGION",
                region_id=region.id,
                region_type=region.region_type,
                region_confidence=region.confidence,
                spatial_containment=containment,
                face_pattern_support=face_support,
                metadata={"reference_support": reference_support},
            )

        # Regiones compactas sin perfil (símbolo/fixture/celda incompleta) nunca
        # se eliminan por bbox solamente. Se marcan REVIEW si la pertenencia es alta.
        state = "REVIEW" if region.confidence >= 0.64 and containment >= 0.64 else "ACTIVE"
        return CandidateContextDecision(
            candidate_id=candidate.id,
            state=state,
            reason=f"{state}_{region.region_type}",
            region_id=region.id,
            region_type=region.region_type,
            region_confidence=region.confidence,
            spatial_containment=containment,
            face_pattern_support=face_support,
        )

    def _evaluate_wall_protected(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> CandidateContextDecision | None:
        segment = region.metadata.get("segment")
        if not isinstance(segment, list) or len(segment) != 2:
            return None
        try:
            wall_line = LineString([
                (float(segment[0][0]), float(segment[0][1])),
                (float(segment[1][0]), float(segment[1][1])),
            ])
        except (TypeError, ValueError, IndexError):
            return None
        candidate_line = self._line(candidate)
        tolerance = float(region.metadata.get("tolerance_px", 4.0) or 4.0)
        angle = float(region.metadata.get("angle_deg", 0.0) or 0.0)
        if self._angle_diff(candidate.angle_deg, angle) > self.ORIENTATION_TOLERANCE_DEG:
            return None
        if candidate_line.distance(wall_line) > tolerance:
            return None
        buffered = wall_line.buffer(tolerance, cap_style=2, join_style=2)
        overlap = candidate_line.intersection(buffered).length / max(candidate_line.length, 1e-6)
        if overlap < 0.60:
            return None
        return CandidateContextDecision(
            candidate_id=candidate.id,
            state="PROTECTED",
            reason="F02_PERIMETER_WALL_PROTECTED",
            region_id=region.id,
            region_type=region.region_type,
            region_confidence=1.0,
            spatial_containment=min(1.0, overlap),
            metadata={"f02_wall_id": region.metadata.get("wall_id")},
        )

    def _profile_match(
        self,
        *,
        candidate: WallCandidate,
        profile: RepetitiveAxisProfile,
    ) -> tuple[RepetitiveAxisProfile, float, float] | None:
        diff = self._angle_diff(candidate.angle_deg, profile.angle_deg)
        if diff > self.ORIENTATION_TOLERANCE_DEG:
            return None
        orientation_match = max(0.0, 1.0 - diff / self.ORIENTATION_TOLERANCE_DEG)

        angle = math.radians(profile.angle_deg)
        nx, ny = -math.sin(angle), math.cos(angle)
        midpoint_x = (candidate.start.x + candidate.end.x) / 2.0
        midpoint_y = (candidate.start.y + candidate.end.y) / 2.0
        cross = midpoint_x * nx + midpoint_y * ny
        spacing_match = self._repetition_coordinate_match(
            cross=cross,
            coordinates=profile.track_coordinates,
            spacing=profile.spacing_px,
        )
        return profile, orientation_match, spacing_match

    @staticmethod
    def _repetition_coordinate_match(
        *,
        cross: float,
        coordinates: Sequence[float],
        spacing: float,
    ) -> float:
        ordered = sorted(coordinates)
        if len(ordered) < 3 or spacing <= 1e-6:
            return 0.0
        if cross < ordered[0] - 0.15 * spacing or cross > ordered[-1] + 0.15 * spacing:
            return 0.0

        midpoint_score = 0.0
        for first, second in zip(ordered[:-1], ordered[1:]):
            gap = second - first
            if gap <= 1e-6 or not (first <= cross <= second):
                continue
            regularity = max(0.0, 1.0 - abs(gap - spacing) / max(0.45 * spacing, 1e-6))
            mid = (first + second) / 2.0
            centered = max(0.0, 1.0 - abs(cross - mid) / max(0.50 * gap, 1e-6))
            midpoint_score = max(midpoint_score, regularity * centered)

        track_score = 0.0
        for index, coordinate in enumerate(ordered):
            if index in {0, len(ordered) - 1}:
                continue
            distance = abs(cross - coordinate)
            local = max(0.0, 1.0 - distance / max(0.28 * spacing, 1e-6))
            track_score = max(track_score, 0.86 * local)

        return max(0.0, min(1.0, max(midpoint_score, track_score)))

    def _apply_structural_lineage_guard(
        self,
        *,
        decision: CandidateContextDecision,
        lineage: StructuralLineage | None,
    ) -> CandidateContextDecision:
        if lineage is None:
            return decision
        metadata = {
            **decision.metadata,
            "structural_lineage_version": StructuralLineageAnalyzer.VERSION,
            "structural_lineage": lineage.as_metadata(),
        }
        if decision.region_type not in self.STRUCTURAL_LINEAGE_RESCUE_TYPES:
            return decision.model_copy(update={"metadata": metadata})

        # El eje por sí solo NO rescata una línea: el propio eje gráfico puede
        # producir axis_support alto. Para salir de QUARANTINE/REVIEW se exige una
        # ancla geométrica independiente (continuación o junction) además del score.
        anchor_support = max(
            lineage.collinear_anchor_support,
            lineage.junction_anchor_support,
        )
        very_strong = lineage.score >= 0.78 and anchor_support >= 0.78

        if decision.state == "QUARANTINE" and lineage.strong:
            state = "ACTIVE" if very_strong else "REVIEW"
            return decision.model_copy(
                update={
                    "state": state,
                    "reason": f"{state}_STRUCTURAL_LINEAGE_OVER_{decision.region_type}",
                    "metadata": metadata,
                }
            )
        if decision.state == "REVIEW" and very_strong:
            return decision.model_copy(
                update={
                    "state": "ACTIVE",
                    "reason": f"ACTIVE_STRUCTURAL_LINEAGE_OVER_{decision.region_type}",
                    "metadata": metadata,
                }
            )
        return decision.model_copy(update={"metadata": metadata})

    @staticmethod
    def _candidate_lineage_support(*, candidate: WallCandidate, target_ids) -> float:
        target = {str(item) for item in target_ids if item}
        if not target:
            return 0.0

        raw_groups = candidate.metadata.get("face_lineage_groups")
        groups: list[set[str]] = []
        if isinstance(raw_groups, list):
            for raw in raw_groups:
                if isinstance(raw, (list, tuple, set)):
                    group = {str(item) for item in raw if item}
                    if group:
                        groups.append(group)

        lineage = {str(item) for item in candidate.face_ids if item}
        lineage.update(str(item) for item in candidate.evidence_ids if item)
        raw_lineage = candidate.metadata.get("lineage_line_ids")
        if isinstance(raw_lineage, (list, tuple, set)):
            lineage.update(str(item) for item in raw_lineage if item)

        if not groups:
            if not lineage:
                return 0.0
            return 1.0 if lineage & target else 0.0

        hits = sum(1 for group in groups if group & target)
        grouped_support = max(0.0, min(1.0, hits / max(1, len(groups))))
        if grouped_support > 0.0:
            return grouped_support
        # Algunos tracks consolidados conservan el RawEvidence en `evidence_ids`
        # aunque sus grupos de caras procedan de otra representación equivalente.
        # Se publica soporte parcial, no 1.0, para no convertir una coincidencia
        # global en pertenencia de todas las caras.
        return 0.50 if lineage & target else 0.0

    @staticmethod
    def _direct_member_support(*, candidate: WallCandidate, region: ContextRegion) -> float:
        """Mide provenance física contra los miembros publicados por la región."""
        return CandidateContextGate._candidate_lineage_support(
            candidate=candidate,
            target_ids=region.member_line_ids,
        )

    def _pattern_geometry_support(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        raw_segments = region.metadata.get("pattern_segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            return 0.0
        candidate_line = self._line(candidate)
        tolerance = max(
            2.5,
            0.75 * float(candidate.thickness_px),
            float(region.metadata.get("member_tolerance_px", 0.0) or 0.0),
        )
        best = 0.0
        for raw in raw_segments:
            if not isinstance(raw, list) or len(raw) != 4:
                continue
            try:
                segment = LineString([(float(raw[0]), float(raw[1])), (float(raw[2]), float(raw[3]))])
            except (TypeError, ValueError):
                continue
            if segment.length <= 1e-6:
                continue
            sx1, sy1 = segment.coords[0]
            sx2, sy2 = segment.coords[-1]
            angle = math.degrees(math.atan2(sy2 - sy1, sx2 - sx1)) % 180.0
            if self._angle_diff(candidate.angle_deg, angle) > self.ORIENTATION_TOLERANCE_DEG:
                continue
            buffered = segment.buffer(tolerance, cap_style=2, join_style=2)
            overlap = candidate_line.intersection(buffered).length / max(candidate_line.length, 1e-6)
            best = max(best, overlap)
        return max(0.0, min(1.0, best))

    @staticmethod
    def _metric_support(candidate: WallCandidate) -> float:
        """Compatibilidad: soporte puramente de espesor físico."""
        raw = candidate.metadata.get("metric_thickness_support")
        if raw is None:
            return 1.0
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _metric_band_support(candidate: WallCandidate) -> float:
        """Soporte físico de banda = espesor + esbeltez, calculado por V1.2."""
        raw = candidate.metadata.get("metric_wall_band_support")
        if raw is None:
            return CandidateContextGate._metric_support(candidate)
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return CandidateContextGate._metric_support(candidate)

    def _pattern_pair_support(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        """Detecta el falso muro formado al emparejar dos pistas del patrón.

        Una retícula o escalera puede producir dos líneas paralelas muy limpias.
        Candidate Discovery hace bien en conservar la hipótesis DOUBLE_FACE, pero
        si el eje candidato queda bracketed por pistas repetitivas a ambos lados y
        la separación coincide con `thickness_px`, la evidencia pertenece al
        patrón, no a una banda arquitectónica independiente.
        """
        offsets = self._parallel_pattern_offsets(candidate=candidate, region=region)
        best = 0.0
        positives = sorted(value for value, _ in offsets if value > 0.0)
        negatives = sorted((-value) for value, _ in offsets if value < 0.0)
        if positives and negatives:
            pos = positives[0]
            neg = negatives[0]
            gap = pos + neg
            target = max(float(candidate.thickness_px), 1e-6)
            thickness_match = max(0.0, 1.0 - abs(gap - target) / max(0.65 * target, 2.0))
            balance = max(0.0, 1.0 - abs(pos - neg) / max(gap, 1e-6))
            best = max(best, thickness_match * balance)

        # Los perfiles aportan una segunda medición aun cuando `pattern_segments`
        # no esté disponible (p. ej. regiones ensambladas desde celdas repetidas).
        for profile in region.profiles:
            if self._angle_diff(candidate.angle_deg, profile.angle_deg) > self.ORIENTATION_TOLERANCE_DEG:
                continue
            angle = math.radians(profile.angle_deg)
            nx, ny = -math.sin(angle), math.cos(angle)
            mx = 0.5 * (candidate.start.x + candidate.end.x)
            my = 0.5 * (candidate.start.y + candidate.end.y)
            cross = mx * nx + my * ny
            coords = sorted(float(value) for value in profile.track_coordinates)
            if len(coords) < 2:
                continue
            for first, second in zip(coords[:-1], coords[1:]):
                if not (first <= cross <= second):
                    continue
                gap = second - first
                if gap <= 1e-6:
                    continue
                target = max(float(candidate.thickness_px), 1e-6)
                thickness_match = max(0.0, 1.0 - abs(gap - target) / max(0.65 * target, 2.0))
                centered = max(0.0, 1.0 - abs(cross - 0.5 * (first + second)) / max(0.45 * gap, 1e-6))
                best = max(best, thickness_match * centered)
                break
        return max(0.0, min(1.0, best))

    def _pattern_pair_provenance(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        """Prueba un falso DOUBLE_FACE usando pistas reales publicadas.

        A diferencia de ``_pattern_pair_support`` no usa perfiles inferidos. Solo
        devuelve soporte cuando ``pattern_segments`` contiene dos pistas reales a
        lados opuestos del candidato, con solape longitudinal y separación
        compatible con su espesor. Se usa como riesgo contextual trazable; no como
        veto por bbox.
        """
        offsets = self._parallel_pattern_offsets(candidate=candidate, region=region)
        positives = sorted((value, overlap) for value, overlap in offsets if value > 0.0)
        negatives = sorted((-value, overlap) for value, overlap in offsets if value < 0.0)
        if not positives or not negatives:
            return 0.0
        pos, pos_overlap = positives[0]
        neg, neg_overlap = negatives[0]
        gap = pos + neg
        target = max(float(candidate.thickness_px), 1e-6)
        thickness_match = max(0.0, 1.0 - abs(gap - target) / max(0.60 * target, 2.0))
        balance = max(0.0, 1.0 - abs(pos - neg) / max(gap, 1e-6))
        overlap = min(pos_overlap, neg_overlap)
        return max(0.0, min(1.0, thickness_match * balance * overlap))

    def _parallel_pattern_offsets(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> list[tuple[float, float]]:
        raw_segments = region.metadata.get("pattern_segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            return []

        line = self._line(candidate)
        angle = math.radians(candidate.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        c_mid_x = 0.5 * (candidate.start.x + candidate.end.x)
        c_mid_y = 0.5 * (candidate.start.y + candidate.end.y)
        c0 = min(candidate.start.x * ux + candidate.start.y * uy, candidate.end.x * ux + candidate.end.y * uy)
        c1 = max(candidate.start.x * ux + candidate.start.y * uy, candidate.end.x * ux + candidate.end.y * uy)
        reach = max(8.0, 3.0 * float(candidate.thickness_px))
        values: list[tuple[float, float]] = []

        for raw in raw_segments:
            if not isinstance(raw, list) or len(raw) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, raw)
            except (TypeError, ValueError):
                continue
            dx, dy = x2 - x1, y2 - y1
            seg_len = math.hypot(dx, dy)
            if seg_len <= 1e-6:
                continue
            seg_angle = math.degrees(math.atan2(dy, dx)) % 180.0
            if self._angle_diff(candidate.angle_deg, seg_angle) > self.ORIENTATION_TOLERANCE_DEG:
                continue
            s0 = min(x1 * ux + y1 * uy, x2 * ux + y2 * uy)
            s1 = max(x1 * ux + y1 * uy, x2 * ux + y2 * uy)
            overlap = max(0.0, min(c1, s1) - max(c0, s0))
            overlap_ratio = overlap / max(min(line.length, seg_len), 1e-6)
            if overlap_ratio < 0.35:
                continue
            s_mid_x = 0.5 * (x1 + x2)
            s_mid_y = 0.5 * (y1 + y2)
            signed = (s_mid_x - c_mid_x) * nx + (s_mid_y - c_mid_y) * ny
            if abs(signed) <= reach:
                values.append((float(signed), float(overlap_ratio)))
        return values

    def _pattern_crossing_support(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        """Mide pistas repetitivas que atraviesan materialmente el candidato.

        Se usa solo como evidencia DIRECTA en regiones de escalera cuando la
        orientación del candidato no coincide con el perfil dominante. Un muro que
        delimita la escalera recibe extremos a un solo lado; una línea interior es
        cruzada por varios peldaños y puede aislarse sin depender del bbox.
        """
        raw_segments = region.metadata.get("pattern_segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            return 0.0
        candidate_line = self._line(candidate)
        if candidate_line.length <= 1e-6:
            return 0.0

        tolerance = max(2.0, 0.35 * float(candidate.thickness_px))
        buffered = candidate_line.buffer(tolerance, cap_style=2, join_style=2)
        hits: list[float] = []
        angle = math.radians(candidate.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        c0 = min(
            candidate.start.x * ux + candidate.start.y * uy,
            candidate.end.x * ux + candidate.end.y * uy,
        )
        c1 = max(
            candidate.start.x * ux + candidate.start.y * uy,
            candidate.end.x * ux + candidate.end.y * uy,
        )
        edge_margin = max(4.0, 0.75 * float(candidate.thickness_px))

        for raw in raw_segments:
            if not isinstance(raw, list) or len(raw) != 4:
                continue
            try:
                segment = LineString([
                    (float(raw[0]), float(raw[1])),
                    (float(raw[2]), float(raw[3])),
                ])
            except (TypeError, ValueError):
                continue
            if segment.length <= 1e-6:
                continue
            sx1, sy1 = segment.coords[0]
            sx2, sy2 = segment.coords[-1]
            seg_angle = math.degrees(math.atan2(sy2 - sy1, sx2 - sx1)) % 180.0
            diff = self._angle_diff(candidate.angle_deg, seg_angle)
            if diff < 60.0:
                continue
            if not segment.intersects(buffered):
                continue
            inter = segment.intersection(buffered)
            point = inter.centroid if not inter.is_empty else None
            if point is None:
                continue
            along = float(point.x) * ux + float(point.y) * uy
            if along <= c0 + edge_margin or along >= c1 - edge_margin:
                continue
            hits.append(along)

        if len(hits) < 2:
            return 0.0
        hits.sort()
        distinct = [hits[0]]
        min_sep = max(3.0, 0.50 * float(candidate.thickness_px))
        for value in hits[1:]:
            if value - distinct[-1] >= min_sep:
                distinct.append(value)
        count_score = min(1.0, len(distinct) / 4.0)
        if len(distinct) >= 2:
            span = distinct[-1] - distinct[0]
            span_score = min(1.0, span / max(0.45 * candidate_line.length, 1e-6))
        else:
            span_score = 0.0
        return max(0.0, min(1.0, 0.72 * count_score + 0.28 * span_score))

    def _pattern_boundary_support(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        """Estima si la hipótesis delimita el patrón en vez de pertenecer a él.

        V1.3 separa dos situaciones: muro paralelo al patrón y muro perpendicular
        que recibe los extremos de las pistas (caso típico de escalera/acabado).
        El bbox queda únicamente como fallback débil; nunca puede por sí solo
        convertir una línea interior en muro.
        """
        # V1.4: pair_support ya no anula el borde. Un muro real puede coincidir
        # con dos trazos del patrón y, aun así, tener evidencia repetitiva a un solo
        # lado. La sidedness se evalúa de forma independiente.
        offsets = self._parallel_pattern_offsets(candidate=candidate, region=region)
        positives = [item for item in offsets if item[0] > 0.0]
        negatives = [item for item in offsets if item[0] < 0.0]
        parallel_one_sided = 0.0
        if bool(positives) ^ bool(negatives):
            side = positives if positives else negatives
            nearest = min(abs(value) for value, _ in side)
            proximity = max(
                0.0,
                1.0 - nearest / max(2.5 * float(candidate.thickness_px), 6.0),
            )
            overlap = max(score for _, score in side)
            parallel_one_sided = 0.55 + 0.45 * min(proximity, overlap)
        elif positives and negatives:
            parallel_one_sided = 0.0

        endpoint_one_sided = self._endpoint_pattern_boundary_support(
            candidate=candidate,
            region=region,
        )
        pattern_support = max(parallel_one_sided, endpoint_one_sided)
        bbox_support = self._bbox_boundary_support(candidate=candidate, region=region)
        if offsets or endpoint_one_sided > 0.0:
            return max(0.0, min(1.0, 0.86 * pattern_support + 0.14 * bbox_support))

        # Sin pistas explícitas el bbox solo conserva una posibilidad REVIEW.
        return 0.60 * bbox_support

    def _endpoint_pattern_boundary_support(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        """Detecta evidencia repetitiva que termina a un solo lado del candidato.

        Esto protege un muro real perpendicular a peldaños/baldosas: el muro no
        comparte orientación con las pistas, pero varias pistas terminan cerca de
        su cara y todas viven del mismo lado. En el interior de una retícula hay
        evidencia a ambos lados, por lo que el soporte cae a cero.
        """
        raw_segments = region.metadata.get("pattern_segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            return 0.0

        line = self._line(candidate)
        if line.length <= 1e-6:
            return 0.0

        angle = math.radians(candidate.angle_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        c_mid_x = 0.5 * (candidate.start.x + candidate.end.x)
        c_mid_y = 0.5 * (candidate.start.y + candidate.end.y)
        c0 = min(
            candidate.start.x * ux + candidate.start.y * uy,
            candidate.end.x * ux + candidate.end.y * uy,
        )
        c1 = max(
            candidate.start.x * ux + candidate.start.y * uy,
            candidate.end.x * ux + candidate.end.y * uy,
        )
        along_margin = max(6.0, 1.5 * float(candidate.thickness_px))
        lateral_reach = max(8.0, 2.8 * float(candidate.thickness_px))

        positive: list[float] = []
        negative: list[float] = []
        for raw in raw_segments:
            if not isinstance(raw, list) or len(raw) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, raw)
            except (TypeError, ValueError):
                continue

            a1 = x1 * ux + y1 * uy
            a2 = x2 * ux + y2 * uy
            seg_lo, seg_hi = min(a1, a2), max(a1, a2)
            if seg_hi < c0 - along_margin or seg_lo > c1 + along_margin:
                continue

            d1 = (x1 - c_mid_x) * nx + (y1 - c_mid_y) * ny
            d2 = (x2 - c_mid_x) * nx + (y2 - c_mid_y) * ny
            nearest = min(abs(d1), abs(d2))
            if nearest > lateral_reach:
                continue

            # Una pista que cruza materialmente la hipótesis no demuestra borde.
            if d1 * d2 < 0.0 and min(abs(d1), abs(d2)) > 0.35 * float(candidate.thickness_px):
                positive.append(0.0)
                negative.append(0.0)
                continue

            signed = 0.5 * (d1 + d2)
            proximity = max(0.0, 1.0 - nearest / lateral_reach)
            if signed > 1e-6:
                positive.append(proximity)
            elif signed < -1e-6:
                negative.append(proximity)

        # Exige más de una pista: un único encuentro puede ser una puerta/símbolo.
        if positive and negative:
            return 0.0
        side = positive if positive else negative
        if len(side) < 2:
            return 0.0
        density = min(1.0, len(side) / 4.0)
        proximity = sum(side) / max(len(side), 1)
        return max(0.0, min(1.0, 0.58 + 0.22 * density + 0.20 * proximity))

    def _region_boundary_support(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        """Alias histórico: en V1.2 el soporte real usa patrón + bbox débil."""
        return self._pattern_boundary_support(candidate=candidate, region=region)

    def _bbox_boundary_support(
        self,
        *,
        candidate: WallCandidate,
        region: ContextRegion,
    ) -> float:
        line = self._line(candidate)
        if line.length <= 1e-6:
            return 0.0
        region_geom = box(
            region.bbox.x_min,
            region.bbox.y_min,
            region.bbox.x_max,
            region.bbox.y_max,
        )
        tolerance = max(4.0, 1.35 * float(candidate.thickness_px))
        near_boundary = line.intersection(
            region_geom.boundary.buffer(tolerance, cap_style=2, join_style=2)
        ).length
        return max(0.0, min(1.0, float(near_boundary / line.length)))

    @staticmethod
    def _context_risk(decision: CandidateContextDecision) -> float:
        # Texto es contexto de visualización, nunca evidencia negativa por bbox.
        # Un host wall explícitamente preservado a través de un opening tampoco
        # debe heredar penalización por encontrarse dentro del símbolo.
        if decision.region_type == "TEXT_SYMBOL_REGION":
            return 0.0
        if (
            decision.region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"}
            and decision.state == "ACTIVE"
            and bool(decision.metadata.get("opening_bridge_candidate"))
        ):
            return 0.0

        # AXIS/DIMENSION suelen abarcar bandas extensas del plano. Estar dentro de
        # su bbox no es evidencia negativa: el riesgo depende de que el candidato
        # sea realmente reference-like.
        if decision.region_type in {"AXIS_GRID_REGION", "DIMENSION_REGION"}:
            raw_reference = decision.metadata.get("reference_support", 0.0)
            try:
                reference_support = max(0.0, min(1.0, float(raw_reference)))
            except (TypeError, ValueError):
                reference_support = 0.0
            risk = decision.region_confidence * decision.spatial_containment * reference_support
            return max(0.0, min(1.0, float(risk)))

        raw_pair_provenance = decision.metadata.get("pattern_pair_provenance", 0.0)
        try:
            pair_provenance = max(0.0, min(1.0, float(raw_pair_provenance)))
        except (TypeError, ValueError):
            pair_provenance = 0.0
        pattern_membership = max(
            decision.face_pattern_support,
            decision.orientation_match * decision.spacing_match,
            pair_provenance,
        )
        if pattern_membership <= 0.0:
            # Para patrones 2D (p. ej. escalera) una orientación perpendicular no
            # elimina el riesgo; se conserva un residual moderado.
            pattern_membership = 0.35
        risk = decision.region_confidence * decision.spatial_containment * pattern_membership
        return max(0.0, min(1.0, float(risk)))

    @classmethod
    def _decision_rank(cls, decision: CandidateContextDecision) -> tuple[int, int, float, float, float]:
        # PROTECTED continúa dominando por provenir de F02 inmutable. Entre
        # regiones con el mismo estado, V6.1 prioriza clases funcionales locales
        # (escalera/acabado/hatch) sobre referencias extensas (ejes/cotas). Así
        # una cota superpuesta no reemplaza una clasificación fuerte de escalera.
        state_rank = {"ACTIVE": 0, "REVIEW": 1, "QUARANTINE": 2, "PROTECTED": 3}[decision.state]
        type_rank = cls.REGION_DECISION_PRIORITY.get(decision.region_type or "", 0)
        return (
            state_rank,
            type_rank,
            decision.region_confidence,
            decision.face_pattern_support,
            decision.spacing_match,
        )

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)

    @staticmethod
    def _line(candidate: WallCandidate) -> LineString:
        return LineString([
            (candidate.start.x, candidate.start.y),
            (candidate.end.x, candidate.end.y),
        ])
