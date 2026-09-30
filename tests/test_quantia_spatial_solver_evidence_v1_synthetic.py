from __future__ import annotations

from app.quantia_spatialV1.reconstruction_core.candidate_models import WallCandidate, WallEvidenceVector
from app.quantia_spatialV1.reconstruction_core.context_models import (
    CandidateContextDecision,
    CandidateContextGateResult,
)
from app.quantia_spatialV1.reconstruction_core.drawing_model import DrawingPoint
from app.quantia_spatialV1.reconstruction_core.global_topology_solver import GlobalTopologySolver


def _candidate(*, candidate_id: str, vector: float, prior: float = 0.52) -> WallCandidate:
    return WallCandidate(
        id=candidate_id,
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=10.0, y=10.0),
        end=DrawingPoint(x=210.0, y=10.0),
        angle_deg=0.0,
        length_px=200.0,
        thickness_px=8.0,
        face_ids=[f"{candidate_id}_A", f"{candidate_id}_B"],
        evidence_ids=[],
        source_names=["PYMUPDF"] if vector else ["LSD_LOCAL"],
        evidence=WallEvidenceVector(
            pair_overlap=0.96,
            thickness_support=0.84,
            vector_support=vector,
            raster_line_support=1.0,
            region_support=0.90,
            source_consensus=0.50,
            perimeter_containment=1.0,
            semantic_support=0.0,
            axis_support=0.55,
            dashed_penalty=0.0,
        ),
        prior_score=prior,
    )


def test_vector_support_no_cambia_identidad_de_muro() -> None:
    raster_only = _candidate(candidate_id="R", vector=0.0)
    vector_backed = _candidate(candidate_id="V", vector=1.0)
    solver = GlobalTopologySolver()
    assert solver._wall_identity_probability(raster_only) == solver._wall_identity_probability(vector_backed)


def test_prior_bajo_no_convierte_geometria_fuerte_en_weak_evidence() -> None:
    candidate = _candidate(candidate_id="W", vector=0.0, prior=0.42)
    solver = GlobalTopologySolver()
    # Con dos conexiones estructurales, una geometría de muro fuerte debe poder
    # competir aunque no tenga vector PDF independiente.
    assert solver._unary(candidate, topology_degree=2) > 0.0


def test_review_con_riesgo_reduce_score_sin_excluir() -> None:
    candidate = _candidate(candidate_id="Q", vector=1.0, prior=0.78)
    decision = CandidateContextDecision(
        candidate_id="Q",
        state="REVIEW",
        reason="AMBIGUOUS_REPETITIVE_CONTEXT_REGION",
        region_id="CTX_1",
        region_type="REPETITIVE_GRID_REGION",
        region_confidence=0.90,
        spatial_containment=1.0,
        face_pattern_support=0.75,
        orientation_match=1.0,
        spacing_match=0.90,
        context_risk=0.81,
    )
    gate = CandidateContextGateResult(
        active_candidates=[],
        quarantined_candidates=[],
        review_candidates=[candidate],
        decisions=[decision],
    )
    annotated = gate.solver_candidates[0]
    solver = GlobalTopologySolver()
    assert annotated.metadata["context_state"] == "REVIEW"
    assert annotated.metadata["context_risk"] == 0.81
    assert solver._context_penalty(annotated) > 0.0
    assert solver._unary(annotated, topology_degree=2) < solver._unary(candidate, topology_degree=2)


def test_active_no_recibe_penalizacion_contextual() -> None:
    candidate = _candidate(candidate_id="A", vector=0.0)
    decision = CandidateContextDecision(
        candidate_id="A",
        state="ACTIVE",
        reason="NO_CONTEXT_REGION_MATCH",
    )
    gate = CandidateContextGateResult(
        active_candidates=[candidate],
        decisions=[decision],
    )
    annotated = gate.solver_candidates[0]
    solver = GlobalTopologySolver()
    assert annotated.metadata["context_state"] == "ACTIVE"
    assert solver._context_penalty(annotated) == 0.0
