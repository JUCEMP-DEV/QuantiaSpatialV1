from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.adaptive_reconstruction.contracts import SingleLineWallGraph


class CanonicalWallGraphIntegrity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_count: int = Field(ge=0)
    logical_gap_count: int = Field(ge=0)
    invalid_gap_reference_count: int = Field(ge=0)
    duplicate_wall_id_count: int = Field(ge=0)
    selected_wall_count_matches: bool
    role_counts_match: bool
    interior_space_count_matches: bool
    valid: bool


class ReconstructionEvidenceBundle(BaseModel):
    """Snapshot autónomo y serializable de trazabilidad de reconstrucción.

    No sustituye el WallGraph operativo. Conserva evidencia y decisiones que
    explican cómo se llegó al grafo canónico final sin reinyectar esa evidencia
    en consumidores posteriores.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = "RECONSTRUCTION_EVIDENCE_BUNDLE_V2"
    level_view_id: str
    level_name: str | None = None
    source_document_id: str | None = None
    source_page_number: int
    source_bbox_px: dict[str, int]
    source_raster_sha256: str
    raw_evidence: list[dict[str, Any]] = Field(default_factory=list)
    f03_seed_candidates: list[dict[str, Any]] = Field(default_factory=list)
    discovered_candidates: list[dict[str, Any]] = Field(default_factory=list)
    quarantined_candidates: list[dict[str, Any]] = Field(default_factory=list)
    hybrid_candidates: list[dict[str, Any]] = Field(default_factory=list)
    context_decisions: list[dict[str, Any]] = Field(default_factory=list)
    context_regions: list[dict[str, Any]] = Field(default_factory=list)
    adaptive_walls: list[dict[str, Any]] = Field(default_factory=list)
    adaptive_logical_gaps: list[dict[str, Any]] = Field(default_factory=list)
    post_filter_decisions: list[dict[str, Any]] = Field(default_factory=list)
    architectural_elements: list[dict[str, Any]] = Field(default_factory=list)
    excluded_graphics: list[dict[str, Any]] = Field(default_factory=list)
    unresolved: list[dict[str, Any]] = Field(default_factory=list)
    canonical_mapping: list[dict[str, Any]] = Field(default_factory=list)
    final_wall_ids: list[str] = Field(default_factory=list)
    final_logical_gap_ids: list[str] = Field(default_factory=list)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    preserved_artifacts: dict[str, Any] = Field(default_factory=dict)
    final_source_mapping: dict[str, list[str]] = Field(default_factory=dict)
    correction_review: dict[str, Any] | None = None


class CanonicalWallGraphResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "CANONICAL_WALLGRAPH_FINALIZER_V2"
    graph: SingleLineWallGraph
    integrity: CanonicalWallGraphIntegrity
    evidence_bundle: ReconstructionEvidenceBundle
    audit: dict[str, Any] = Field(default_factory=dict)
