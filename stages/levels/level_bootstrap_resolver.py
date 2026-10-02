from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.ai.schemas.gemini_extraction import (
    GeminiLevelDiscoveryResponse,
)
from app.quantia_spatialV1.core.models.project_site_context import (
    ProjectSiteContext,
)


BootstrapSource = Literal[
    "USER_DECLARED",
    "DOCUMENT_METADATA",
    "PDF_VECTOR",
    "PROJECT_CONTEXT",
    "GEMINI",
    "UNRESOLVED",
]

BootstrapState = Literal["RESOLVED", "REVIEW", "UNRESOLVED"]


class PageLevelBootstrapDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "LEVEL_BOOTSTRAP_V1"
    page_number: int = Field(ge=1)
    known_level_names: list[str] = Field(default_factory=list)
    metadata_level_candidates: list[str] = Field(default_factory=list)
    single_level_isolated: bool = False
    fallback_full_page: bool = False
    fallback_level_name: str | None = None
    state: BootstrapState
    sources: list[BootstrapSource] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LevelBootstrapResolver:
    """Resuelve bootstrap F01 sin hardcodes por proyecto o archivo.

    Orden de evidencia:
    1. declaraciones explícitas del caller;
    2. marcadores vectoriales del PDF;
    3. Gemini Discovery;
    4. contexto de proyecto;
    5. metadata documental conservadora;
    6. unresolved.

    La metadata del nombre de archivo nunca confirma por sí sola un level_name
    único. Puede habilitar un fallback de página completa en estado inferido o,
    si contiene varios niveles, aportar candidatos para que la localización
    visual los valide.
    """

    TRACE_ID = "QSV1-CONT-20261002-A"

    def resolve_page(
        self,
        *,
        page_number: int,
        page_count: int,
        explicit_level_names: list[str] | None = None,
        vector_level_names: list[str] | None = None,
        discovery: GeminiLevelDiscoveryResponse | None = None,
        project_site_context: ProjectSiteContext | None = None,
        source_file_name: str | None = None,
        explicitly_isolated: bool = False,
    ) -> PageLevelBootstrapDecision:
        if page_number <= 0 or page_count <= 0 or page_number > page_count:
            raise ValueError("Página inválida para LevelBootstrapResolver.")

        explicit = self._unique_names(explicit_level_names or [])
        vector = self._unique_names(vector_level_names or [])
        gemini = self._unique_names(
            discovery.to_known_level_names() if discovery is not None else []
        )
        metadata = (
            self._metadata_level_candidates(source_file_name)
            if page_count == 1
            else []
        )

        known: list[str] = []
        sources: list[BootstrapSource] = []
        reasons: list[str] = []
        warnings: list[str] = []

        if explicit:
            known = self._merge_names(known, explicit)
            sources.append("USER_DECLARED")
            reasons.append("explicit_level_names")

        if vector:
            known = self._merge_names(known, vector)
            sources.append("PDF_VECTOR")
            reasons.append("pdf_vector_level_markers")

        if gemini:
            known = self._merge_names(known, gemini)
            sources.append("GEMINI")
            reasons.append("gemini_discovery_named_levels")

        fallback_full_page = False
        fallback_level_name: str | None = None
        single_level_isolated = bool(explicitly_isolated)

        if explicitly_isolated:
            if "USER_DECLARED" not in sources:
                sources.append("USER_DECLARED")
            reasons.append("explicitly_isolated_page")

        stronger_names = bool(known)

        if metadata:
            if stronger_names:
                matched = [
                    candidate
                    for candidate in metadata
                    if self._name_key(candidate)
                    in {self._name_key(name) for name in known}
                ]
                if matched:
                    sources.append("DOCUMENT_METADATA")
                    reasons.append("document_metadata_corrobora_nivel")
                else:
                    warnings.append(
                        "La metadata del archivo propone nivel(es) distintos "
                        "a la evidencia explícita/vectorial/Gemini; no se usan "
                        "como verdad de nivel."
                    )
            elif len(metadata) > 1:
                known = self._merge_names(known, metadata)
                sources.append("DOCUMENT_METADATA")
                reasons.append(
                    "document_metadata_multiple_candidates_require_localization"
                )
            elif len(metadata) == 1:
                sources.append("DOCUMENT_METADATA")
                reasons.append(
                    "document_metadata_single_candidate_not_published_as_level_name"
                )
                if page_count == 1:
                    fallback_full_page = True
                    fallback_level_name = None

        if (
            discovery is not None
            and discovery.total_visible_level_count == 1
            and page_count == 1
        ):
            if "GEMINI" not in sources:
                sources.append("GEMINI")
            fallback_full_page = True
            reasons.append(
                "gemini_single_named_level"
                if discovery.has_named_levels
                else "gemini_single_unnamed_level"
            )

        if (
            project_site_context is not None
            and project_site_context.level_count == 1
            and page_count == 1
        ):
            if "PROJECT_CONTEXT" not in sources:
                sources.append("PROJECT_CONTEXT")
            if len(known) > 1 or (
                discovery is not None
                and discovery.total_visible_level_count > 1
            ):
                fallback_full_page = False
                warnings.append(
                    "El proyecto declara un nivel, pero la evidencia documental "
                    "propone múltiples plantas; no se aplica fallback de página completa."
                )
                reasons.append("project_level_count_conflict")
            else:
                fallback_full_page = True
                reasons.append("project_declares_single_level")

        # Un nombre único respaldado por evidencia fuerte puede usarse como
        # nombre del fallback solo cuando ya existe evidencia independiente.
        if fallback_full_page and len(known) == 1:
            fallback_level_name = known[0]

        strong_name_source = any(
            source in {"USER_DECLARED", "PDF_VECTOR", "GEMINI"}
            for source in sources
        )

        if single_level_isolated:
            state: BootstrapState = "RESOLVED"
        elif known and strong_name_source and not fallback_full_page:
            state = "RESOLVED"
        elif known or fallback_full_page:
            state = "REVIEW"
        else:
            state = "UNRESOLVED"
            sources.append("UNRESOLVED")
            reasons.append("insufficient_bootstrap_evidence")

        return PageLevelBootstrapDecision(
            page_number=page_number,
            known_level_names=known,
            metadata_level_candidates=metadata,
            single_level_isolated=single_level_isolated,
            fallback_full_page=fallback_full_page,
            fallback_level_name=fallback_level_name,
            state=state,
            sources=self._ordered_sources(sources),
            reasons=self._unique_text(reasons),
            warnings=self._unique_text(warnings),
        )

    @classmethod
    def _metadata_level_candidates(
        cls,
        source_file_name: str | None,
    ) -> list[str]:
        raw = str(source_file_name or "").strip()
        if not raw:
            return []

        stem = Path(raw).stem
        normalized = cls._plain_text(stem)
        tokens = set(normalized.split())
        candidates: list[str] = []

        if (
            re.search(r"\bplanta\s+baja\b", normalized)
            or "plantabaja" in tokens
            or "pb" in tokens
        ):
            candidates.append("Planta Baja")
        if (
            re.search(r"\bplanta\s+alta\b", normalized)
            or "plantaalta" in tokens
            or "pa" in tokens
        ):
            candidates.append("Planta Alta")
        if re.search(r"\bsemi\s*sotano\b", normalized) or "semisotano" in tokens:
            candidates.append("Semisótano")
        elif re.search(r"\bsotano\b", normalized):
            candidates.append("Sótano")
        if re.search(r"\bazotea\b", normalized):
            candidates.append("Azotea")

        for match in re.finditer(r"\bnivel\s+(\d+)\b", normalized):
            candidates.append(f"Nivel {int(match.group(1))}")
        for match in re.finditer(r"\bplanta\s+(\d+)\b", normalized):
            candidates.append(f"Planta {int(match.group(1))}")

        return cls._unique_names(candidates)

    @classmethod
    def _unique_names(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            name = str(value or "").strip()
            if not name:
                continue
            key = cls._name_key(name)
            if key in seen:
                continue
            seen.add(key)
            result.append(name)
        return result

    @classmethod
    def _merge_names(cls, first: list[str], second: list[str]) -> list[str]:
        return cls._unique_names([*first, *second])

    @classmethod
    def _name_key(cls, value: str) -> str:
        normalized = cls._plain_text(value)
        aliases = {
            "pb": "planta baja",
            "pa": "planta alta",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _plain_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
        normalized = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _ordered_sources(values: list[BootstrapSource]) -> list[BootstrapSource]:
        order = [
            "USER_DECLARED",
            "PDF_VECTOR",
            "GEMINI",
            "PROJECT_CONTEXT",
            "DOCUMENT_METADATA",
            "UNRESOLVED",
        ]
        present = set(values)
        return [value for value in order if value in present]

    @staticmethod
    def _unique_text(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
