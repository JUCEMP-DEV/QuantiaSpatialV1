from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from typing import Any

from app.quantia_spatialV1.models.level_view import (
    LevelIdentificationState,
    LevelViewEvidence,
    PixelBBox,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ============================================================
# BBOX NORMALIZADO — ENTRADA GEMINI
# ============================================================


class NormalizedBBox(BaseModel):
    """
    Bounding box normalizado recibido desde Gemini.

    Sistema:
        0.0 -> 1.0

    No representa:
        - metros;
        - coordenadas PDF;
        - geometría arquitectónica confirmada.
    """

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    x_max: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_max: float = Field(
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_bbox(
        self,
    ) -> "NormalizedBBox":

        if self.x_max <= self.x_min:
            raise ValueError("x_max debe ser mayor que x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("y_max debe ser mayor que y_min.")

        return self


# ============================================================
# LOCALIZACIÓN GEMINI
# ============================================================


class GeminiLevelLocalization(BaseModel):
    """
    Representación interna mínima de la respuesta actual
    de localización de nivel de Gemini.

    Los campos adicionales como `espacios` se ignoran porque
    Fase 1 solo necesita identificar y localizar niveles.
    """

    model_config = ConfigDict(extra="ignore")

    nombre: str

    localizado: bool

    bbox_normalizado: NormalizedBBox | None = None

    confianza: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    escala_declarada: str | None = None

    confianza_escala: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# RESULTADO DE DETECCIÓN DE UNA REGIÓN
# ============================================================


class LevelRegionDetection(BaseModel):
    """
    Región propuesta para un nivel dentro de una página.

    Este objeto todavía NO contiene raster recortado.

    Esa responsabilidad pertenece a:

        level_view_builder.py
    """

    level_name: str | None = None

    source_page_number: int

    bbox_px: PixelBBox | None = None

    state: LevelIdentificationState

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    evidence: list[LevelViewEvidence] = Field(default_factory=list)

    @property
    def is_buildable(self) -> bool:
        """
        Una región puede convertirse en LevelView cuando:

        - posee bbox raster válido;
        - no presenta conflicto;
        - no fue declarada NO_IDENTIFICADA.

        CANDIDATO es válido porque una planta aislada puede
        conocerse geométricamente aunque todavía no tenga
        nombre de nivel.
        """

        return self.bbox_px is not None and self.state in {
            "DETECTADO",
            "INFERIDO",
            "CANDIDATO",
        }


# ============================================================
# RESULTADO DE FASE 1 — DETECTOR
# ============================================================


class LevelDetectionResult(BaseModel):
    source_page_number: int

    source_page_width_px: int

    source_page_height_px: int

    levels: list[LevelRegionDetection] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    @property
    def buildable_levels(
        self,
    ) -> list[LevelRegionDetection]:

        return [level for level in self.levels if level.is_buildable]


# ============================================================
# DETECTOR
# ============================================================


class LevelDetector:
    """
    FASE 1 — IDENTIFICACIÓN DE NIVEL.

    Responsabilidad:

        determinar cuántos niveles existen
                +
        determinar dónde se encuentran
                ↓
        LevelRegionDetection[]

    Puede combinar:

        extracción semántica
        PyMuPDF level markers
        Gemini level localization
        declaración de planta aislada

    NO:

        - recorta imágenes;
        - ejecuta Gemini;
        - ejecuta PyMuPDF;
        - ejecuta OpenCV;
        - detecta muros;
        - interpreta espacios;
        - convierte px -> m;
        - depende del motor spatial legacy.
    """

    # ========================================================
    # API PRINCIPAL
    # ========================================================

    def detect(
        self,
        *,
        source_page_number: int,
        source_page_width_px: int,
        source_page_height_px: int,
        known_level_names: list[str] | None = None,
        pdf_level_markers: list[str] | None = None,
        gemini_payload: dict[str, Any] | None = None,
        single_level_isolated: bool = False,
    ) -> LevelDetectionResult:
        """
        Detecta niveles dentro de una página raster.

        known_level_names:
            niveles conocidos mediante extracción semántica.

        pdf_level_markers:
            nombres detectados explícitamente en texto vectorial,
            por ejemplo:

                Planta Baja
                Planta Alta
                PB
                PA

        gemini_payload:
            respuesta actual del localizador Gemini:

                {
                    "pagina": 1,
                    "niveles": [...]
                }

        single_level_isolated:
            debe ser True únicamente cuando la entrada ya está
            declarada como una única planta aislada.

            En ese caso puede utilizarse toda la página como
            LevelView sin inventar una subdivisión.
        """

        self._validate_page(
            source_page_number=source_page_number,
            source_page_width_px=source_page_width_px,
            source_page_height_px=source_page_height_px,
        )

        known_names = self._ordered_unique_names(known_level_names or [])

        marker_names = self._ordered_unique_names(pdf_level_markers or [])

        expected_names = self._ordered_unique_names(
            [
                *known_names,
                *marker_names,
            ]
        )

        gemini_levels = self._parse_gemini_payload(
            payload=gemini_payload,
            source_page_number=(source_page_number),
        )

        warnings: list[str] = []

        # ----------------------------------------------------
        # INCONSISTENCIA DE ENTRADA
        # ----------------------------------------------------

        if single_level_isolated and len(expected_names) > 1:
            raise ValueError(
                "single_level_isolated=True es "
                "incompatible con más de un nivel "
                "identificado."
            )

        # ----------------------------------------------------
        # TENEMOS NIVELES ESPERADOS
        # ----------------------------------------------------

        if expected_names:
            levels = self._resolve_expected_levels(
                expected_names=expected_names,
                known_names=known_names,
                marker_names=marker_names,
                gemini_levels=gemini_levels,
                source_page_number=(source_page_number),
                source_page_width_px=(source_page_width_px),
                source_page_height_px=(source_page_height_px),
                single_level_isolated=(single_level_isolated),
                warnings=warnings,
            )

        # ----------------------------------------------------
        # NO TENEMOS NIVELES ESPERADOS
        # ----------------------------------------------------

        else:
            levels = self._resolve_without_expected_names(
                gemini_levels=gemini_levels,
                source_page_number=(source_page_number),
                source_page_width_px=(source_page_width_px),
                source_page_height_px=(source_page_height_px),
                single_level_isolated=(single_level_isolated),
                warnings=warnings,
            )

        # ----------------------------------------------------
        # CONFLICTOS ESPACIALES DETERMINÍSTICOS
        # ----------------------------------------------------

        self._mark_exact_bbox_conflicts(
            levels=levels,
            warnings=warnings,
        )

        return LevelDetectionResult(
            source_page_number=(source_page_number),
            source_page_width_px=(source_page_width_px),
            source_page_height_px=(source_page_height_px),
            levels=levels,
            warnings=warnings,
        )

    # ========================================================
    # RESOLUCIÓN CON NIVELES ESPERADOS
    # ========================================================

    def _resolve_expected_levels(
        self,
        *,
        expected_names: list[str],
        known_names: list[str],
        marker_names: list[str],
        gemini_levels: list[GeminiLevelLocalization],
        source_page_number: int,
        source_page_width_px: int,
        source_page_height_px: int,
        single_level_isolated: bool,
        warnings: list[str],
    ) -> list[LevelRegionDetection]:

        grouped_gemini = self._group_gemini_levels(gemini_levels)

        levels: list[LevelRegionDetection] = []

        for level_name in expected_names:
            key = self._normalize_level_key(level_name)

            matches = grouped_gemini.get(
                key,
                [],
            )

            base_evidence = self._name_evidence(
                level_name=level_name,
                known_names=known_names,
                marker_names=marker_names,
                source_page_number=(source_page_number),
            )

            # ------------------------------------------------
            # MÁS DE UNA LOCALIZACIÓN PARA EL MISMO NIVEL
            # ------------------------------------------------

            if len(matches) > 1:
                evidence = list(base_evidence)

                for match in matches:
                    evidence.extend(
                        self._gemini_evidence(
                            candidate=match,
                            source_page_number=(source_page_number),
                            source_page_width_px=(source_page_width_px),
                            source_page_height_px=(source_page_height_px),
                        )
                    )

                levels.append(
                    LevelRegionDetection(
                        level_name=(level_name),
                        source_page_number=(source_page_number),
                        bbox_px=None,
                        state="CONFLICTO",
                        confidence=(self._max_confidence(matches)),
                        evidence=evidence,
                    )
                )

                warnings.append(
                    "Gemini devolvió más de una "
                    "localización para el nivel "
                    f"'{level_name}'."
                )

                continue

            # ------------------------------------------------
            # UNA LOCALIZACIÓN GEMINI
            # ------------------------------------------------

            if len(matches) == 1:
                candidate = matches[0]

                evidence = [
                    *base_evidence,
                    *self._gemini_evidence(
                        candidate=candidate,
                        source_page_number=(source_page_number),
                        source_page_width_px=(source_page_width_px),
                        source_page_height_px=(source_page_height_px),
                    ),
                ]

                if candidate.localizado and candidate.bbox_normalizado is not None:
                    bbox_px = self._normalized_to_pixel_bbox(
                        bbox=(candidate.bbox_normalizado),
                        page_width_px=(source_page_width_px),
                        page_height_px=(source_page_height_px),
                    )

                    levels.append(
                        LevelRegionDetection(
                            level_name=(level_name),
                            source_page_number=(source_page_number),
                            bbox_px=bbox_px,
                            state="DETECTADO",
                            confidence=(candidate.confianza),
                            evidence=evidence,
                        )
                    )

                    continue

                # --------------------------------------------
                # UNA PLANTA YA AISLADA
                # --------------------------------------------

                if single_level_isolated:
                    levels.append(
                        self._build_full_page_region(
                            level_name=(level_name),
                            source_page_number=(source_page_number),
                            source_page_width_px=(source_page_width_px),
                            source_page_height_px=(source_page_height_px),
                            evidence=evidence,
                            confidence=(candidate.confianza),
                        )
                    )

                    continue

                # --------------------------------------------
                # NIVEL CONOCIDO PERO NO LOCALIZADO
                # --------------------------------------------

                levels.append(
                    LevelRegionDetection(
                        level_name=level_name,
                        source_page_number=(source_page_number),
                        bbox_px=None,
                        state=("NO_IDENTIFICADO"),
                        confidence=(candidate.confianza),
                        evidence=evidence,
                    )
                )

                continue

            # ------------------------------------------------
            # SIN LOCALIZACIÓN GEMINI
            # ------------------------------------------------

            if single_level_isolated:
                levels.append(
                    self._build_full_page_region(
                        level_name=level_name,
                        source_page_number=(source_page_number),
                        source_page_width_px=(source_page_width_px),
                        source_page_height_px=(source_page_height_px),
                        evidence=base_evidence,
                        confidence=None,
                    )
                )

                continue

            levels.append(
                LevelRegionDetection(
                    level_name=level_name,
                    source_page_number=(source_page_number),
                    bbox_px=None,
                    state="NO_IDENTIFICADO",
                    confidence=None,
                    evidence=base_evidence,
                )
            )

        # ----------------------------------------------------
        # GEMINI NO PUEDE AGREGAR SILENCIOSAMENTE NIVELES
        # ----------------------------------------------------

        expected_keys = {self._normalize_level_key(name) for name in expected_names}

        unexpected = [
            level.nombre
            for level in gemini_levels
            if (self._normalize_level_key(level.nombre) not in expected_keys)
        ]

        if unexpected:
            warnings.append(
                "Gemini devolvió niveles que "
                "no estaban en la lista "
                "semántica/documental esperada: " + ", ".join(unexpected)
            )

        return levels

    # ========================================================
    # RESOLUCIÓN SIN NIVELES ESPERADOS
    # ========================================================

    def _resolve_without_expected_names(
        self,
        *,
        gemini_levels: list[GeminiLevelLocalization],
        source_page_number: int,
        source_page_width_px: int,
        source_page_height_px: int,
        single_level_isolated: bool,
        warnings: list[str],
    ) -> list[LevelRegionDetection]:

        # ----------------------------------------------------
        # GEMINI TIENE PROPUESTAS DE NIVEL
        # ----------------------------------------------------

        if gemini_levels:
            grouped = self._group_gemini_levels(gemini_levels)

            levels: list[LevelRegionDetection] = []

            for candidates in grouped.values():
                # --------------------------------------------
                # DUPLICADOS
                # --------------------------------------------

                if len(candidates) > 1:
                    display_name = candidates[0].nombre

                    evidence: list[LevelViewEvidence] = []

                    for candidate in candidates:
                        evidence.extend(
                            self._gemini_evidence(
                                candidate=(candidate),
                                source_page_number=(source_page_number),
                                source_page_width_px=(source_page_width_px),
                                source_page_height_px=(source_page_height_px),
                            )
                        )

                    levels.append(
                        LevelRegionDetection(
                            level_name=(display_name),
                            source_page_number=(source_page_number),
                            bbox_px=None,
                            state="CONFLICTO",
                            confidence=(self._max_confidence(candidates)),
                            evidence=evidence,
                        )
                    )

                    warnings.append(
                        f"Gemini devolvió múltiples entradas para '{display_name}'."
                    )

                    continue

                candidate = candidates[0]

                evidence = self._gemini_evidence(
                    candidate=candidate,
                    source_page_number=(source_page_number),
                    source_page_width_px=(source_page_width_px),
                    source_page_height_px=(source_page_height_px),
                )

                # --------------------------------------------
                # LOCALIZADO
                # --------------------------------------------

                if candidate.localizado and candidate.bbox_normalizado is not None:
                    levels.append(
                        LevelRegionDetection(
                            level_name=(candidate.nombre),
                            source_page_number=(source_page_number),
                            bbox_px=(
                                self._normalized_to_pixel_bbox(
                                    bbox=(candidate.bbox_normalizado),
                                    page_width_px=(source_page_width_px),
                                    page_height_px=(source_page_height_px),
                                )
                            ),
                            state="DETECTADO",
                            confidence=(candidate.confianza),
                            evidence=evidence,
                        )
                    )

                # --------------------------------------------
                # NO LOCALIZADO
                # --------------------------------------------

                else:
                    levels.append(
                        LevelRegionDetection(
                            level_name=(candidate.nombre),
                            source_page_number=(source_page_number),
                            bbox_px=None,
                            state=("NO_IDENTIFICADO"),
                            confidence=(candidate.confianza),
                            evidence=evidence,
                        )
                    )

            return levels

        # ----------------------------------------------------
        # PLANTA AISLADA, PERO NIVEL AÚN SIN NOMBRE
        # ----------------------------------------------------

        if single_level_isolated:
            full_bbox = PixelBBox(
                x_min=0,
                y_min=0,
                x_max=(source_page_width_px),
                y_max=(source_page_height_px),
            )

            warnings.append(
                "La página fue declarada como "
                "una planta aislada, pero todavía "
                "no existe evidencia para nombrar "
                "el nivel."
            )

            return [
                LevelRegionDetection(
                    level_name=None,
                    source_page_number=(source_page_number),
                    bbox_px=full_bbox,
                    state="CANDIDATO",
                    confidence=None,
                    evidence=[
                        LevelViewEvidence(
                            source=("phase_01_input"),
                            reference=("single_level_isolated"),
                            page_number=(source_page_number),
                            bbox_px=full_bbox,
                            confidence=None,
                        )
                    ],
                )
            ]

        # ----------------------------------------------------
        # SIN EVIDENCIA
        # ----------------------------------------------------

        warnings.append(
            "No existe evidencia suficiente para identificar niveles en la página."
        )

        return []

    # ========================================================
    # VALIDACIÓN DE PÁGINA
    # ========================================================

    @staticmethod
    def _validate_page(
        *,
        source_page_number: int,
        source_page_width_px: int,
        source_page_height_px: int,
    ) -> None:

        if source_page_number <= 0:
            raise ValueError("source_page_number debe ser mayor que cero.")

        if source_page_width_px <= 0:
            raise ValueError("source_page_width_px debe ser mayor que cero.")

        if source_page_height_px <= 0:
            raise ValueError("source_page_height_px debe ser mayor que cero.")

    # ========================================================
    # PAYLOAD GEMINI
    # ========================================================

    @staticmethod
    def _parse_gemini_payload(
        *,
        payload: dict[str, Any] | None,
        source_page_number: int,
    ) -> list[GeminiLevelLocalization]:

        if payload is None:
            return []

        page = payload.get("pagina")

        if page is not None:
            try:
                payload_page = int(page)

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError("gemini_payload.pagina debe ser un entero.") from exc

            if payload_page != source_page_number:
                raise ValueError(
                    "La página del payload Gemini no coincide con source_page_number."
                )

        raw_levels = payload.get(
            "niveles",
            [],
        )

        if not isinstance(
            raw_levels,
            list,
        ):
            raise ValueError("gemini_payload.niveles debe ser una lista.")

        return [GeminiLevelLocalization.model_validate(item) for item in raw_levels]

    # ========================================================
    # AGRUPACIÓN GEMINI
    # ========================================================

    @classmethod
    def _group_gemini_levels(
        cls,
        levels: list[GeminiLevelLocalization],
    ) -> dict[
        str,
        list[GeminiLevelLocalization],
    ]:

        grouped: dict[
            str,
            list[GeminiLevelLocalization],
        ] = defaultdict(list)

        for level in levels:
            grouped[cls._normalize_level_key(level.nombre)].append(level)

        return dict(grouped)

    # ========================================================
    # NORMALIZACIÓN DE NOMBRES
    # ========================================================

    @classmethod
    def _ordered_unique_names(
        cls,
        values: list[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            name = str(value or "").strip()

            if not name:
                continue

            key = cls._normalize_level_key(name)

            if key in seen:
                continue

            seen.add(key)

            result.append(name)

        return result

    @staticmethod
    def _normalize_level_key(
        value: str,
    ) -> str:

        normalized = unicodedata.normalize(
            "NFKD",
            str(value or ""),
        )

        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )

        normalized = normalized.strip().lower().replace("_", " ").replace("-", " ")

        normalized = " ".join(normalized.split())

        aliases = {
            "pb": "planta baja",
            "pa": "planta alta",
        }

        return aliases.get(
            normalized,
            normalized,
        )

    # ========================================================
    # NORMALIZED -> RASTER PX
    # ========================================================

    @staticmethod
    def _normalized_to_pixel_bbox(
        *,
        bbox: NormalizedBBox,
        page_width_px: int,
        page_height_px: int,
    ) -> PixelBBox:
        """
        Convierte coordenadas normalizadas Gemini
        directamente al raster conocido.

        Esto NO es conversión px -> m.

        Se conserva completamente el área propuesta:
            min -> floor
            max -> ceil
        """

        x_min = max(
            0,
            min(
                page_width_px - 1,
                math.floor(bbox.x_min * page_width_px),
            ),
        )

        y_min = max(
            0,
            min(
                page_height_px - 1,
                math.floor(bbox.y_min * page_height_px),
            ),
        )

        x_max = max(
            1,
            min(
                page_width_px,
                math.ceil(bbox.x_max * page_width_px),
            ),
        )

        y_max = max(
            1,
            min(
                page_height_px,
                math.ceil(bbox.y_max * page_height_px),
            ),
        )

        if x_max <= x_min:
            raise ValueError("El bbox normalizado produce un ancho raster inválido.")

        if y_max <= y_min:
            raise ValueError("El bbox normalizado produce un alto raster inválido.")

        return PixelBBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    # ========================================================
    # EVIDENCIA DE NOMBRE
    # ========================================================

    @classmethod
    def _name_evidence(
        cls,
        *,
        level_name: str,
        known_names: list[str],
        marker_names: list[str],
        source_page_number: int,
    ) -> list[LevelViewEvidence]:

        key = cls._normalize_level_key(level_name)

        evidence: list[LevelViewEvidence] = []

        if any(cls._normalize_level_key(name) == key for name in known_names):
            evidence.append(
                LevelViewEvidence(
                    source=("semantic_extraction"),
                    reference=("known_level_name"),
                    text=level_name,
                    page_number=(source_page_number),
                    bbox_px=None,
                    confidence=None,
                )
            )

        if any(cls._normalize_level_key(name) == key for name in marker_names):
            evidence.append(
                LevelViewEvidence(
                    source=("pymupdf_vector_text"),
                    reference=("level_marker"),
                    text=level_name,
                    page_number=(source_page_number),
                    bbox_px=None,
                    confidence=None,
                )
            )

        return evidence

    # ========================================================
    # EVIDENCIA GEMINI
    # ========================================================

    @classmethod
    def _gemini_evidence(
        cls,
        *,
        candidate: GeminiLevelLocalization,
        source_page_number: int,
        source_page_width_px: int,
        source_page_height_px: int,
    ) -> list[LevelViewEvidence]:

        bbox_px: PixelBBox | None = None

        if candidate.localizado and candidate.bbox_normalizado is not None:
            bbox_px = cls._normalized_to_pixel_bbox(
                bbox=(candidate.bbox_normalizado),
                page_width_px=(source_page_width_px),
                page_height_px=(source_page_height_px),
            )

        evidence = [
            LevelViewEvidence(
                source="gemini",
                reference=("level_localization"),
                text=candidate.nombre,
                page_number=(source_page_number),
                bbox_px=bbox_px,
                confidence=(candidate.confianza),
            )
        ]

        scale_text = str(candidate.escala_declarada or "").strip()
        if scale_text:
            evidence.append(
                LevelViewEvidence(
                    source="gemini",
                    reference="declared_scale_hint",
                    text=scale_text,
                    page_number=source_page_number,
                    bbox_px=None,
                    confidence=candidate.confianza_escala,
                )
            )

        return evidence

    # ========================================================
    # PÁGINA COMPLETA
    # ========================================================

    @staticmethod
    def _build_full_page_region(
        *,
        level_name: str,
        source_page_number: int,
        source_page_width_px: int,
        source_page_height_px: int,
        evidence: list[LevelViewEvidence],
        confidence: float | None,
    ) -> LevelRegionDetection:
        """
        Caso explícito:

            una planta aislada
                ->
            toda la página pertenece a ese nivel.

        No se introduce margen ni recorte artificial.
        """

        bbox = PixelBBox(
            x_min=0,
            y_min=0,
            x_max=(source_page_width_px),
            y_max=(source_page_height_px),
        )

        return LevelRegionDetection(
            level_name=level_name,
            source_page_number=(source_page_number),
            bbox_px=bbox,
            state="DETECTADO",
            confidence=confidence,
            evidence=[
                *evidence,
                LevelViewEvidence(
                    source=("phase_01_input"),
                    reference=("single_level_isolated"),
                    text=level_name,
                    page_number=(source_page_number),
                    bbox_px=bbox,
                    confidence=None,
                ),
            ],
        )

    # ========================================================
    # CONFIANZA
    # ========================================================

    @staticmethod
    def _max_confidence(
        candidates: list[GeminiLevelLocalization],
    ) -> float | None:

        values = [
            candidate.confianza
            for candidate in candidates
            if candidate.confianza is not None
        ]

        if not values:
            return None

        return max(values)

    # ========================================================
    # CONFLICTOS ESPACIALES
    # ========================================================

    @staticmethod
    def _mark_exact_bbox_conflicts(
        *,
        levels: list[LevelRegionDetection],
        warnings: list[str],
    ) -> None:
        """
        Detecta únicamente un conflicto determinístico:

        dos niveles diferentes con exactamente
        el mismo bbox.

        No utiliza tolerancias inventadas ni umbrales
        arbitrarios de overlap.
        """

        grouped: dict[
            tuple[
                int,
                int,
                int,
                int,
            ],
            list[LevelRegionDetection],
        ] = defaultdict(list)

        for level in levels:
            if level.bbox_px is None:
                continue

            bbox = level.bbox_px

            grouped[
                (
                    bbox.x_min,
                    bbox.y_min,
                    bbox.x_max,
                    bbox.y_max,
                )
            ].append(level)

        for same_bbox_levels in grouped.values():
            if len(same_bbox_levels) <= 1:
                continue

            distinct_keys = {
                LevelDetector._normalize_level_key(level.level_name or "")
                for level in same_bbox_levels
            }

            if len(distinct_keys) <= 1:
                continue

            names = [
                level.level_name or "NO_IDENTIFICADO" for level in same_bbox_levels
            ]

            for level in same_bbox_levels:
                level.state = "CONFLICTO"

            warnings.append(
                "Dos o más niveles diferentes "
                "comparten exactamente la misma "
                "región raster: " + ", ".join(names)
            )
