from __future__ import annotations

from typing import Any

from app.quantia_spatialV1.models.level_view import (
    LevelView,
)
from app.quantia_spatialV1.phase_01_level.gemini_level_localization_service import (
    GeminiLevelLocalizationService,
    GeminiLevelLocalizationServiceError,
)
from app.quantia_spatialV1.phase_01_level.level_identification_service import (
    LevelIdentificationResult,
    LevelIdentificationService,
)
from app.quantia_spatialV1.phase_01_level.pymupdf_level_source import (
    PyMuPDFLevelSource,
    PyMuPDFLevelSourceResult,
    PyMuPDFSourcePage,
)
from pydantic import BaseModel, Field

# ============================================================
# RESULTADO DOCUMENTAL DE FASE 1
# ============================================================


class PDFLevelIdentificationResult(BaseModel):
    """
    Resultado de identificación de niveles para un PDF completo.

    Una página puede producir:

        0 LevelView
        1 LevelView
        N LevelView
    """

    source_document_id: str | None = None

    page_count: int

    pages: list[LevelIdentificationResult] = Field(default_factory=list)

    level_views: list[LevelView] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    @property
    def level_count(self) -> int:
        return len(self.level_views)

    @property
    def unresolved_count(self) -> int:
        return sum(page.unresolved_count for page in self.pages)


# ============================================================
# SERVICIO PDF -> LEVEL VIEW
# ============================================================


class PDFLevelIdentificationService:
    """
    Orquestador documental de Fase 1.

    Flujo:

        PDF
          ↓
        PyMuPDF
          ↓
        raster + marcadores vectoriales
          ↓
        decisión de localización
          ├── evidencia suficiente -> no Gemini
          └── localización necesaria -> Gemini
          ↓
        LevelIdentificationService
          ↓
        LevelView[]

    Casos principales:

    1. PDF multipágina y una planta identificada por página:
       PyMuPDF es suficiente.
       Se utiliza la página completa.

    2. Página explícitamente declarada como planta aislada:
       no se utiliza Gemini.

    3. Dos o más plantas en una página:
       Gemini localiza las regiones.

    4. PDF de una sola página con nivel conocido pero sin
       confirmación de que sea una planta aislada:
       Gemini localiza la región.

    5. Sin nombre de nivel conocido:
       Gemini no puede ejecutarse porque el prompt de Fase 1
       requiere niveles esperados.
       El resultado permanece pendiente/no identificado.

    NO:

        - identifica muros;
        - identifica espacios;
        - convierte px -> m;
        - importa legacy;
        - publica contrato 03.2 -> 04.
    """

    def __init__(
        self,
        *,
        pdf_source: PyMuPDFLevelSource | None = None,
        identification_service: LevelIdentificationService | None = None,
        localization_service: GeminiLevelLocalizationService | None = None,
    ) -> None:

        self.pdf_source = pdf_source or PyMuPDFLevelSource()

        self.identification_service = (
            identification_service or LevelIdentificationService()
        )

        self.localization_service = (
            localization_service or GeminiLevelLocalizationService()
        )

    # ========================================================
    # API PRINCIPAL
    # ========================================================

    def identify_pdf(
        self,
        *,
        document_bytes: bytes,
        render_scale: float,
        source_document_id: str | None = None,
        known_level_names_by_page: (dict[int, list[str]] | None) = None,
        gemini_payloads_by_page: (dict[int, dict[str, Any]] | None) = None,
        isolated_pages: (set[int] | None) = None,
    ) -> PDFLevelIdentificationResult:
        """
        Ejecuta la identificación completa de niveles
        para todas las páginas del PDF.

        gemini_payloads_by_page permite:

            - replay;
            - diagnóstico;
            - pruebas;
            - reutilizar una localización ya existente.

        Si no existe payload previo, el servicio decide
        automáticamente si necesita ejecutar Gemini.
        """

        source_result = self.pdf_source.read(
            document_bytes=document_bytes,
            render_scale=render_scale,
        )

        known_by_page = known_level_names_by_page or {}

        supplied_gemini_by_page = gemini_payloads_by_page or {}

        explicitly_isolated = isolated_pages or set()

        self._validate_page_keys(
            page_count=source_result.page_count,
            known_level_names_by_page=(known_by_page),
            gemini_payloads_by_page=(supplied_gemini_by_page),
            isolated_pages=(explicitly_isolated),
        )

        page_results: list[LevelIdentificationResult] = []

        all_level_views: list[LevelView] = []

        warnings = list(source_result.warnings)

        for source_page in source_result.pages:
            page_number = source_page.page_number

            # ------------------------------------------------
            # 1. EVIDENCIA VECTORIAL
            # ------------------------------------------------

            vector_level_names = self._unique_marker_names(source_page)

            # ------------------------------------------------
            # 2. NIVELES ESPERADOS
            # ------------------------------------------------

            known_level_names = self._merge_level_names(
                known_by_page.get(
                    page_number,
                    [],
                ),
                vector_level_names,
            )

            # ------------------------------------------------
            # 3. ¿LA PÁGINA YA ES UNA PLANTA AISLADA?
            # ------------------------------------------------

            single_level_isolated = self._resolve_isolated_page(
                source_result=(source_result),
                known_level_names=(known_level_names),
                explicitly_isolated=(page_number in explicitly_isolated),
            )

            # ------------------------------------------------
            # 4. PAYLOAD GEMINI PREEXISTENTE
            # ------------------------------------------------

            gemini_payload = supplied_gemini_by_page.get(page_number)

            # ------------------------------------------------
            # 5. GEMINI AUTOMÁTICO CUANDO REALMENTE HACE FALTA
            # ------------------------------------------------

            if gemini_payload is None and self._should_localize_with_gemini(
                source_result=source_result,
                known_level_names=known_level_names,
                single_level_isolated=(single_level_isolated),
            ):
                try:
                    localization = self.localization_service.localize(
                        page_number=(page_number),
                        expected_level_names=(known_level_names),
                        raster_bytes=(source_page.raster_bytes),
                        raster_mime_type=(source_page.raster_mime_type),
                    )

                    gemini_payload = localization.to_detector_payload()

                except GeminiLevelLocalizationServiceError as exc:
                    # Gemini es evidencia, no una excusa para
                    # fabricar geometría.
                    #
                    # La página continúa hacia LevelDetector,
                    # que conservará los niveles como
                    # NO_IDENTIFICADO si no existe bbox válido.

                    warnings.append(
                        f"Página {page_number}: "
                        "Gemini no pudo localizar los niveles. "
                        f"{exc}"
                    )

            # ------------------------------------------------
            # 6. RESOLUCIÓN FINAL DE LA PÁGINA
            # ------------------------------------------------

            page_result = self.identification_service.identify_page(
                source_raster_bytes=(source_page.raster_bytes),
                source_page_number=(page_number),
                source_document_id=(source_document_id),
                known_level_names=(known_level_names),
                pdf_level_markers=(vector_level_names),
                gemini_payload=(gemini_payload),
                single_level_isolated=(single_level_isolated),
            )

            page_results.append(page_result)

            all_level_views.extend(page_result.level_views)

            warnings.extend(
                self._page_warnings(
                    page_number=page_number,
                    warnings=(page_result.warnings),
                )
            )

        self._validate_unique_level_view_ids(all_level_views)

        return PDFLevelIdentificationResult(
            source_document_id=(source_document_id),
            page_count=(source_result.page_count),
            pages=(page_results),
            level_views=(all_level_views),
            warnings=(warnings),
        )

    # ========================================================
    # ¿GEMINI ES NECESARIO?
    # ========================================================

    @staticmethod
    def _should_localize_with_gemini(
        *,
        source_result: PyMuPDFLevelSourceResult,
        known_level_names: list[str],
        single_level_isolated: bool,
    ) -> bool:
        """
        Gemini se utiliza únicamente cuando existe algo
        concreto que localizar.

        NO se ejecuta si:

            - no conocemos ningún nivel;
            - la página ya está clasificada como planta aislada;
            - el PDF multipágina tiene exactamente un nivel
              identificado en esa página.

        En los demás casos sí necesitamos localización visual.
        """

        if not known_level_names:
            return False

        if single_level_isolated:
            return False

        if source_result.page_count > 1 and len(known_level_names) == 1:
            return False

        return True

    # ========================================================
    # MARCADORES VECTORIALES
    # ========================================================

    @staticmethod
    def _unique_marker_names(
        page: PyMuPDFSourcePage,
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for marker in page.level_markers:
            key = marker.normalized_text

            if key in seen:
                continue

            seen.add(key)

            result.append(marker.text)

        return result

    # ========================================================
    # FUSIÓN DE NOMBRES
    # ========================================================

    @staticmethod
    def _merge_level_names(
        first: list[str],
        second: list[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for value in [
            *first,
            *second,
        ]:
            name = str(value or "").strip()

            if not name:
                continue

            key = name.casefold().replace("_", " ").replace("-", " ")

            key = " ".join(key.split())

            if key in seen:
                continue

            seen.add(key)

            result.append(name)

        return result

    # ========================================================
    # ¿PÁGINA AISLADA?
    # ========================================================

    @staticmethod
    def _resolve_isolated_page(
        *,
        source_result: PyMuPDFLevelSourceResult,
        known_level_names: list[str],
        explicitly_isolated: bool,
    ) -> bool:

        if explicitly_isolated:
            return True

        # PDF multipágina:
        # una sola planta identificada en la página.
        #
        # No existe necesidad de recortar esa página.

        if source_result.page_count > 1 and len(known_level_names) == 1:
            return True

        return False

    # ========================================================
    # VALIDACIÓN DE CONFIGURACIÓN DE PÁGINAS
    # ========================================================

    @staticmethod
    def _validate_page_keys(
        *,
        page_count: int,
        known_level_names_by_page: dict[
            int,
            list[str],
        ],
        gemini_payloads_by_page: dict[
            int,
            dict[str, Any],
        ],
        isolated_pages: set[int],
    ) -> None:

        page_keys = {
            *known_level_names_by_page.keys(),
            *gemini_payloads_by_page.keys(),
            *isolated_pages,
        }

        for page_number in page_keys:
            if not isinstance(
                page_number,
                int,
            ):
                raise ValueError("Las claves de página deben ser enteros.")

            if page_number <= 0 or page_number > page_count:
                raise ValueError(
                    "Se recibió información para una página "
                    f"inexistente: {page_number}."
                )

    # ========================================================
    # IDS
    # ========================================================

    @staticmethod
    def _validate_unique_level_view_ids(
        level_views: list[LevelView],
    ) -> None:

        ids = [level_view.id for level_view in level_views]

        if len(ids) != len(set(ids)):
            raise ValueError("El PDF produjo LevelView con identificadores duplicados.")

    # ========================================================
    # WARNINGS
    # ========================================================

    @staticmethod
    def _page_warnings(
        *,
        page_number: int,
        warnings: list[str],
    ) -> list[str]:

        return [f"Página {page_number}: {warning}" for warning in warnings]


# ============================================================
# FACTORY
# ============================================================


def get_pdf_level_identification_service() -> PDFLevelIdentificationService:
    return PDFLevelIdentificationService()
