from __future__ import annotations

from typing import Any

from app.quantia_spatialV1.prompts.level_localization import (
    build_level_localization_prompt,
)
from app.quantia_spatialV1.providers.vision import (
    GeminiSpatialVisionProvider,
    SpatialVisionProviderError,
)
from app.quantia_spatialV1.transport.gemini_level_localization import (
    GeminiLevelLocalizationResponse,
    get_gemini_level_localization_schema,
)
from pydantic import ValidationError

# ============================================================
# ERROR DE LOCALIZACIÓN
# ============================================================


class GeminiLevelLocalizationServiceError(RuntimeError):
    """
    Error interno de Fase 1 relacionado exclusivamente
    con la localización visual de niveles.
    """


# ============================================================
# SERVICIO
# ============================================================


class GeminiLevelLocalizationService:
    """
    Conecta:

        Level raster
            ↓
        prompt de localización
            ↓
        GeminiSpatialVisionProvider
            ↓
        GeminiLevelLocalizationResponse

    NO:

        - identifica muros;
        - identifica espacios;
        - convierte px -> m;
        - modifica LevelView;
        - importa legacy;
        - publica contrato 03.2 -> 04.
    """

    def __init__(
        self,
        *,
        provider: GeminiSpatialVisionProvider | None = None,
    ) -> None:
        self.provider = provider or GeminiSpatialVisionProvider()

    # ========================================================
    # API PRINCIPAL
    # ========================================================

    def localize(
        self,
        *,
        page_number: int,
        expected_level_names: list[str],
        raster_bytes: bytes,
        raster_mime_type: str,
    ) -> GeminiLevelLocalizationResponse:
        """
        Localiza niveles esperados dentro de una página raster.
        """

        expected_names = self._normalize_expected_names(expected_level_names)

        prompt = build_level_localization_prompt(
            page_number=page_number,
            expected_level_names=expected_names,
        )

        schema = get_gemini_level_localization_schema()

        try:
            vision_result = self.provider.analyze(
                prompt=prompt,
                media_bytes=raster_bytes,
                media_mime_type=raster_mime_type,
                response_json_schema=schema,
            )

        except SpatialVisionProviderError as exc:
            raise GeminiLevelLocalizationServiceError(
                "Falló el provider Gemini durante la localización de niveles."
            ) from exc

        data = vision_result.data

        if not isinstance(
            data,
            dict,
        ):
            raise GeminiLevelLocalizationServiceError(
                "Gemini no devolvió un objeto JSON para la localización de niveles."
            )

        try:
            response = GeminiLevelLocalizationResponse.model_validate(data)

        except ValidationError as exc:
            raise GeminiLevelLocalizationServiceError(
                "La respuesta Gemini no cumple el contrato de localización de niveles."
            ) from exc

        self._validate_page_number(
            expected_page_number=page_number,
            response=response,
        )

        self._validate_expected_levels(
            expected_level_names=expected_names,
            response=response,
        )

        return response

    # ========================================================
    # NOMBRES ESPERADOS
    # ========================================================

    @classmethod
    def _normalize_expected_names(
        cls,
        values: list[str],
    ) -> list[str]:

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

        if not result:
            raise GeminiLevelLocalizationServiceError(
                "No existen niveles esperados válidos para localizar."
            )

        return result

    # ========================================================
    # VALIDACIÓN DE PÁGINA
    # ========================================================

    @staticmethod
    def _validate_page_number(
        *,
        expected_page_number: int,
        response: GeminiLevelLocalizationResponse,
    ) -> None:

        if response.pagina != expected_page_number:
            raise GeminiLevelLocalizationServiceError(
                "Gemini respondió para una página distinta. "
                f"Esperada={expected_page_number}, "
                f"recibida={response.pagina}."
            )

    # ========================================================
    # VALIDACIÓN DE NIVELES
    # ========================================================

    @classmethod
    def _validate_expected_levels(
        cls,
        *,
        expected_level_names: list[str],
        response: GeminiLevelLocalizationResponse,
    ) -> None:
        """
        El prompt exige exactamente una respuesta por cada
        nivel esperado y prohíbe agregar o renombrar niveles.

        Se toleran únicamente diferencias de:
            - mayúsculas/minúsculas;
            - espacios repetidos.
        """

        expected_keys = [cls._name_key(name) for name in expected_level_names]

        received_keys = [cls._name_key(level.nombre) for level in response.niveles]

        if len(received_keys) != len(expected_keys):
            raise GeminiLevelLocalizationServiceError(
                "Gemini no devolvió exactamente una entrada por cada nivel esperado."
            )

        if set(received_keys) != set(expected_keys):
            raise GeminiLevelLocalizationServiceError(
                "Gemini agregó, eliminó o renombró niveles respecto a la solicitud."
            )

    # ========================================================
    # CLAVE TEXTUAL
    # ========================================================

    @staticmethod
    def _name_key(
        value: str,
    ) -> str:

        return " ".join(str(value or "").strip().casefold().split())


# ============================================================
# FACTORY
# ============================================================


def get_gemini_level_localization_service() -> GeminiLevelLocalizationService:
    return GeminiLevelLocalizationService()
