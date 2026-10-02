from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from app.core.config import settings

# ============================================================
# ERRORES
# ============================================================


class SpatialVisionProviderError(RuntimeError):
    """
    Error interno del motor app.quantia_spatialV1.

    No depende de:
        app.services.vision_provider
        ni de ningún servicio legacy.
    """


class _GeminiHTTPError(SpatialVisionProviderError):
    """
    Error HTTP interno que conserva el status code.
    """

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
    ) -> None:
        super().__init__(message)

        self.status_code = int(
            status_code
        )


# ============================================================
# RESULTADO
# ============================================================


@dataclass(frozen=True)
class SpatialVisionResult:
    """
    Resultado bruto de una llamada multimodal.

    `data` permanece sin interpretar arquitectónicamente.
    La validación de transporte corresponde a los modelos
    Pydantic de app.quantia_spatialV1.ai.schemas.
    """

    provider: str

    model: str

    text: str

    data: (
        dict[str, Any]
        | list[Any]
        | None
    )

    fallback_used: bool

    raw: dict[str, Any]


# ============================================================
# PROVIDER GEMINI
# ============================================================


class GeminiSpatialVisionProvider:
    """
    Provider Gemini privado del nuevo motor Quantia Spatial.

    Responsabilidad:

        prompt
        +
        raster
        +
        JSON Schema opcional
            ↓
        Gemini REST
            ↓
        JSON bruto validable posteriormente

    NO:

        - identifica niveles por sí mismo;
        - interpreta arquitectura;
        - convierte px -> m;
        - genera muros;
        - genera espacios;
        - importa servicios legacy;
        - publica contrato 03.2 -> 04.

    Configuración utilizada:

        settings.gemini_api_key
        settings.gemini_primary_model
        settings.gemini_fallback_model
    """

    BASE_URL = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models"
    )

    SUPPORTED_MEDIA_MIME_TYPES = {
        "image/png",
        "image/jpeg",
    }

    TRANSIENT_HTTP_STATUS = {
        408,
        429,
        500,
        502,
        503,
        504,
    }

    FORMAT_HTTP_STATUS = {
        400,
        422,
    }

    MAX_ATTEMPTS_PER_MODEL = max(
        3,
        int(os.getenv("QUANTIA_GEMINI_MAX_ATTEMPTS_PER_MODEL", "4")),
    )

    # 503 "high demand" suele no incluir Retry-After. Un backoff de 1/2 s
    # solo vuelve a golpear la misma ventana de saturacion. Estos valores son
    # deliberadamente mas amplios y configurables sin tocar codigo.
    HIGH_DEMAND_503_DELAYS = (15.0, 30.0, 60.0)

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        *,
        api_key: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
    ) -> None:

        self.api_key = str(
            api_key
            if api_key is not None
            else settings.gemini_api_key
        ).strip()

        self.primary_model = str(
            primary_model
            if primary_model is not None
            else settings.gemini_primary_model
        ).strip()

        self.fallback_model = str(
            fallback_model
            if fallback_model is not None
            else settings.gemini_fallback_model
        ).strip()

        self.models = (
            self._build_model_order()
        )

    # ========================================================
    # API PRINCIPAL
    # ========================================================

    def analyze(
        self,
        *,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
        response_json_schema: (
            dict[str, Any]
            | None
        ) = None,
    ) -> SpatialVisionResult:
        """
        Ejecuta análisis multimodal.

        Con schema:

            1. responseJsonSchema
            2. JSON MIME sin schema
            3. plain

        Un error temporal agota reintentos del modelo actual
        y después permite utilizar el fallback.
        """

        normalized_mime = (
            self._normalize_mime_type(
                media_mime_type
            )
        )

        self._validate_request(
            prompt=prompt,
            media_bytes=media_bytes,
            media_mime_type=normalized_mime,
        )

        output_modes = (
            self._build_output_modes(
                response_json_schema
            )
        )

        all_errors: list[str] = []

        for model_index, model in enumerate(
            self.models
        ):
            fallback_used = (
                model_index > 0
            )

            model_errors: list[str] = []

            for output_mode in output_modes:
                try:
                    return self._generate_once(
                        model=model,
                        prompt=prompt,
                        media_bytes=media_bytes,
                        media_mime_type=(
                            normalized_mime
                        ),
                        response_json_schema=(
                            response_json_schema
                        ),
                        output_mode=(
                            output_mode
                        ),
                        fallback_used=(
                            fallback_used
                        ),
                    )

                except _GeminiHTTPError as exc:
                    model_errors.append(
                        f"{output_mode}: {exc}"
                    )

                    if (
                        exc.status_code
                        in self.TRANSIENT_HTTP_STATUS
                    ):
                        break

                    if (
                        exc.status_code
                        in self.FORMAT_HTTP_STATUS
                    ):
                        continue

                    break

                except SpatialVisionProviderError as exc:
                    model_errors.append(
                        f"{output_mode}: {exc}"
                    )

                    continue

            all_errors.append(
                f"{model}: "
                + " | ".join(
                    model_errors
                )
            )

        raise SpatialVisionProviderError(
            "Todos los modelos Gemini configurados fallaron. "
            + " || ".join(
                all_errors
            )
        )

    # ========================================================
    # MODOS DE RESPUESTA
    # ========================================================

    @staticmethod
    def _build_output_modes(
        response_json_schema: (
            dict[str, Any]
            | None
        ),
    ) -> tuple[str, ...]:

        if response_json_schema:
            return (
                "response_json_schema",
                "json_only",
                "plain",
            )

        return (
            "plain",
        )

    # ========================================================
    # LLAMADA HTTP
    # ========================================================

    def _generate_once(
        self,
        *,
        model: str,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
        response_json_schema: (
            dict[str, Any]
            | None
        ),
        output_mode: str,
        fallback_used: bool,
    ) -> SpatialVisionResult:

        encoded_media = (
            base64.b64encode(
                media_bytes
            ).decode(
                "utf-8"
            )
        )

        payload: dict[
            str,
            Any,
        ] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt,
                        },
                        {
                            "inlineData": {
                                "mimeType": (
                                    media_mime_type
                                ),
                                "data": (
                                    encoded_media
                                ),
                            }
                        },
                    ],
                }
            ]
        }

        generation_config = (
            self._build_generation_config(
                output_mode=(
                    output_mode
                ),
                response_json_schema=(
                    response_json_schema
                ),
            )
        )

        if generation_config:
            payload[
                "generationConfig"
            ] = generation_config

        url = (
            f"{self.BASE_URL}/"
            f"{model}:generateContent"
        )

        headers = {
            "x-goog-api-key": (
                self.api_key
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        response: (
            requests.Response
            | None
        ) = None

        for attempt in range(
            1,
            self.MAX_ATTEMPTS_PER_MODEL + 1,
        ):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

            except requests.RequestException as exc:
                if (
                    attempt
                    == self.MAX_ATTEMPTS_PER_MODEL
                ):
                    raise SpatialVisionProviderError(
                        "No fue posible conectar con Gemini: "
                        f"{exc}"
                    ) from exc

                time.sleep(
                    2 ** (
                        attempt - 1
                    )
                )

                continue

            if response.ok:
                break

            status_code = int(
                response.status_code
            )

            if (
                status_code
                not in self.TRANSIENT_HTTP_STATUS
            ):
                raise _GeminiHTTPError(
                    status_code=(
                        status_code
                    ),
                    message=(
                        self._http_error_message(
                            response
                        )
                    ),
                )

            if (
                attempt
                == self.MAX_ATTEMPTS_PER_MODEL
            ):
                raise _GeminiHTTPError(
                    status_code=(
                        status_code
                    ),
                    message=(
                        self._http_error_message(
                            response
                        )
                    ),
                )

            delay = self._retry_delay_seconds(
                response=response,
                attempt=attempt,
            )

            time.sleep(delay)

        if response is None:
            raise SpatialVisionProviderError(
                "Gemini no produjo una respuesta HTTP."
            )

        try:
            raw = response.json()

        except ValueError as exc:
            raise SpatialVisionProviderError(
                "Gemini devolvió una respuesta HTTP "
                "sin JSON válido."
            ) from exc

        if not isinstance(
            raw,
            dict,
        ):
            raise SpatialVisionProviderError(
                "La raíz de la respuesta Gemini "
                "no es un objeto JSON."
            )

        text = (
            self._extract_text(
                raw
            )
        )

        if not text:
            raise SpatialVisionProviderError(
                self._empty_response_message(
                    raw
                )
            )

        parsed_data: (
            dict[str, Any]
            | list[Any]
            | None
        ) = None

        if response_json_schema:
            parsed_data = (
                self._parse_json(
                    text
                )
            )

            self._validate_required_fields(
                data=parsed_data,
                schema=(
                    response_json_schema
                ),
            )

        return SpatialVisionResult(
            provider="gemini",
            model=model,
            text=text,
            data=parsed_data,
            fallback_used=(
                fallback_used
            ),
            raw=raw,
        )

    # ========================================================
    # RETRY / QUOTA BACKOFF
    # ========================================================

    @classmethod
    def _retry_delay_seconds(
        cls,
        *,
        response: requests.Response,
        attempt: int,
    ) -> float:
        """Respeta el retry real informado por Gemini.

        Prioridad:
            Retry-After HTTP
            google.rpc.RetryInfo.retryDelay
            mensaje "Please retry in Xs"
            backoff exponencial

        Se agrega 1 segundo de margen para no golpear el mismo borde
        de cuota al reintentar.
        """

        candidates: list[float] = []

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                candidates.append(float(retry_after))
            except (TypeError, ValueError):
                pass

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                details = error.get("details")
                if isinstance(details, list):
                    for detail in details:
                        if not isinstance(detail, dict):
                            continue
                        retry_delay = detail.get("retryDelay")
                        parsed = cls._parse_duration_seconds(retry_delay)
                        if parsed is not None:
                            candidates.append(parsed)

                message = error.get("message")
                if isinstance(message, str):
                    match = re.search(
                        r"Please\s+retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
                        message,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        candidates.append(float(match.group(1)))

        if candidates:
            return max(1.0, max(candidates) + 1.0)

        # 503 de Gemini por alta demanda normalmente no informa retryDelay.
        # Tratarlo como backoff de disponibilidad, no como retry HTTP rapido.
        if int(getattr(response, "status_code", 0)) == 503:
            raw = os.getenv("QUANTIA_GEMINI_503_DELAYS", "").strip()
            delays = cls.HIGH_DEMAND_503_DELAYS
            if raw:
                try:
                    parsed = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
                    if parsed:
                        delays = parsed
                except ValueError:
                    pass
            index = min(max(attempt - 1, 0), len(delays) - 1)
            return max(1.0, float(delays[index]))

        return float(max(1, 2 ** (attempt - 1)))

    @staticmethod
    def _parse_duration_seconds(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        if not isinstance(value, str):
            return None
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*", value)
        if not match:
            return None
        return max(0.0, float(match.group(1)))

    # ========================================================
    # GENERATION CONFIG
    # ========================================================

    def _build_generation_config(
        self,
        *,
        output_mode: str,
        response_json_schema: (
            dict[str, Any]
            | None
        ),
    ) -> dict[str, Any]:

        if output_mode == "plain":
            return {}

        if output_mode == "json_only":
            return {
                "responseMimeType": (
                    "application/json"
                ),
            }

        if output_mode == (
            "response_json_schema"
        ):
            if not response_json_schema:
                raise SpatialVisionProviderError(
                    "response_json_schema requiere "
                    "un schema."
                )

            prepared_schema = (
                self._prepare_schema_for_gemini(
                    response_json_schema
                )
            )

            return {
                "responseMimeType": (
                    "application/json"
                ),
                "responseJsonSchema": (
                    prepared_schema
                ),
            }

        raise SpatialVisionProviderError(
            "Modo de salida Gemini desconocido: "
            f"{output_mode}"
        )

    # ========================================================
    # PREPARACIÓN DEL SCHEMA
    # ========================================================

    @classmethod
    def _prepare_schema_for_gemini(
        cls,
        schema: dict[str, Any],
    ) -> dict[str, Any]:

        unsupported_keys = {
            "$schema",
            "default",
            "examples",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "readOnly",
            "writeOnly",
        }

        def clean(
            value: Any,
        ) -> Any:
            if isinstance(
                value,
                list,
            ):
                return [
                    clean(item)
                    for item
                    in value
                ]

            if not isinstance(
                value,
                dict,
            ):
                return value

            result: dict[
                str,
                Any,
            ] = {}

            for key, item in (
                value.items()
            ):
                if (
                    key
                    in unsupported_keys
                ):
                    continue

                result[key] = clean(
                    item
                )

            return result

        prepared = clean(
            schema
        )

        if not isinstance(
            prepared,
            dict,
        ):
            raise SpatialVisionProviderError(
                "El JSON Schema preparado para Gemini "
                "no es un objeto válido."
            )

        return prepared

    # ========================================================
    # VALIDACIÓN DE CAMPOS REQUIRED
    # ========================================================

    @staticmethod
    def _validate_required_fields(
        *,
        data: (
            dict[str, Any]
            | list[Any]
        ),
        schema: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):
            raise SpatialVisionProviderError(
                "Gemini devolvió JSON, pero la raíz "
                "no es un objeto."
            )

        required = schema.get(
            "required",
            [],
        )

        if not isinstance(
            required,
            list,
        ):
            return

        missing = [
            field
            for field
            in required
            if field not in data
        ]

        if missing:
            raise SpatialVisionProviderError(
                "Gemini devolvió JSON que incumple "
                "el schema solicitado. "
                "Campos requeridos ausentes: "
                + ", ".join(
                    str(field)
                    for field
                    in missing
                )
            )

    # ========================================================
    # EXTRACCIÓN TEXTO
    # ========================================================

    @staticmethod
    def _extract_text(
        response: dict[str, Any],
    ) -> str:

        candidates = (
            response.get(
                "candidates"
            )
        )

        if (
            not isinstance(
                candidates,
                list,
            )
            or not candidates
        ):
            return ""

        candidate = candidates[0]

        if not isinstance(
            candidate,
            dict,
        ):
            return ""

        content = (
            candidate.get(
                "content"
            )
        )

        if not isinstance(
            content,
            dict,
        ):
            return ""

        parts = (
            content.get(
                "parts"
            )
        )

        if not isinstance(
            parts,
            list,
        ):
            return ""

        texts: list[str] = []

        for part in parts:
            if not isinstance(
                part,
                dict,
            ):
                continue

            text = part.get(
                "text"
            )

            if (
                isinstance(
                    text,
                    str,
                )
                and text.strip()
            ):
                texts.append(
                    text.strip()
                )

        return "\n".join(
            texts
        ).strip()

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def _parse_json(
        text: str,
    ) -> (
        dict[str, Any]
        | list[Any]
    ):

        cleaned = str(
            text or ""
        ).strip()

        if not cleaned:
            raise SpatialVisionProviderError(
                "Gemini devolvió una respuesta JSON vacía."
            )

        if cleaned.startswith(
            "```"
        ):
            lines = (
                cleaned.splitlines()
            )

            if lines:
                first = (
                    lines[0]
                    .strip()
                    .lower()
                )

                if first in {
                    "```",
                    "```json",
                }:
                    lines = (
                        lines[1:]
                    )

            if (
                lines
                and lines[-1].strip()
                == "```"
            ):
                lines = (
                    lines[:-1]
                )

            cleaned = (
                "\n".join(
                    lines
                ).strip()
            )

        try:
            parsed = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:
            raise SpatialVisionProviderError(
                "Gemini respondió, pero no entregó "
                "JSON válido."
            ) from exc

        if not isinstance(
            parsed,
            (dict, list),
        ):
            raise SpatialVisionProviderError(
                "La respuesta estructurada Gemini "
                "no es un objeto ni una lista JSON."
            )

        return parsed

    # ========================================================
    # HTTP ERROR
    # ========================================================

    @staticmethod
    def _http_error_message(
        response: requests.Response,
    ) -> str:

        try:
            payload = (
                response.json()
            )

        except ValueError:
            payload = None

        if isinstance(
            payload,
            dict,
        ):
            error = payload.get(
                "error"
            )

            if isinstance(
                error,
                dict,
            ):
                message = (
                    error.get(
                        "message"
                    )
                )

                status = (
                    error.get(
                        "status"
                    )
                )

                parts = [
                    f"Gemini HTTP "
                    f"{response.status_code}"
                ]

                if status:
                    parts.append(
                        str(status)
                    )

                if message:
                    parts.append(
                        str(message)
                    )

                return ": ".join(
                    parts
                )

        return (
            f"Gemini HTTP "
            f"{response.status_code}: "
            f"{str(response.text or '')[:1500]}"
        )

    # ========================================================
    # RESPUESTA VACÍA
    # ========================================================

    @staticmethod
    def _empty_response_message(
        response: dict[str, Any],
    ) -> str:

        prompt_feedback = (
            response.get(
                "promptFeedback"
            )
            or {}
        )

        block_reason = (
            prompt_feedback.get(
                "blockReason"
            )
            if isinstance(
                prompt_feedback,
                dict,
            )
            else None
        )

        candidates = (
            response.get(
                "candidates"
            )
        )

        finish_reason = None

        if (
            isinstance(
                candidates,
                list,
            )
            and candidates
            and isinstance(
                candidates[0],
                dict,
            )
        ):
            finish_reason = (
                candidates[0].get(
                    "finishReason"
                )
            )

        parts = [
            "Gemini no devolvió contenido utilizable."
        ]

        if block_reason:
            parts.append(
                f"blockReason={block_reason}"
            )

        if finish_reason:
            parts.append(
                f"finishReason={finish_reason}"
            )

        return " ".join(
            parts
        )

    # ========================================================
    # REQUEST
    # ========================================================

    def _validate_request(
        self,
        *,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
    ) -> None:

        if not self.api_key:
            raise SpatialVisionProviderError(
                "GEMINI_API_KEY no está configurada."
            )

        if not self.models:
            raise SpatialVisionProviderError(
                "No existen modelos Gemini configurados."
            )

        if (
            not isinstance(
                prompt,
                str,
            )
            or not prompt.strip()
        ):
            raise SpatialVisionProviderError(
                "El prompt está vacío."
            )

        if not media_bytes:
            raise SpatialVisionProviderError(
                "No se recibió contenido visual."
            )

        if (
            media_mime_type
            not in self.SUPPORTED_MEDIA_MIME_TYPES
        ):
            raise SpatialVisionProviderError(
                "MIME visual no soportado: "
                f"{media_mime_type or 'desconocido'}"
            )

    # ========================================================
    # MIME
    # ========================================================

    @staticmethod
    def _normalize_mime_type(
        value: str,
    ) -> str:

        return str(
            value or ""
        ).strip().lower()

    # ========================================================
    # MODELOS
    # ========================================================

    def _build_model_order(
        self,
    ) -> tuple[str, ...]:

        models: list[str] = []

        for model in (
            self.primary_model,
            self.fallback_model,
        ):
            normalized = str(
                model or ""
            ).strip()

            if (
                normalized
                and normalized
                not in models
            ):
                models.append(
                    normalized
                )

        return tuple(
            models
        )


# ============================================================
# FACTORY
# ============================================================


def get_spatial_vision_provider(
) -> GeminiSpatialVisionProvider:
    return GeminiSpatialVisionProvider()