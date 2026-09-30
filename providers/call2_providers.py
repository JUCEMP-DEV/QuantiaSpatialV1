from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class Call2ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Call2ProviderResult:
    provider: str
    model: str
    text: str
    data: dict[str, Any] | list[Any] | None
    fallback_used: bool
    raw: dict[str, Any]


class Call2StrictSchemaAdapter:
    """Normaliza CALL2_SCHEMA al subconjunto estricto común.

    Groq strict exige todos los campos en `required` y objetos cerrados.
    Los campos opcionales del contrato interno se vuelven `anyOf[..., null]`.
    También elimina restricciones de array que no son necesarias para el
    transporte; Pydantic/Call2DeltaValidator conserva la validación final.
    """

    DROP_KEYS = {"minItems", "maxItems"}

    @classmethod
    def adapt(cls, schema: dict[str, Any]) -> dict[str, Any]:
        return cls._adapt_node(schema, optional=False)

    @classmethod
    def _adapt_node(cls, node: Any, *, optional: bool) -> Any:
        if not isinstance(node, dict):
            return node

        clean: dict[str, Any] = {
            key: value
            for key, value in node.items()
            if key not in cls.DROP_KEYS and key not in {"required", "additionalProperties"}
        }
        node_type = clean.get("type")

        if node_type == "object":
            original_props = node.get("properties", {}) or {}
            original_required = set(node.get("required", []) or [])
            props: dict[str, Any] = {}
            for name, child in original_props.items():
                adapted = cls._adapt_node(child, optional=False)
                if name not in original_required:
                    adapted = {"anyOf": [adapted, {"type": "null"}]}
                props[name] = adapted
            clean["properties"] = props
            clean["required"] = list(original_props.keys())
            clean["additionalProperties"] = False

        elif node_type == "array" and "items" in node:
            clean["items"] = cls._adapt_node(node["items"], optional=False)

        if "anyOf" in node:
            clean["anyOf"] = [cls._adapt_node(item, optional=False) for item in node["anyOf"]]

        if optional:
            return {"anyOf": [clean, {"type": "null"}]}
        return clean


class _EnvKeyResolver:
    """Lee secretos sin imprimirlos ni persistirlos.

    Primero usa el proceso; si las claves están solo en `.env`/`.env.local`,
    busca desde el cwd hacia sus padres. No modifica os.environ.
    """

    @staticmethod
    def get(name: str) -> str:
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
        candidates: list[Path] = []
        cwd = Path.cwd().resolve()
        for parent in (cwd, *cwd.parents):
            candidates.extend([parent / ".env", parent / ".env.local"])
        seen: set[Path] = set()
        for path in candidates:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                for raw in path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, raw_value = line.split("=", 1)
                    if key.strip() != name:
                        continue
                    value = raw_value.strip().strip('"').strip("'")
                    if value:
                        return value
            except OSError:
                continue
        return ""


class _BaseHTTPCall2Provider:
    provider_name = "base"
    default_model = ""
    models_url = ""
    completion_url = ""

    def __init__(self, *, api_key: str | None = None, model: str | None = None, timeout_s: float = 180.0) -> None:
        self.api_key = str(api_key or self._load_api_key()).strip()
        self.model = str(model or self._configured_model() or self.default_model).strip()
        self.timeout_s = float(timeout_s)
        if not self.api_key:
            raise Call2ProviderError(f"Falta API key para provider={self.provider_name}.")
        if not self.model:
            raise Call2ProviderError(f"Falta model id para provider={self.provider_name}.")

    def _load_api_key(self) -> str:
        raise NotImplementedError

    def _configured_model(self) -> str:
        return ""

    @staticmethod
    def _data_url(media_bytes: bytes, media_mime_type: str) -> str:
        encoded = base64.b64encode(media_bytes).decode("ascii")
        return f"data:{media_mime_type};base64,{encoded}"

    def check_model_available(self) -> dict[str, Any]:
        response = requests.get(
            self.models_url,
            headers=self._headers(),
            timeout=min(self.timeout_s, 60.0),
        )
        self._raise_http(response)
        payload = response.json()
        models = payload.get("data", payload) if isinstance(payload, dict) else payload
        ids: set[str] = set()
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict):
                    model_id = item.get("id") or item.get("name")
                    if model_id:
                        ids.add(str(model_id))
        return {
            "provider": self.provider_name,
            "model": self.model,
            "available": self.model in ids,
            "model_count": len(ids),
        }

    def analyze(
        self,
        *,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
        response_json_schema: dict[str, Any] | None = None,
    ) -> Call2ProviderResult:
        if not prompt.strip():
            raise Call2ProviderError("Prompt vacío.")
        if not media_bytes:
            raise Call2ProviderError("Imagen vacía.")
        if media_mime_type not in {"image/png", "image/jpeg"}:
            raise Call2ProviderError(f"MIME no soportado: {media_mime_type}")

        request_json = self.build_request(
            prompt=prompt,
            media_bytes=media_bytes,
            media_mime_type=media_mime_type,
            response_json_schema=response_json_schema,
        )
        response = requests.post(
            self.completion_url,
            headers=self._headers(),
            json=request_json,
            timeout=self.timeout_s,
        )
        self._raise_http(response)
        raw = response.json()
        text = self._extract_text(raw)
        data: dict[str, Any] | list[Any] | None = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                data = parsed
        except json.JSONDecodeError as exc:
            raise Call2ProviderError(
                f"{self.provider_name}/{self.model} devolvió texto no JSON: {exc}"
            ) from exc
        return Call2ProviderResult(
            provider=self.provider_name,
            model=self.model,
            text=text,
            data=data,
            fallback_used=False,
            raw=raw if isinstance(raw, dict) else {"response": raw},
        )

    def build_request(
        self,
        *,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
        response_json_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _raise_http(self, response: requests.Response) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
            message = json.dumps(payload, ensure_ascii=False)
        except Exception:
            message = response.text
        if len(message) > 2000:
            message = message[:2000] + "..."
        raise Call2ProviderError(
            f"{self.provider_name}/{self.model} HTTP {response.status_code}: {message}"
        )

    @staticmethod
    def _extract_text(raw: dict[str, Any]) -> str:
        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception as exc:
            raise Call2ProviderError("Respuesta sin choices[0].message.content") from exc
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        chunks.append(text)
            joined = "".join(chunks).strip()
            if joined:
                return joined
        raise Call2ProviderError("Contenido de respuesta no textual.")


class GroqQwenCall2Provider(_BaseHTTPCall2Provider):
    provider_name = "groq"
    default_model = "qwen/qwen3.8-27b"
    models_url = "https://api.groq.com/openai/v1/models"
    completion_url = "https://api.groq.com/openai/v1/chat/completions"

    def _load_api_key(self) -> str:
        return _EnvKeyResolver.get("GROQ_API_KEY")

    def _configured_model(self) -> str:
        return str(os.getenv("QUANTIA_GROQ_CALL2_MODEL", "") or "").strip()

    def build_request(self, *, prompt: str, media_bytes: bytes, media_mime_type: str, response_json_schema: dict[str, Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._data_url(media_bytes, media_mime_type)}},
                ],
            }],
            "temperature": 0,
            "max_completion_tokens": int(os.getenv("QUANTIA_CALL2_MAX_OUTPUT_TOKENS", "4096")),
            "reasoning_effort": str(os.getenv("QUANTIA_GROQ_REASONING_EFFORT", "medium") or "medium"),
        }
        if response_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "quantia_call2",
                    "strict": True,
                    "schema": Call2StrictSchemaAdapter.adapt(response_json_schema),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        return payload


class MistralCall2Provider(_BaseHTTPCall2Provider):
    provider_name = "mistral"
    default_model = "mistral-medium-3-5"
    models_url = "https://api.mistral.ai/v1/models"
    completion_url = "https://api.mistral.ai/v1/chat/completions"

    def _load_api_key(self) -> str:
        return _EnvKeyResolver.get("MISTRAL_API_KEY")

    def _configured_model(self) -> str:
        return str(os.getenv("QUANTIA_MISTRAL_CALL2_MODEL", "") or "").strip()

    def build_request(self, *, prompt: str, media_bytes: bytes, media_mime_type: str, response_json_schema: dict[str, Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": self._data_url(media_bytes, media_mime_type)},
                ],
            }],
            "temperature": 0,
            "max_tokens": int(os.getenv("QUANTIA_CALL2_MAX_OUTPUT_TOKENS", "4096")),
            "reasoning_effort": str(os.getenv("QUANTIA_MISTRAL_REASONING_EFFORT", "medium") or "medium"),
        }
        if response_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "quantia_call2",
                    "schema": Call2StrictSchemaAdapter.adapt(response_json_schema),
                    "strict": True,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        return payload


def build_call2_provider(provider: str | None = None) -> Any:
    name = str(provider or os.getenv("QUANTIA_CALL2_PROVIDER", "groq") or "groq").strip().lower()
    if name == "groq":
        return GroqQwenCall2Provider()
    if name == "mistral":
        return MistralCall2Provider()
    if name == "gemini":
        # Import diferido para no acoplar Groq/Mistral a app.core.config.
        from app.quantia_spatialV1.providers.vision import GeminiSpatialVisionProvider
        return GeminiSpatialVisionProvider()
    raise Call2ProviderError(f"QUANTIA_CALL2_PROVIDER no soportado: {name}")
