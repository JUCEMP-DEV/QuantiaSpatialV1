from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_contract import (
    SEMANTIC_CATEGORIES,
    SEMANTIC_CONTRACT_VERSION,
    SEMANTIC_SOURCE_MODE,
    SEMANTIC_STATES,
)


class GeminiSemanticExtractor:
    """
    Fase 01.5 — postprocesamiento semántico de la extracción legacy.

    No modifica el prompt legacy ni vuelve a consultar Gemini.
    Recibe el JSON de QUANTIA_EXTRACTION_PROMPT y lo descompone en
    observaciones semánticas asociadas al LevelView correspondiente.
    """

    def extract(
        self,
        *,
        payload: dict[str, Any],
        level_view: LevelView,
        allow_unique_level_fallback: bool = False,
    ) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []

        self._append_document_observations(payload=payload, observations=observations)
        self._append_site_observations(payload=payload, observations=observations)

        level = self._select_level(
            payload=payload,
            level_name=level_view.level_name,
            allow_unique_level_fallback=allow_unique_level_fallback,
        )

        if level is not None:
            self._append_level_observation(level=level, observations=observations)
            self._append_dimensions(level=level, observations=observations)
            self._append_axes(level=level, observations=observations)
            self._append_spaces(level=level, observations=observations)
            self._append_stairs(level=level, observations=observations)

        self._append_constructive_information(
            payload=payload,
            observations=observations,
        )

        normalized = [
            self._normalize_observation(item)
            for item in observations
            if isinstance(item, dict)
        ]

        return {
            "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
            "semantic_source_mode": SEMANTIC_SOURCE_MODE,
            "summary": str(payload.get("resumen") or "").strip(),
            "observations": normalized,
            "conflicts": self._conflict_descriptions(payload.get("conflictos")),
            "unidentified_relevant_data": self._string_list(
                payload.get("datos_no_identificados")
            ),
            "legacy_confirmations": self._dict_list(
                payload.get("confirmaciones_requeridas")
            ),
            "matched_level_name": (
                str(level.get("nombre") or "").strip() if level is not None else None
            ),
        }

    def _append_document_observations(
        self,
        *,
        payload: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        document = payload.get("documento")
        if not isinstance(document, dict):
            return

        properties = self._properties(
            (
                ("title", document.get("titulo")),
                ("drawing_type", document.get("tipo_plano")),
                ("declared_scale", document.get("escala")),
                ("north_orientation", document.get("orientacion")),
            ),
            state="DETECTADO",
            confidence=None,
        )

        if properties:
            observations.append(
                self._observation(
                    category="DOCUMENT",
                    subtype="LEGACY_DOCUMENT",
                    name=self._text(document.get("titulo")),
                    description="Información documental extraída por Gemini legacy.",
                    state="DETECTADO",
                    confidence=None,
                    properties=properties,
                )
            )

        scale = self._text(document.get("escala"))
        if scale:
            observations.append(
                self._observation(
                    category="SCALE",
                    subtype="DECLARED_SCALE",
                    description=f"Escala declarada observada: {scale}",
                    visible_text=scale,
                    state="DETECTADO",
                    confidence=None,
                    properties=self._properties(
                        (("scale_text", scale),),
                        state="DETECTADO",
                        confidence=None,
                    ),
                )
            )

        orientation = self._text(document.get("orientacion"))
        if orientation:
            observations.append(
                self._observation(
                    category="ORIENTATION",
                    subtype="DOCUMENT_ORIENTATION",
                    description=f"Orientación declarada/observada: {orientation}",
                    visible_text=orientation,
                    state="DETECTADO",
                    confidence=None,
                    properties=self._properties(
                        (("cardinal_direction", orientation),),
                        state="DETECTADO",
                        confidence=None,
                    ),
                )
            )

    def _append_site_observations(
        self,
        *,
        payload: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        site = payload.get("predio")
        if not isinstance(site, dict):
            return

        state = self._state(site.get("estado"))
        confidence = self._confidence(site.get("confianza"))
        measurements: list[dict[str, Any]] = []

        self._append_measurement(
            measurements,
            name="width",
            value=site.get("ancho_m"),
            unit="M",
            state=state,
            confidence=confidence,
        )
        self._append_measurement(
            measurements,
            name="depth",
            value=site.get("fondo_m"),
            unit="M",
            state=state,
            confidence=confidence,
        )
        self._append_measurement(
            measurements,
            name="area",
            value=site.get("area_m2"),
            unit="M2",
            state=state,
            confidence=confidence,
        )

        properties = self._properties(
            (
                ("access", site.get("acceso_principal")),
                ("orientation", site.get("orientacion")),
            ),
            state=state,
            confidence=confidence,
        )

        boundaries = site.get("colindancias")
        if isinstance(boundaries, dict):
            for cardinal in ("norte", "sur", "este", "oeste"):
                value = self._text(boundaries.get(cardinal))
                if value:
                    properties.append(
                        self._property(
                            name=f"{cardinal}_boundary",
                            value=value,
                            state=state,
                            confidence=confidence,
                        )
                    )

        if measurements or properties or self._string_list(site.get("evidencia")):
            observations.append(
                self._observation(
                    category="SITE",
                    subtype="PLOT",
                    description="Predio extraído por Gemini legacy.",
                    state=state,
                    confidence=confidence,
                    measurements=measurements,
                    properties=properties,
                    evidence=self._string_list(site.get("evidencia")),
                )
            )

        access = self._text(site.get("acceso_principal"))
        if access:
            observations.append(
                self._observation(
                    category="ACCESS",
                    subtype="PRIMARY_ACCESS",
                    description=access,
                    visible_text=access,
                    state=state,
                    confidence=confidence,
                    evidence=self._string_list(site.get("evidencia")),
                )
            )

    def _append_level_observation(
        self,
        *,
        level: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        name = self._text(level.get("nombre"))
        observations.append(
            self._observation(
                category="LEVEL",
                subtype="LEGACY_LEVEL",
                name=name,
                description=f"Nivel extraído por Gemini legacy: {name or 'NO_IDENTIFICADO'}.",
                state="DETECTADO" if name else "NO_IDENTIFICADO",
                confidence=None,
                properties=self._properties(
                    (("level_name", name),),
                    state="DETECTADO" if name else "NO_IDENTIFICADO",
                    confidence=None,
                ),
            )
        )

    def _append_dimensions(
        self,
        *,
        level: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        dimensions = level.get("cotas")
        if not isinstance(dimensions, list):
            return

        for index, item in enumerate(dimensions, start=1):
            if not isinstance(item, dict):
                continue

            reference = self._text(item.get("referencia"))
            parsed = self._parse_dimension_reference(reference)
            state = self._state(item.get("estado"))
            confidence = self._confidence(item.get("confianza"))
            visible_text = self._text(item.get("texto"))
            value = self._number(item.get("valor_m"))

            measurement: dict[str, Any] = {
                "name": "dimension",
                "state": state,
            }
            if visible_text:
                measurement["text"] = visible_text
            if value is not None:
                measurement["value"] = value
                measurement["unit"] = "M"
            if confidence is not None:
                measurement["confidence"] = confidence
            if parsed["reference_start"]:
                measurement["reference_start"] = parsed["reference_start"]
            if parsed["reference_end"]:
                measurement["reference_end"] = parsed["reference_end"]
            if parsed["orientation"]:
                measurement["orientation"] = parsed["orientation"]
            if parsed["dimension_side"]:
                measurement["dimension_side"] = parsed["dimension_side"]
            if parsed["span_type"]:
                measurement["span_type"] = parsed["span_type"]

            observations.append(
                self._observation(
                    category="DIMENSION",
                    subtype="LEGACY_DIMENSION",
                    name=f"dimension_{index}",
                    description=reference or visible_text or "Cota extraída por Gemini legacy.",
                    visible_text=visible_text,
                    state=state,
                    confidence=confidence,
                    orientation=parsed["orientation"],
                    dimension_side=parsed["dimension_side"],
                    reference_start=parsed["reference_start"],
                    reference_end=parsed["reference_end"],
                    span_type=parsed["span_type"],
                    measurements=[measurement],
                    properties=self._properties(
                        (("chain_reference", reference),),
                        state=state,
                        confidence=confidence,
                    ),
                )
            )

    def _append_axes(
        self,
        *,
        level: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        labels: set[str] = set()

        dimensions = level.get("cotas")
        if isinstance(dimensions, list):
            for item in dimensions:
                if not isinstance(item, dict):
                    continue
                parsed = self._parse_dimension_reference(
                    self._text(item.get("referencia"))
                )
                for key in ("reference_start", "reference_end"):
                    value = parsed[key]
                    if value:
                        labels.add(value)

        for label in sorted(labels, key=self._axis_sort_key):
            observations.append(
                self._observation(
                    category="AXIS",
                    subtype="DIMENSION_REFERENCE_AXIS",
                    name=label,
                    description=f"Eje {label} referenciado explícitamente por cotas legacy.",
                    visible_text=label,
                    state="DETECTADO",
                    confidence=None,
                    properties=self._properties(
                        (("axis_label", label),),
                        state="DETECTADO",
                        confidence=None,
                    ),
                )
            )

    def _append_spaces(
        self,
        *,
        level: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        spaces = level.get("espacios")
        if not isinstance(spaces, list):
            return

        for space in spaces:
            if not isinstance(space, dict):
                continue

            state = self._state(space.get("estado"))
            confidence = self._confidence(space.get("confianza"))
            name = self._text(space.get("nombre"))
            space_id = self._text(space.get("id_propuesto"))
            measurements: list[dict[str, Any]] = []

            self._append_measurement(
                measurements,
                name="width",
                value=space.get("ancho_m"),
                unit="M",
                state=state,
                confidence=confidence,
            )
            self._append_measurement(
                measurements,
                name="length",
                value=space.get("largo_m"),
                unit="M",
                state=state,
                confidence=confidence,
            )
            self._append_measurement(
                measurements,
                name="area",
                value=space.get("area_m2"),
                unit="M2",
                state=state,
                confidence=confidence,
            )

            relations: list[dict[str, Any]] = []
            for target in self._string_list(space.get("comunica_con")):
                relations.append(
                    self._relation(
                        relation_type="COMMUNICATES_WITH",
                        target=target,
                        target_category="SPACE",
                        state=state,
                        confidence=confidence,
                    )
                )
            for target in self._string_list(space.get("comparte_muro_con")):
                relations.append(
                    self._relation(
                        relation_type="SHARES_WALL_WITH",
                        target=target,
                        target_category="SPACE",
                        state=state,
                        confidence=confidence,
                    )
                )

            observations.append(
                self._observation(
                    category="SPACE",
                    subtype=self._text(space.get("tipo")),
                    name=name or space_id,
                    description=self._text(space.get("ubicacion")) or name or "Espacio legacy.",
                    state=state,
                    confidence=confidence,
                    location_text=self._text(space.get("ubicacion")),
                    bbox_normalized=self._bbox(space.get("bbox_normalizado")),
                    measurements=measurements,
                    properties=self._properties(
                        (
                            ("id_proposed", space_id),
                            ("type", space.get("tipo")),
                            ("location", space.get("ubicacion")),
                        ),
                        state=state,
                        confidence=confidence,
                    ),
                    relations=relations,
                    evidence=self._string_list(space.get("evidencia")),
                )
            )

            self._append_openings_for_space(
                space=space,
                space_id=space_id or name,
                observations=observations,
            )

    def _append_openings_for_space(
        self,
        *,
        space: dict[str, Any],
        space_id: str | None,
        observations: list[dict[str, Any]],
    ) -> None:
        for field_name, category in (("puertas", "DOOR"), ("ventanas", "WINDOW")):
            items = space.get(field_name)
            if not isinstance(items, list):
                continue

            for index, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue

                state = self._state(item.get("estado"))
                confidence = self._confidence(item.get("confianza"))
                measurements: list[dict[str, Any]] = []
                self._append_measurement(
                    measurements,
                    name="width",
                    value=item.get("ancho_m"),
                    unit="M",
                    state=state,
                    confidence=confidence,
                )
                self._append_measurement(
                    measurements,
                    name="height",
                    value=item.get("alto_m"),
                    unit="M",
                    state=state,
                    confidence=confidence,
                )

                relations: list[dict[str, Any]] = []
                if space_id:
                    relations.append(
                        self._relation(
                            relation_type="LOCATED_IN",
                            target=space_id,
                            target_category="SPACE",
                            state=state,
                            confidence=confidence,
                        )
                    )

                target = self._text(item.get("hacia"))
                if target:
                    relations.append(
                        self._relation(
                            relation_type="OPENS_TO",
                            target=target,
                            target_category="SPACE",
                            state=state,
                            confidence=confidence,
                        )
                    )

                observations.append(
                    self._observation(
                        category=category,
                        subtype="LEGACY_OPENING",
                        name=f"{space_id or 'SPACE'}_{category}_{index}",
                        description=self._text(item.get("ubicacion")) or category,
                        state=state,
                        confidence=confidence,
                        location_text=self._text(item.get("ubicacion")),
                        measurements=measurements,
                        properties=self._properties(
                            (("host_reference", item.get("ubicacion")),),
                            state=state,
                            confidence=confidence,
                        ),
                        relations=relations,
                        evidence=self._string_list(item.get("evidencia")),
                    )
                )

    def _append_stairs(
        self,
        *,
        level: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        stairs = level.get("escaleras")
        if not isinstance(stairs, list):
            return

        for index, stair in enumerate(stairs, start=1):
            if not isinstance(stair, dict):
                continue

            state = self._state(stair.get("estado"))
            confidence = self._confidence(stair.get("confianza"))
            relations = [
                self._relation(
                    relation_type="CONTINUES_TO",
                    target=target,
                    target_category="LEVEL",
                    state=state,
                    confidence=confidence,
                )
                for target in self._string_list(stair.get("comunica_con"))
            ]

            observations.append(
                self._observation(
                    category="STAIR",
                    subtype="LEGACY_STAIR",
                    name=f"STAIR_{index}",
                    description=self._text(stair.get("ubicacion")) or "Escalera legacy.",
                    visible_text=self._text(stair.get("sentido")),
                    state=state,
                    confidence=confidence,
                    location_text=self._text(stair.get("ubicacion")),
                    properties=self._properties(
                        (
                            ("direction", stair.get("sentido")),
                            ("location", stair.get("ubicacion")),
                        ),
                        state=state,
                        confidence=confidence,
                    ),
                    relations=relations,
                    evidence=self._string_list(stair.get("evidencia")),
                )
            )

    def _append_constructive_information(
        self,
        *,
        payload: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        items = payload.get("informacion_constructiva")
        if not isinstance(items, list):
            return

        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            field = self._text(item.get("campo"))
            value = self._text(item.get("valor"))
            state = self._state(item.get("estado"))
            confidence = self._confidence(item.get("confianza"))
            normalized = self._normalize_text(f"{field or ''} {value or ''}")

            if "castillo" in normalized or "armex" in normalized or "refuerzo" in normalized:
                category = "STRUCTURAL_REINFORCEMENT"
            elif "material" in normalized:
                category = "MATERIAL"
            else:
                category = "NOTE"

            observations.append(
                self._observation(
                    category=category,
                    subtype="LEGACY_CONSTRUCTIVE_INFORMATION",
                    name=field or f"constructive_{index}",
                    description=" - ".join(part for part in (field, value) if part),
                    visible_text=value,
                    state=state,
                    confidence=confidence,
                    properties=self._properties(
                        (("construction_note", value), ("label", field)),
                        state=state,
                        confidence=confidence,
                    ),
                    evidence=self._string_list(item.get("evidencia")),
                )
            )

    @classmethod
    def _select_level(
        cls,
        *,
        payload: dict[str, Any],
        level_name: str | None,
        allow_unique_level_fallback: bool = False,
    ) -> dict[str, Any] | None:
        levels = payload.get("niveles")
        if not isinstance(levels, list):
            return None

        valid_levels = [level for level in levels if isinstance(level, dict)]
        target = cls._normalize_text(level_name)
        for level in valid_levels:
            if cls._normalize_text(level.get("nombre")) == target:
                return level

        # Fallback seguro para documentos/páginas con un único LevelView conocido:
        # si Gemini devolvió exactamente un nivel, el nombre legacy puede ser genérico
        # (p. ej. "Nivel Planta") aunque la geometría pertenezca inequívocamente al
        # único LevelView. La política se habilita desde el adapter, que sí conoce
        # cuántos LevelViews existen en la página. En páginas multinivel no se usa.
        if allow_unique_level_fallback and len(valid_levels) == 1:
            return valid_levels[0]

        return None

    @classmethod
    def _parse_dimension_reference(
        cls,
        reference: str | None,
    ) -> dict[str, str | None]:
        result: dict[str, str | None] = {
            "orientation": None,
            "dimension_side": None,
            "reference_start": None,
            "reference_end": None,
            "span_type": None,
        }
        if not reference:
            return result

        parts = [part.strip() for part in reference.split("|")]
        for part in parts:
            normalized = cls._normalize_text(part)

            # Gemini legacy puede enriquecer la orientación con el lado de la
            # cadena: "horizontal superior", "horizontal inferior",
            # "vertical izquierda" o "vertical derecha". La orientación
            # canónica se normaliza sin perder ese lado porque Scale Foundation
            # lo usa para localizar la línea gráfica GENERAL correcta.
            if "horizontal" in normalized:
                result["orientation"] = "HORIZONTAL"
                if "superior" in normalized or "arriba" in normalized:
                    result["dimension_side"] = "TOP"
                elif "inferior" in normalized or "abajo" in normalized:
                    result["dimension_side"] = "BOTTOM"
                continue

            if "vertical" in normalized:
                result["orientation"] = "VERTICAL"
                if "izquierda" in normalized:
                    result["dimension_side"] = "LEFT"
                elif "derecha" in normalized:
                    result["dimension_side"] = "RIGHT"
                continue

            if normalized in {"general", "tramo", "parcial", "local"}:
                result["span_type"] = normalized.upper()
                continue

            axis_match = re.search(
                r"\beje\s+([A-Za-z0-9]+)\s*(?:a|->|→|-)\s*([A-Za-z0-9]+)\b",
                part,
                flags=re.IGNORECASE,
            )
            if axis_match:
                result["reference_start"] = axis_match.group(1)
                result["reference_end"] = axis_match.group(2)

        return result

    @staticmethod
    def _observation(
        *,
        category: str,
        description: str,
        state: str,
        confidence: float | None,
        subtype: str | None = None,
        name: str | None = None,
        visible_text: str | None = None,
        location_text: str | None = None,
        orientation: str | None = None,
        dimension_side: str | None = None,
        reference_start: str | None = None,
        reference_end: str | None = None,
        span_type: str | None = None,
        bbox_normalized: dict[str, float] | None = None,
        measurements: list[dict[str, Any]] | None = None,
        properties: list[dict[str, Any]] | None = None,
        relations: list[dict[str, Any]] | None = None,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "category": category,
            "subtype": subtype,
            "name": name,
            "description": description,
            "visible_text": visible_text,
            "state": state,
            "confidence": confidence,
            "location_text": location_text,
            "orientation": orientation,
            "dimension_side": dimension_side,
            "reference_start": reference_start,
            "reference_end": reference_end,
            "span_type": span_type,
            "bbox_normalized": bbox_normalized,
            "measurements": measurements or [],
            "properties": properties or [],
            "relations": relations or [],
            "evidence": evidence or [],
        }

    @classmethod
    def _normalize_observation(cls, raw: dict[str, Any]) -> dict[str, Any]:
        observation = dict(raw)
        category = str(raw.get("category") or "OTHER").strip().upper()
        state = cls._state(raw.get("state"))

        if category not in SEMANTIC_CATEGORIES:
            category = "OTHER"

        observation["category"] = category
        observation["state"] = state
        observation["description"] = str(raw.get("description") or "").strip()
        observation["confidence"] = cls._confidence(raw.get("confidence"))
        observation["measurements"] = cls._dict_list(raw.get("measurements"))
        observation["properties"] = cls._dict_list(raw.get("properties"))
        observation["relations"] = cls._dict_list(raw.get("relations"))
        observation["evidence"] = cls._string_list(raw.get("evidence"))
        return observation

    @classmethod
    def _properties(
        cls,
        values: tuple[tuple[str, object], ...],
        *,
        state: str,
        confidence: float | None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name, value in values:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            result.append(
                cls._property(
                    name=name,
                    value=value,
                    state=state,
                    confidence=confidence,
                )
            )
        return result

    @staticmethod
    def _property(
        *,
        name: str,
        value: object,
        state: str,
        confidence: float | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"name": name, "state": state}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result["value_number"] = float(value)
        else:
            result["value_text"] = str(value)
        if confidence is not None:
            result["confidence"] = confidence
        return result

    @staticmethod
    def _relation(
        *,
        relation_type: str,
        target: str,
        target_category: str,
        state: str,
        confidence: float | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": relation_type,
            "target": target,
            "target_category": target_category,
            "state": state,
        }
        if confidence is not None:
            result["confidence"] = confidence
        return result

    @classmethod
    def _append_measurement(
        cls,
        target: list[dict[str, Any]],
        *,
        name: str,
        value: object,
        unit: str,
        state: str,
        confidence: float | None,
    ) -> None:
        parsed = cls._number(value)
        if parsed is None:
            return
        item: dict[str, Any] = {
            "name": name,
            "value": parsed,
            "unit": unit,
            "state": state,
        }
        if confidence is not None:
            item["confidence"] = confidence
        target.append(item)

    @staticmethod
    def _bbox(value: object) -> dict[str, float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            x_min, y_min, x_max, y_max = (float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
            return None
        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    @staticmethod
    def _number(value: object) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _confidence(value: object) -> float | None:
        parsed = GeminiSemanticExtractor._number(value)
        if parsed is None or not 0.0 <= parsed <= 1.0:
            return None
        return parsed

    @staticmethod
    def _state(value: object) -> str:
        state = str(value or "NO_IDENTIFICADO").strip().upper()
        return state if state in SEMANTIC_STATES else "NO_IDENTIFICADO"

    @staticmethod
    def _text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_text(value: object) -> str:
        text = str(value or "").strip().lower()
        text = unicodedata.normalize("NFKD", text)
        return "".join(char for char in text if not unicodedata.combining(char))

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _dict_list(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @classmethod
    def _conflict_descriptions(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, dict):
                description = cls._text(item.get("descripcion"))
                if description:
                    result.append(description)
            elif cls._text(item):
                result.append(str(item).strip())
        return result

    @staticmethod
    def _axis_sort_key(value: str) -> tuple[int, object]:
        return (0, int(value)) if value.isdigit() else (1, value.upper())
