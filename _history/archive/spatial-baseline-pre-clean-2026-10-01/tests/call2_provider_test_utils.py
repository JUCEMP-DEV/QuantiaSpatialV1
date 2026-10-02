from __future__ import annotations

import os
import re
from typing import Any

from app.quantia_spatialV1.tests.quantia_case_loader import CASE_LOADERS


def selected_case_id() -> str:
    value = str(os.getenv("QUANTIA_CALL2_CASE", "casa_viri") or "casa_viri").strip().lower()
    if value not in CASE_LOADERS:
        raise AssertionError(f"QUANTIA_CALL2_CASE no soportado: {value}; opciones={sorted(CASE_LOADERS)}")
    return value


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def select_loaded_level(case: Any) -> Any:
    selector = _norm(str(os.getenv("QUANTIA_CALL2_LEVEL", "planta_alta") or "planta_alta"))
    matches = []
    for loaded in case.levels:
        name = _norm(str(loaded.level_name))
        if selector == name or selector in name:
            matches.append(loaded)
    if len(matches) != 1:
        available = [str(item.level_name) for item in case.levels]
        raise AssertionError(
            f"QUANTIA_CALL2_LEVEL={selector!r} produjo {len(matches)} coincidencias; available={available}"
        )
    return matches[0]


def load_selected_case_and_level() -> tuple[Any, Any]:
    case = CASE_LOADERS[selected_case_id()]()
    return case, select_loaded_level(case)


def safe_name(value: str) -> str:
    return _norm(value)
