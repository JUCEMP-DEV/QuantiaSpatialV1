from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.quantia_spatialV1.providers import build_call2_provider
from app.quantia_spatialV1.tests.call2_provider_test_utils import load_selected_case_and_level, safe_name

RUN = str(os.getenv("QUANTIA_RUN_CALL2_PROVIDER_SMOKE", "0") or "0").strip() == "1"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "call2_provider_compat"

SMOKE_SCHEMA = {
    "type": "object",
    "properties": {
        "level_view_id": {"type": "string"},
        "contains_architectural_plan": {"type": "boolean"},
        "status": {"type": "string", "enum": ["OK"]},
    },
    "required": ["level_view_id", "contains_architectural_plan", "status"],
    "additionalProperties": False,
}


@pytest.mark.skipif(not RUN, reason="Requiere QUANTIA_RUN_CALL2_PROVIDER_SMOKE=1")
def test_call2_provider_vision_schema_smoke_single_level() -> None:
    case, loaded = load_selected_case_and_level()
    level_view = loaded.level_result.level_view
    provider = build_call2_provider()

    availability = provider.check_model_available()
    assert availability["available"] is True, availability

    prompt = (
        "Compatibility smoke test only. Inspect the attached image. "
        "Return the requested JSON. level_view_id must be exactly "
        f"{level_view.id!r}; status must be OK."
    )
    result = provider.analyze(
        prompt=prompt,
        media_bytes=level_view.raster_bytes,
        media_mime_type="image/png",
        response_json_schema=SMOKE_SCHEMA,
    )
    assert isinstance(result.data, dict)
    assert result.data["level_view_id"] == level_view.id
    assert result.data["status"] == "OK"
    assert isinstance(result.data["contains_architectural_plan"], bool)

    out = OUTPUT_ROOT / safe_name(result.provider) / safe_name(result.model)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{case.case_id}__{safe_name(loaded.level_name)}__smoke.json").write_text(
        json.dumps({
            "provider": result.provider,
            "model": result.model,
            "availability": availability,
            "response": result.data,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
