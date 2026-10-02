from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from app.quantia_spatialV1.engine import QuantiaSpatialEngine
from app.quantia_spatialV1.phase_01_level.pymupdf_level_source import PyMuPDFLevelSource
from app.quantia_spatialV1.phase_015_evidence.gemini_evidence_adapter import GeminiEvidenceAdapter
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_history import GeminiSemanticHistory
from app.prompts.quantia_extraction_prompt import QUANTIA_EXTRACTION_PROMPT
from app.schemas.gemini_extraction_transport import get_gemini_extraction_transport_schema


TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"

MIGUEL_H_PDF_PATH = Path(
    r"D:\03 INGENIEIRA SISTEMAS\03 RESIDENCIAS PROFESIONALES"
    r"\REFERENCIAS Y ANEXOS Quantia General"
    r"\Plano Migue H sin muebles.pdf"
)
MIGUEL_H_RENDER_SCALE = 0.7935


@dataclass(frozen=True)
class LoadedLevel:
    level_name: str
    level_result: Any
    replay_source: dict[str, Any] | None = None


@dataclass(frozen=True)
class LoadedCase:
    case_id: str
    case_name: str
    levels: tuple[LoadedLevel, ...]


class _ReplayOnlyProvider:
    models = ("replay-only",)

    def analyze(self, **kwargs):
        raise AssertionError("La prueba estandarizada es replay-only; no permite llamadas Gemini reales.")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _find_casa_localization_replay(*, history_path: Path, raster_bytes: bytes) -> dict[str, Any] | None:
    records = _read_jsonl(history_path)
    raster_hash = _sha256_bytes(raster_bytes)
    started_by_key = {
        str(row.get("call_key")): row
        for row in records
        if row.get("event") == "STARTED" and row.get("call_key")
    }
    matches: list[dict[str, Any]] = []
    for row in records:
        if row.get("event") != "SUCCEEDED":
            continue
        data = row.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("niveles"), list):
            continue
        started = started_by_key.get(str(row.get("call_key")))
        if started and str(started.get("raster_sha256")) == raster_hash:
            matches.append(row)
    return matches[-1]["data"] if matches else None


def _find_casa_semantic_replay(
    *, history_path: Path, source_document_id: str, source_page_number: int, raster_bytes: bytes
) -> dict[str, Any] | None:
    records = _read_jsonl(history_path)
    raster_hash = _sha256_bytes(raster_bytes)
    started_by_call_id = {
        str(row.get("call_id")): row
        for row in records
        if row.get("event") == "STARTED" and row.get("call_id")
    }
    matches: list[dict[str, Any]] = []
    for row in records:
        if row.get("event") != "SUCCEEDED":
            continue
        if str(row.get("source_document_id")) != source_document_id:
            continue
        if int(row.get("source_page_number") or -1) != source_page_number:
            continue
        payload = row.get("response_payload")
        if not isinstance(payload, dict):
            continue
        started = started_by_call_id.get(str(row.get("call_id")))
        if started and str(started.get("raster_sha256")) == raster_hash:
            matches.append(row)
    return matches[-1]["response_payload"] if matches else None


def _find_exact_extraction_replay(
    *, history_path: Path, source_document_id: str, source_page_number: int, raster_bytes: bytes
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    records = _read_jsonl(history_path)
    prompt_hash = _sha256_text(QUANTIA_EXTRACTION_PROMPT)
    schema_hash = _sha256_json(get_gemini_extraction_transport_schema())
    raster_hash = _sha256_bytes(raster_bytes)

    started: dict[str, dict[str, Any]] = {}
    for record in records:
        call_id = str(record.get("call_id") or "").strip()
        if call_id and record.get("event") == "STARTED":
            started[call_id] = record

    for record in reversed(records):
        if record.get("event") != "SUCCEEDED":
            continue
        call_id = str(record.get("call_id") or "").strip()
        begin = started.get(call_id)
        if begin is None:
            continue
        if str(begin.get("source_document_id") or "") != source_document_id:
            continue
        if int(begin.get("source_page_number") or 0) != source_page_number:
            continue
        if begin.get("prompt_sha256") != prompt_hash:
            continue
        if begin.get("schema_sha256") != schema_hash:
            continue
        if begin.get("raster_sha256") != raster_hash:
            continue
        payload = record.get("response_payload")
        if not isinstance(payload, dict):
            continue
        replay = {
            "result": {
                "data": payload,
                "model": record.get("model"),
                "fallback_used": bool(record.get("fallback_used", False)),
            }
        }
        source = {
            "call_id": call_id,
            "provider": record.get("provider"),
            "model": record.get("model"),
            "timestamp_utc": record.get("timestamp_utc"),
        }
        return replay, source
    return None, None


def load_casa_viri() -> LoadedCase:
    case_dir = DATA_DIR / "casa_viri"
    config_path = _first_existing(case_dir / "case.json", case_dir / "input" / "case.json")
    config = _read_json(config_path)
    phase_01_history = _first_existing(
        case_dir / "replay" / "phase_01_vision_history.jsonl",
        case_dir / "output" / "phase_01_vision_history.jsonl",
    )
    semantic_history = _first_existing(
        case_dir / "replay" / "gemini_semantic_history.jsonl",
        case_dir / "output" / "gemini_semantic_history.jsonl",
    )

    pdf_path = Path(str(config["pdf_path"]))
    if not pdf_path.exists():
        pytest.skip(f"PDF Casa Viri no disponible: {pdf_path}")

    document_bytes = pdf_path.read_bytes()
    render_scale = float(config["render_scale"])
    source_document_id = str(config["source_document_id"])
    page_number = 1
    page = PyMuPDFLevelSource().read(document_bytes=document_bytes, render_scale=render_scale).pages[0]

    localization = _find_casa_localization_replay(
        history_path=phase_01_history,
        raster_bytes=page.raster_bytes,
    )
    assert localization is not None, "Falta replay exacto F01 Casa Viri."
    extraction = _find_casa_semantic_replay(
        history_path=semantic_history,
        source_document_id=source_document_id,
        source_page_number=page_number,
        raster_bytes=page.raster_bytes,
    )
    assert extraction is not None, "Falta replay exacto F01.5 Casa Viri."

    level_names = [
        str(item.get("nombre"))
        for item in localization.get("niveles", [])
        if isinstance(item, dict) and item.get("nombre")
    ]
    assert level_names

    provider = _ReplayOnlyProvider()
    engine = QuantiaSpatialEngine(
        vision_provider=provider,
        gemini_evidence_adapter=GeminiEvidenceAdapter(
            provider=provider,
            semantic_history=GeminiSemanticHistory(history_path=None),
            prompt="LEGACY_REPLAY_ONLY",
            response_json_schema={},
        ),
    )
    result = engine.run(
        document_bytes=document_bytes,
        media_mime_type=str(config["media_mime_type"]),
        source_document_id=source_document_id,
        render_scale=render_scale,
        known_level_names_by_page={1: level_names},
        localization_payloads_by_page={1: localization},
        gemini_extraction_payloads_by_page={1: extraction},
        enable_gemini_discovery=False,
        enable_gemini_extraction=True,
    )
    by_name = {item.level_view.level_name: item for item in result.levels}
    return LoadedCase(
        case_id="casa_viri",
        case_name="Casa Viri",
        levels=tuple(LoadedLevel(name, by_name[name]) for name in level_names),
    )


def load_miguel_h() -> LoadedCase:
    if not MIGUEL_H_PDF_PATH.exists():
        pytest.skip(f"PDF Miguel H limpio no disponible: {MIGUEL_H_PDF_PATH}")

    case_dir = DATA_DIR / "miguel_h"
    localization = _read_json(case_dir / "miguel_h_gemini_localization_replay.json")["result"]["data"]
    extraction = _read_json(case_dir / "miguel_h_gemini_extraction_replay.json")
    level_names = ("Planta Baja", "Planta Alta")

    provider = _ReplayOnlyProvider()
    engine = QuantiaSpatialEngine(
        vision_provider=provider,
        gemini_evidence_adapter=GeminiEvidenceAdapter(
            provider=provider,
            semantic_history=GeminiSemanticHistory(history_path=None),
            prompt="LEGACY_REPLAY_ONLY",
            response_json_schema={},
        ),
    )
    result = engine.run(
        document_bytes=MIGUEL_H_PDF_PATH.read_bytes(),
        media_mime_type="application/pdf",
        source_document_id="MIGUEL_H_CLEAN_STANDARDIZED_REPLAY",
        render_scale=MIGUEL_H_RENDER_SCALE,
        known_level_names_by_page={1: list(level_names)},
        localization_payloads_by_page={1: localization},
        gemini_extraction_payloads_by_page={1: extraction},
        enable_gemini_discovery=False,
        enable_gemini_extraction=True,
    )
    by_name = {item.level_view.level_name: item for item in result.levels}
    return LoadedCase(
        case_id="miguel_h",
        case_name="Miguel H limpio",
        levels=tuple(LoadedLevel(name, by_name[name]) for name in level_names),
    )


def load_miguel_v() -> LoadedCase:
    case_dir = DATA_DIR / "miguel_v"
    config_path = _first_existing(case_dir / "case.json", case_dir / "input" / "case.json")
    config = _read_json(config_path)
    documents = config.get("documents")
    if not isinstance(documents, list):
        raise ValueError("case.json de Miguel V no contiene documents[].")
    semantic_history = _first_existing(
        case_dir / "replay" / "gemini_semantic_history.jsonl",
        case_dir / "output" / "gemini_semantic_history.jsonl",
    )

    source = PyMuPDFLevelSource()
    loaded: list[LoadedLevel] = []
    for item in documents:
        pdf_path = Path(str(item["pdf_path"]))
        if not pdf_path.exists():
            pytest.skip(f"PDF Miguel V no disponible: {pdf_path}")
        document_bytes = pdf_path.read_bytes()
        render_scale = float(config["render_scale"])
        page = source.read(document_bytes=document_bytes, render_scale=render_scale).pages[0]
        replay_payload, replay_source = _find_exact_extraction_replay(
            history_path=semantic_history,
            source_document_id=str(item["document_id"]),
            source_page_number=int(item["source_page_number"]),
            raster_bytes=page.raster_bytes,
        )
        if replay_payload is None:
            pytest.skip(f"No hay replay exacto para {item['document_id']}; no se hará llamada Gemini.")

        result = QuantiaSpatialEngine().run(
            document_bytes=document_bytes,
            media_mime_type=str(config["media_mime_type"]),
            source_document_id=str(item["document_id"]),
            render_scale=render_scale,
            known_level_names_by_page={1: [str(item["level_name"])]},
            isolated_pages={1},
            enable_gemini_discovery=False,
            enable_gemini_extraction=True,
            gemini_extraction_payloads_by_page={1: replay_payload},
        )
        assert len(result.levels) == 1
        loaded.append(LoadedLevel(str(item["level_name"]), result.levels[0], replay_source))

    return LoadedCase(
        case_id="miguel_v",
        case_name="Miguel V",
        levels=tuple(loaded),
    )


CASE_LOADERS: dict[str, Callable[[], LoadedCase]] = {
    "casa_viri": load_casa_viri,
    "miguel_h": load_miguel_h,
    "miguel_v": load_miguel_v,
}
