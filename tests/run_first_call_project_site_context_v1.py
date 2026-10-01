from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.prompts.quantia_extraction_prompt import QUANTIA_EXTRACTION_PROMPT
from app.schemas.gemini_extraction_transport import (
    get_gemini_extraction_transport_schema,
)
from app.quantia_spatialV1.models.level_view import (
    LevelView,
    LevelViewTransform,
    PixelBBox,
)
from app.quantia_spatialV1.models.project_site_context import ProjectSiteContext
from app.quantia_spatialV1.phase_01_level.pymupdf_level_source import PyMuPDFLevelSource
from app.quantia_spatialV1.phase_015_evidence.gemini_evidence_adapter import (
    GeminiEvidenceAdapter,
)
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_history import (
    GeminiSemanticHistory,
)
from app.quantia_spatialV1.phase_015_evidence.project_site_prompt_context import (
    build_project_site_context_prompt,
)
from app.quantia_spatialV1.phase_015_evidence.replay_signature import (
    ReplaySignatureValidator,
)


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "output" / "first_call_project_site_context_v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} debe contener un objeto JSON.")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "run"


def _full_page_view(*, page, document_id: str, level_name: str) -> LevelView:
    bbox = PixelBBox(
        x_min=0,
        y_min=0,
        x_max=page.raster_width_px,
        y_max=page.raster_height_px,
    )
    return LevelView(
        id=f"{document_id}__PAGE_{page.page_number}__{_slug(level_name).upper()}",
        level_name=level_name,
        source_document_id=document_id,
        source_page_number=page.page_number,
        source_bbox_px=bbox,
        source_page_width_px=page.raster_width_px,
        source_page_height_px=page.raster_height_px,
        raster_width_px=page.raster_width_px,
        raster_height_px=page.raster_height_px,
        raster_mime_type=page.raster_mime_type,
        raster_bytes=page.raster_bytes,
        transform=LevelViewTransform(
            offset_x_px=0,
            offset_y_px=0,
            source_page_width_px=page.raster_width_px,
            source_page_height_px=page.raster_height_px,
            local_width_px=page.raster_width_px,
            local_height_px=page.raster_height_px,
        ),
        state="DETECTADO",
        confidence=1.0,
    )


def run(
    *,
    pdf_path: Path,
    site_context_path: Path,
    document_id: str,
    level_name: str,
    page_number: int,
    render_scale: float,
    output_root: Path,
) -> Path:
    document_bytes = pdf_path.read_bytes()
    source = PyMuPDFLevelSource().read(
        document_bytes=document_bytes,
        render_scale=render_scale,
    )
    if not 1 <= page_number <= source.page_count:
        raise ValueError(f"Página {page_number} fuera del PDF ({source.page_count}).")

    page = source.pages[page_number - 1]
    context = ProjectSiteContext.coerce(_json(site_context_path))
    if context is None or not context.has_meaningful_data:
        raise ValueError("PROJECT_SITE_CONTEXT_V1 requiere datos declarados reales.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"{_slug(document_id)}__p{page_number}__{timestamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    schema = get_gemini_extraction_transport_schema()
    prompt = build_project_site_context_prompt(
        base_prompt=QUANTIA_EXTRACTION_PROMPT,
        project_site_context=context,
    )
    signature = ReplaySignatureValidator.build(
        prompt=prompt,
        schema=schema,
        raster_bytes=page.raster_bytes,
        raster_mime_type=page.raster_mime_type,
        project_site_context=context.as_prompt_payload(),
    )
    history_path = output / "gemini_semantic_history.jsonl"
    adapter = GeminiEvidenceAdapter(
        semantic_history=GeminiSemanticHistory(history_path=history_path),
        prompt=QUANTIA_EXTRACTION_PROMPT,
        response_json_schema=schema,
    )
    view = _full_page_view(
        page=page,
        document_id=document_id,
        level_name=level_name,
    )

    result = adapter.extract_page_with_trace(
        page_raster_bytes=page.raster_bytes,
        page_raster_mime_type=page.raster_mime_type,
        source_document_id=document_id,
        source_page_number=page_number,
        level_views=[view],
        replay_payload=None,
        project_site_context=context,
    )
    if result.replay_used:
        raise AssertionError("La corrida limpia no puede usar replay.")

    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    succeeded = next(
        row
        for row in reversed(records)
        if row.get("event") == "SUCCEEDED" and row.get("call_id") == result.call_id
    )

    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write(output / "schema.json", schema)
    _write(output / "project_site_context.json", context.as_prompt_payload())
    _write(output / "provider_raw_response.json", succeeded.get("provider_raw_response"))
    _write(output / "response_payload.json", result.response_payload)
    _write(output / "semantic_payloads.json", result.semantic_payloads_by_level_view)
    _write(
        output / "evidence.json",
        {
            key: [item.model_dump(mode="json") for item in values]
            for key, values in result.evidence_by_level_view.items()
        },
    )
    _write(
        output / "input_manifest.json",
        {
            "version": "FIRST_CALL_PROJECT_SITE_CONTEXT_V1",
            "document_id": document_id,
            "level_name": level_name,
            "page_number": page_number,
            "render_scale": render_scale,
            "source_pdf_sha256": hashlib.sha256(document_bytes).hexdigest(),
            "page_raster_sha256": hashlib.sha256(page.raster_bytes).hexdigest(),
            "page_raster_mime_type": page.raster_mime_type,
            "project_site_context": context.as_prompt_payload(),
            "request_signature": signature.as_dict(),
        },
    )
    _write(
        output / "run_manifest.json",
        {
            "status": "SUCCEEDED",
            "call_id": result.call_id,
            "provider": succeeded.get("provider"),
            "model": result.model,
            "fallback_used": result.fallback_used,
            "replay_used": result.replay_used,
            "timestamp_utc": succeeded.get("timestamp_utc"),
            "response_sha256": hashlib.sha256(
                json.dumps(
                    result.response_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "history_file": history_path.name,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Primera llamada Gemini limpia con PROJECT_SITE_CONTEXT_V1."
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--site-context", required=True, type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--level-name", required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--render-scale", type=float, default=0.7935)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    output = run(
        pdf_path=args.pdf,
        site_context_path=args.site_context,
        document_id=args.document_id,
        level_name=args.level_name,
        page_number=args.page,
        render_scale=args.render_scale,
        output_root=args.output_root,
    )
    print(f"FIRST_CALL_PROJECT_SITE_CONTEXT_V1: {output}")


if __name__ == "__main__":
    main()
