"""One-LevelView Call 2 capture and Interfaz 04 export. Live calls are explicit.

From Backend: python -B -m app.quantia_spatialV1.tests.run_call2_interface04 --live
Without --live, only an exact prior replay is accepted. No provider fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from app.quantia_spatialV1.adaptive_reconstruction.multimodal_review import WallGraphMultimodalReviewer, CALL2_SCHEMA
from app.quantia_spatialV1.providers import build_call2_provider
from app.quantia_spatialV1.providers.call2_providers import _EnvKeyResolver
from app.quantia_spatialV1.process_engine import QuantiaSpatialV1ProcessEngine
from app.quantia_spatialV1.tests.call2_editor_snapshot import ROOT, load_snapshot
from app.quantia_spatialV1.tests.test_quantia_spatial_call2_provider_smoke_v1 import SMOKE_SCHEMA


class ReplayOnly:
    def analyze(self, **kwargs):
        raise RuntimeError("Exact Call 2 replay unavailable; use --live explicitly")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Synthetic no-change response; never contacts an API")
    parser.add_argument("--case", default="casa_viri")
    parser.add_argument("--level", default="planta_alta")
    args = parser.parse_args()
    if args.live and args.offline:
        parser.error("--live and --offline are mutually exclusive")
    name = _EnvKeyResolver.get("QUANTIA_CALL2_PROVIDER") or "groq"
    # Non-secret settings are honored from the same config sources as credentials.
    for key in ("QUANTIA_GROQ_CALL2_MODEL", "QUANTIA_MISTRAL_CALL2_MODEL", "QUANTIA_CALL2_MAX_OUTPUT_TOKENS"):
        if key not in os.environ and _EnvKeyResolver.get(key):
            os.environ[key] = _EnvKeyResolver.get(key)
    configured = build_call2_provider(name)
    provider = configured if args.live else ReplayOnly()
    model = getattr(configured, "model", None)
    view, graph, runtime, post, evidence, stem = load_snapshot(args.case, args.level)
    model_key = hashlib.sha256(str(model).encode()).hexdigest()[:12]
    output = ROOT / "tests/output/call2_interface04" / ("offline" if args.offline else name) / model_key / stem
    output.mkdir(parents=True, exist_ok=True)
    (output / "original.png").write_bytes(view.raster_bytes)
    reviewer = WallGraphMultimodalReviewer(provider=provider, history_path=output / "history.jsonl")
    context = QuantiaSpatialV1ProcessEngine._build_call2_filter_context(post)
    (output / "call2_input.png").write_bytes(reviewer.build_review_image(level_view=view, graph=graph))
    (output / "prompt.txt").write_text(reviewer.build_prompt(graph=graph, level_view=view, filter_context=context), encoding="utf-8")
    write_json(output / "schema.json", CALL2_SCHEMA)
    write_json(output / "input_graph.json", graph.model_dump(mode="json"))
    identity = {"provider": name, "model": model, "source_sha256": hashlib.sha256(view.raster_bytes).hexdigest()}
    print(f"Call 2 prepared: {stem}; provider={name}; live={args.live}", flush=True)
    try:
        if args.live:
            if hasattr(provider, "check_model_available"):
                availability = provider.check_model_available()
                write_json(output / "availability.json", availability)
                if not availability["available"]:
                    raise RuntimeError("Configured model unavailable; no fallback performed")
            smoke_path = output / "smoke.json"
            saved = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else {}
            if saved.get("identity") != identity:
                smoke = provider.analyze(
                    prompt=f"Inspect this architectural plan. Return level_view_id exactly {view.id!r}, contains_architectural_plan and status OK.",
                    media_bytes=view.raster_bytes, media_mime_type="image/png", response_json_schema=SMOKE_SCHEMA)
                if smoke.data.get("level_view_id") != view.id or smoke.data.get("status") != "OK" or smoke.data.get("contains_architectural_plan") is not True:
                    raise RuntimeError("Vision/schema smoke failed")
                write_json(smoke_path, {"identity": identity, "response": smoke.data})
            print("Provider vision/schema smoke passed", flush=True)
        if args.offline:
            from app.quantia_spatialV1.adaptive_reconstruction.contracts import MultimodalWallReview
            review = MultimodalWallReview(level_view_id=view.id, graph_state="REVIEW",
                summary="OFFLINE fixture: no model executed; no semantic validation claimed")
        else:
            review = reviewer.review(level_view=view, graph=graph, filter_context=context)
        write_json(output / "review.json", review.model_dump(mode="json"))
        print(f"Call 2 response: {len(review.deltas)} deltas, {len(review.non_wall_architectural_regions)} architectural candidates", flush=True)
        if not args.capture_only:
            from app.quantia_spatialV1.tests.call2_interface04_export import export_review
            export_review(view=view, graph=graph, runtime=runtime, post=post, evidence=evidence, review=review, output=output)
        write_json(output / "run_status.json", {**identity, "status": "CAPTURED" if args.capture_only else "EXPORTED", "mode": "OFFLINE_FIXTURE" if args.offline else "LIVE_OR_EXACT_REPLAY" if args.live else "EXACT_REPLAY"})
    except Exception as exc:
        message = str(exc)
        if getattr(provider, "api_key", None):
            message = message.replace(provider.api_key, "[REDACTED]")
        write_json(output / "run_status.json", {**identity, "status": "FAILED", "error_type": type(exc).__name__, "error": message})
        raise RuntimeError(message) from None
    print(output, flush=True)


if __name__ == "__main__":
    main()
