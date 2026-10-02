import json
import socket

import pytest

from app.quantia_spatialV1.stages.walls.adaptive.contracts import MultimodalWallReview
from app.quantia_spatialV1.stages.walls.canonical import CanonicalWallGraphFinalizer
from app.quantia_spatialV1.tests.call2_interface04_export import export_review
from app.quantia_spatialV1.tests.test_quantia_spatial_canonical_wallgraph_contract_v1_synthetic import inputs


def test_delivery_uses_final_graph_and_preserves_uncertainty(inputs, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline test attempted network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    view, runtime, post = inputs
    graph = CanonicalWallGraphFinalizer().finalize(level_view=view, runtime=runtime, post_filter=post, evidence=[]).graph
    review = MultimodalWallReview(level_view_id=view.id, graph_state="REVIEW",
        deltas=[{"action": "REMOVE_WALL", "wall_id": "W1", "confidence": 0.1, "reason": "must reject"}],
        non_wall_architectural_regions=[{"bbox_px": [50, 35, 70, 45], "family_hint": "DOOR", "confidence": .95, "reason": "candidate"}])
    payload = export_review(view=view, graph=graph, runtime=runtime, post=post,
                            evidence=[], review=review, output=tmp_path)
    assert len(payload["muros"]) == len(graph.walls)
    assert payload["puertas"] == [] and payload["call2Candidates"]
    assert not payload["readiness"]["quantification"]
    for wall, source in zip(payload["muros"], graph.walls):
        assert wall["thicknessM"] == source.thickness_px / graph.px_per_m
        assert wall["start"]["x"] == source.start_px[0] / graph.px_per_m
        assert wall["segmentos"][0]["geometria"]["raster"]["vertices"][0]["x"] == source.start_px[0]
        assert wall["heightM"] is None and wall["confirmed"] is False
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["delta_items"][0]["accepted"] is False
    assert json.loads((tmp_path / "integrity.json").read_text())["valid"]
    bundle = json.loads((tmp_path / "evidence_bundle.json").read_text())
    assert set(bundle["final_source_mapping"]) == {w["id"] for w in payload["muros"]}


def test_rejects_review_from_another_level(inputs, tmp_path):
    view, runtime, post = inputs
    with pytest.raises(ValueError, match="another LevelView"):
        export_review(view=view, graph=post.filtered_wall_graph, runtime=runtime, post=post,
                      evidence=[], review=MultimodalWallReview(level_view_id="OTHER", graph_state="VALID"), output=tmp_path)


def test_multimodal_request_to_delivery_offline(inputs, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import cv2
    import numpy as np
    from app.quantia_spatialV1.stages.walls.adaptive.multimodal_review import WallGraphMultimodalReviewer

    def no_network(*args, **kwargs):
        raise AssertionError("Offline test attempted network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    view, runtime, post = inputs
    view = view.model_copy(update={"raster_bytes": cv2.imencode(".png", np.full((200, 200, 3), 255, dtype=np.uint8))[1].tobytes()})
    graph = CanonicalWallGraphFinalizer().finalize(level_view=view, runtime=runtime, post_filter=post, evidence=[]).graph
    calls = []

    class FixtureProvider:
        def analyze(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["media_mime_type"] == "image/png"
            image = cv2.imdecode(np.frombuffer(kwargs["media_bytes"], dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image.shape[:2] == (496, 200)
            assert "deltas" in kwargs["response_json_schema"]["properties"]
            return SimpleNamespace(data={"level_view_id": view.id, "graph_state": "REVIEW", "deltas": [],
                "non_wall_architectural_regions": [], "unresolved_regions": [], "summary": "Synthetic provider"},
                provider="fixture", model="offline", fallback_used=False, raw={})

    review = WallGraphMultimodalReviewer(provider=FixtureProvider(), enable_replay=False,
        persist_input_artifact=False).review(level_view=view, graph=graph)
    payload = export_review(view=view, graph=graph, runtime=runtime, post=post,
                           evidence=[], review=review, output=tmp_path)
    assert len(calls) == 1
    assert {w["id"] for w in payload["muros"]} == {w.id for w in graph.walls}
