from __future__ import annotations

from typing import Any

CALL2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "level_view_id": {"type": "string"},
        "graph_state": {"type": "string", "enum": ["VALID", "PARTIAL", "REVIEW"]},
        "deltas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": [
                        "ADD_WALL", "REMOVE_WALL", "EXTEND_WALL", "TRIM_WALL",
                        "REPOSITION_WALL", "MERGE_WALLS", "SPLIT_WALL"
                    ]},
                    "wall_id": {"type": "string"},
                    "wall_ids": {"type": "array", "items": {"type": "string"}},
                    "start_px": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "end_px": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "endpoint": {"type": "string", "enum": ["START", "END"]},
                    "new_point_px": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "role_hint": {"type": "string", "enum": ["PERIMETER", "DIVIDER", "REVIEW"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": ["action", "confidence", "reason"],
            },
        },
        "non_wall_architectural_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_px": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "family_hint": {"type": "string", "enum": [
                        "DOOR", "WINDOW", "FLOOR_TO_CEILING_GLAZING",
                        "RAILING_GUARDRAIL", "STAIR", "STAIR_HANDRAIL",
                        "ROOF_COVER", "TERRACE", "OPENING_CLOSURE",
                        "SLAB_EDGE_LEVEL_CHANGE", "COLUMN", "OTHER", "UNCERTAIN"
                    ]},
                    "host_wall_id": {"type": "string"},
                    "host_wall_continuity": {"type": "boolean"},
                    "solid_wall_present": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": ["bbox_px", "family_hint", "confidence", "reason"],
            },
        },
        "unresolved_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_px": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": ["bbox_px", "confidence", "reason"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": [
        "level_view_id", "graph_state", "deltas",
        "non_wall_architectural_regions", "unresolved_regions", "summary"
    ],
}
