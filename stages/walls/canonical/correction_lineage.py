from __future__ import annotations


def apply_with_lineage(*, applier, graph, review, source_mapping):
    """Track ancestry using the actual correction applier and full delta prefixes.

    Full prefixes preserve generated ADD IDs and confidence/validity semantics.
    An empty ancestry denotes a new Call 2 wall; removed ancestry remains in the
    original mapping and review stored in the evidence bundle.
    """
    current = graph
    lineage = {key: list(value) for key, value in source_mapping.items()}
    for index, delta in enumerate(review.deltas):
        prefix = review.model_copy(update={"deltas": review.deltas[:index + 1]})
        next_graph = applier.apply(graph=graph, review=prefix)
        before = {w.id for w in current.walls}
        after = {w.id for w in next_graph.walls}
        if delta.confidence >= 0.60:
            if delta.action == "MERGE_WALLS":
                parents = [wid for wid in delta.wall_ids if wid in before]
                if len(parents) >= 2:
                    target = delta.wall_id or parents[0]
                    lineage[target] = sorted({sid for wid in parents for sid in lineage.get(wid, [])})
            elif delta.action == "SPLIT_WALL" and delta.wall_id in before and delta.wall_id not in after:
                for child in (f"{delta.wall_id}__A", f"{delta.wall_id}__B"):
                    if child in after:
                        lineage[child] = list(lineage.get(delta.wall_id, []))
            elif delta.action == "ADD_WALL" and delta.start_px is not None and delta.end_px is not None:
                if delta.wall_id in after:
                    lineage[delta.wall_id] = []
                for wid in after - before:
                    lineage[wid] = []
        lineage = {wid: lineage.get(wid, []) for wid in after}
        current = next_graph
    return current, lineage
