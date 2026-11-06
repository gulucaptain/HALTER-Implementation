"""Serialise a scene graph into text for the LLM: object records + relation triples."""
from __future__ import annotations

import json

from halter.types import SceneGraph


def graph_to_text(g: SceneGraph, include_geometry: bool = True) -> str:
    lines = ["Objects:"]
    for n in g.nodes.values():
        rec = {"name": n.name, "visible": n.visible}
        if include_geometry and n.centroid is not None:
            rec["centroid_m"] = [round(float(v), 3) for v in n.centroid]
            rec["size_m"] = [round(float(v), 3) for v in (n.bbox_max - n.bbox_min)]
        if n.attributes:
            rec.update(n.attributes)
        lines.append("  " + json.dumps(rec, ensure_ascii=False))
    lines.append("Relations:")
    if not g.relations:
        lines.append("  (none)")
    for r in g.relations:
        lines.append(f"  ({r.subject}, {r.predicate}, {r.object})")
    return "\n".join(lines)


def graph_diff(before: SceneGraph, after: SceneGraph) -> str:
    b, a = before.triples(), after.triples()
    added = sorted(a - b)
    removed = sorted(b - a)
    attr_changes = []
    for k, n in after.nodes.items():
        if k in before.nodes and before.nodes[k].attributes != n.attributes:
            attr_changes.append(f"{k}: {before.nodes[k].attributes} -> {n.attributes}")
    fmt = lambda ts: ", ".join(f"({a}, {b}, {c})" for a, b, c in ts) or "none"
    out = ["Added relations: " + fmt(added),
           "Removed relations: " + fmt(removed),
           "Attribute changes: " + ("; ".join(attr_changes) or "none")]
    return "\n".join(out)
