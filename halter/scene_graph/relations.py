"""Geometric inference of spatial relations {on, in, under} and articulated-object states.

The paper infers relations "from geometry" over per-object point clouds; it does not publish thresholds.
Everything here is configurable via `RelationConfig` (see configs/scenes/*.yaml -> relations).
Frame convention: world z axis points up, metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from halter.types import ObjectNode, Relation


@dataclass
class RelationConfig:
    z_contact_tol: float = 0.025        # |bottom(A) - top(B)| for "on"
    xy_overlap_on: float = 0.5          # fraction of A's XY footprint over B for "on"
    xy_overlap_in: float = 0.6          # fraction of A's XY footprint inside container B for "in"
    in_rim_margin: float = 0.01         # A bottom must be below B's rim minus this
    under_gap: float = 0.02             # top(A) <= bottom(B) + gap for "under"
    xy_overlap_under: float = 0.3
    containers: List[str] = field(default_factory=list)   # objects that can hold others (pot, drawer, sink, plate?)
    supports: List[str] = field(default_factory=list)     # surfaces that things can be "on" (table, stove, plate)
    table_name: str = "table"
    table_z: Optional[float] = 0.0      # implicit table plane if the table itself is not segmented
    # container interiors given explicitly (useful for drawers where only the front is visible):
    # {"lower_drawer": {"min": [x,y,z], "max": [x,y,z]}} in world frame, optionally moving with the drawer
    container_regions: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # articulated: {"lower_drawer": {"axis": [1,0,0], "closed_pos": 0.45, "open_threshold": 0.05,
    #                                "track": "centroid"}}
    articulated: Dict[str, Dict] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict) -> "RelationConfig":
        return RelationConfig(**(d or {}))


def _xy_overlap_frac(a: ObjectNode, b: ObjectNode, shrink_b: float = 0.0) -> float:
    """Fraction of a's XY bbox area that lies inside b's XY bbox."""
    a0, a1 = a.bbox_min[:2], a.bbox_max[:2]
    b0, b1 = b.bbox_min[:2] + shrink_b, b.bbox_max[:2] - shrink_b
    lo, hi = np.maximum(a0, b0), np.minimum(a1, b1)
    inter = np.prod(np.clip(hi - lo, 0, None))
    area = np.prod(np.clip(a1 - a0, 1e-6, None))
    return float(inter / area)


def _region_node(name: str, region: Dict[str, List[float]], offset: np.ndarray) -> ObjectNode:
    n = ObjectNode(name=name)
    n.bbox_min = np.asarray(region["min"], float) + offset
    n.bbox_max = np.asarray(region["max"], float) + offset
    n.centroid = (n.bbox_min + n.bbox_max) / 2
    return n


def articulated_states(nodes: Dict[str, ObjectNode], cfg: RelationConfig) -> Dict[str, Dict[str, float]]:
    """Set nodes[...].attributes['state'] in {open, closed}; return displacement per articulated object."""
    disp = {}
    for name, spec in cfg.articulated.items():
        n = nodes.get(name)
        if n is None or n.centroid is None:
            continue
        axis = np.asarray(spec.get("axis", [1, 0, 0]), float)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        pos = float(n.centroid @ axis)
        d = pos - float(spec.get("closed_pos", 0.0))
        disp[name] = d
        n.attributes["state"] = "open" if abs(d) > float(spec.get("open_threshold", 0.05)) else "closed"
    return disp


def infer_relations(nodes: Dict[str, ObjectNode], cfg: RelationConfig) -> List[Relation]:
    disp = articulated_states(nodes, cfg)
    geo = {k: n for k, n in nodes.items() if n.bbox_min is not None}

    # Container nodes used for "in" tests (explicit interior region overrides the observed bbox)
    container_nodes: Dict[str, ObjectNode] = {}
    for c in cfg.containers:
        if c in cfg.container_regions:
            off = np.zeros(3)
            if c in cfg.articulated and c in disp:
                ax = np.asarray(cfg.articulated[c].get("axis", [1, 0, 0]), float)
                off = ax / (np.linalg.norm(ax) + 1e-9) * disp[c]
            container_nodes[c] = _region_node(c, cfg.container_regions[c], off)
        elif c in geo:
            container_nodes[c] = geo[c]

    rels: List[Relation] = []
    placed = set()
    # 1) "in" has priority over "on"
    for a_name, a in geo.items():
        best, best_frac = None, 0.0
        for c_name, c in container_nodes.items():
            if c_name == a_name:
                continue
            frac = _xy_overlap_frac(a, c)
            below_rim = a.bbox_min[2] < c.bbox_max[2] - cfg.in_rim_margin
            above_floor = a.bbox_min[2] > c.bbox_min[2] - cfg.z_contact_tol
            # a closed drawer still "contains" its object; nothing else special-cased
            if frac >= cfg.xy_overlap_in and below_rim and above_floor and frac > best_frac:
                best, best_frac = c_name, frac
        if best is not None:
            rels.append(Relation(a_name, "in", best))
            placed.add(a_name)

    # 2) "on": supported by another object's top surface
    for a_name, a in geo.items():
        if a_name in placed or a_name == cfg.table_name:
            continue
        cands = []
        for b_name, b in geo.items():
            if b_name == a_name or (cfg.supports and b_name not in cfg.supports and b_name not in cfg.containers):
                continue
            dz = a.bbox_min[2] - b.bbox_max[2]
            if abs(dz) <= cfg.z_contact_tol and _xy_overlap_frac(a, b) >= cfg.xy_overlap_on \
                    and a.centroid[2] > b.centroid[2]:
                cands.append((abs(dz), b_name))
        if cands:
            rels.append(Relation(a_name, "on", min(cands)[1]))
            placed.add(a_name)
        elif cfg.table_z is not None and abs(a.bbox_min[2] - cfg.table_z) <= cfg.z_contact_tol * 2:
            rels.append(Relation(a_name, "on", cfg.table_name))
            placed.add(a_name)

    # 3) "under": a is below b with xy overlap (e.g. carrot under the lid, pan under the pot)
    for a_name, a in geo.items():
        for b_name, b in geo.items():
            if a_name == b_name:
                continue
            if a.bbox_max[2] <= b.bbox_min[2] + cfg.under_gap and _xy_overlap_frac(a, b) >= cfg.xy_overlap_under \
                    and not any(r.as_tuple() == (b_name, "on", a_name) for r in rels):
                rels.append(Relation(a_name, "under", b_name))
    return rels
