"""Scene graph from LIBERO (robosuite/MuJoCo) privileged state, for simulation experiments.

Uses body positions + geom AABBs from MuJoCo instead of camera perception, then the same geometric
relation inference as the real pipeline. Swap in `SceneGraphBuilder` with rendered RGB-D to test perception.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from halter.scene_graph.relations import RelationConfig, infer_relations
from halter.types import ObjectNode, SceneGraph


def body_aabb(sim, body_name: str) -> np.ndarray:
    """World-frame AABB corners of all geoms attached to a body (and its children)."""
    m, d = sim.model, sim.data
    bid = m.body_name2id(body_name)
    pts = []
    for gid in range(m.ngeom):
        b = m.geom_bodyid[gid]
        while b != 0 and b != bid:
            b = m.body_parentid[b]
        if b != bid:
            continue
        R = d.geom_xmat[gid].reshape(3, 3)
        c = d.geom_xpos[gid]
        h = m.geom_rbound[gid] if m.geom_size[gid].sum() == 0 else m.geom_size[gid]
        h = np.resize(h, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(c + R @ (h * np.array([sx, sy, sz])))
    return np.array(pts)


def libero_scene_graph(env, objects: Dict[str, str], rel_cfg: RelationConfig) -> SceneGraph:
    """objects: {graph_name: mujoco_body_name}"""
    sim = env.sim if hasattr(env, "sim") else env.env.sim
    nodes = {}
    for name, body in objects.items():
        pts = body_aabb(sim, body)
        if len(pts) == 0:
            continue
        n = ObjectNode(name=name)
        n.update_geometry(pts)
        nodes[name] = n
    return SceneGraph(nodes=nodes, relations=infer_relations(nodes, rel_cfg))
