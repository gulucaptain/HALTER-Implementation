#!/usr/bin/env python
"""Capture the current real scene graph (cameras + GroundedSAM) and save it as a reference-case stub.

  python scripts/capture_graph.py --config configs/real_franka.yaml --task configs/tasks/drawer_store.yaml --out case.yaml
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from halter.factory import _p, load_yaml  # noqa: E402
from halter.perception.cameras import build_cameras  # noqa: E402
from halter.perception.segmenter import GroundedSAMSegmenter  # noqa: E402
from halter.reasoning.task_spec import load_scene, load_task  # noqa: E402
from halter.scene_graph.builder import SceneGraphBuilder  # noqa: E402
from halter.scene_graph.serialize import graph_to_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3, help="updates to accumulate (occlusion memory)")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    task = load_task(_p(a.task))
    scene = load_scene(_p(f"configs/scenes/{task.scene}.yaml"))
    cams = build_cameras([{k: (_p(v) if k == "extrinsics" and isinstance(v, str) else v)
                           for k, v in c.items() if k != "wrist"} for c in cfg["cameras"] if not c.get("wrist")])
    sg = cfg.get("segmenter", {})
    seg = GroundedSAMSegmenter(sg.get("dino_id", "IDEA-Research/grounding-dino-base"),
                               sg.get("sam_id", "facebook/sam-vit-huge"), sg.get("device", "cuda"),
                               aliases=scene.segmenter_aliases)
    b = SceneGraphBuilder(cams, seg, scene.objects, scene.relation_cfg, workspace=scene.workspace,
                          static_objects=scene.static_objects)
    for _ in range(a.frames):
        g = b.update()
    print(graph_to_text(g))
    yaml.safe_dump({"id": "TODO", "graph": g.to_dict(), "completed_subtasks": ["TODO"], "plan": ["TODO"],
                    "note": "TODO"}, open(a.out, "w"), sort_keys=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
