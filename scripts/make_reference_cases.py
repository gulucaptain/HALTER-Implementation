#!/usr/bin/env python
"""Generate reference-case files (paper: 1 success + 3 failure cases per task; 4 success cases per skill).

By default cases come from the symbolic simulator (relations only, no geometry) so the pipeline runs out
of the box. For the real robot, REPLACE them with graphs captured on your rig:
    python scripts/capture_graph.py --config configs/real_franka.yaml --task ... --out case.yaml
and paste them into configs/reference_cases/<task>.yaml.
"""
import os
import random
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from halter.factory import _p  # noqa: E402
from halter.reasoning.mock_policy import _plan_bfs  # noqa: E402
from halter.reasoning.task_spec import load_scene, load_task  # noqa: E402
from halter.sim.symbolic import SymbolicSkillExecutor, SymbolicWorld  # noqa: E402


def gdict(g):
    d = g.to_dict()
    d.pop("timestamp", None)
    for o in d["objects"]:
        for k in ("centroid", "bbox_min", "bbox_max"):
            o.pop(k, None)
    return d


def task_cases(task, scene, seed=0):
    rng = random.Random(seed)
    cases, seen = [], set()
    w = SymbolicWorld(scene, seed=seed)
    # success: all subtasks done
    w.rollout(task, p_subtask=1.0)
    g = w.ground_truth()
    cases.append({"id": "success", "graph": gdict(g), "completed_subtasks": task.completed_prefix(g),
                  "success": True, "plan": _plan_bfs(g, scene), "note": "all subtasks completed"})
    seen.add(frozenset(g.triples()) | {str(g.to_dict()["objects"])})
    tries = 0
    while len(cases) < 4 and tries < 500:
        tries += 1
        w = SymbolicWorld(scene, seed=rng.randint(0, 10**6), p_rollout_drop=0.6)
        w.rollout(task, p_subtask=0.5)
        g = w.ground_truth()
        done = task.completed_prefix(g)
        key = frozenset(g.triples()) | {str(g.to_dict()["objects"])}
        if len(done) == len(task.subtasks) or key in seen:
            continue
        seen.add(key)
        cases.append({"id": f"failure_{len(cases)}", "graph": gdict(g), "completed_subtasks": done,
                      "success": False, "plan": _plan_bfs(g, scene),
                      "note": f"{len(done)}/{len(task.subtasks)} subtasks completed"})
    return cases


def skill_cases(scene, task, n=4, seed=0):
    out = {}
    for name, sk in scene.skills.items():
        cs, tries = [], 0
        rng = random.Random(seed)
        while len(cs) < n and tries < 300:
            tries += 1
            w = SymbolicWorld(scene, seed=rng.randint(0, 10**6), p_skill_success=1.0, p_rollout_drop=0.6)
            w.rollout(task, p_subtask=0.5)
            plan = _plan_bfs(w.ground_truth(), scene)
            ex = SymbolicSkillExecutor(w)
            for step in plan:
                before = w.ground_truth()
                ex.execute(scene.skills[step])
                if step == name:
                    cs.append({"before": gdict(before), "after": gdict(w.ground_truth()), "success": True,
                               "note": f"{name} achieved {sk.effects_add or sk.attribute_effects}"})
                    break
        out[name] = cs[:n]
    return out


def main():
    tdir = _p("configs/tasks")
    os.makedirs(_p("configs/reference_cases"), exist_ok=True)
    for f in sorted(os.listdir(tdir)):
        task = load_task(os.path.join(tdir, f))
        scene = load_scene(_p(f"configs/scenes/{task.scene}.yaml"))
        doc = {"task_cases": task_cases(task, scene), "skill_cases": skill_cases(scene, task)}
        path = _p(f"configs/reference_cases/{task.name}.yaml")
        with open(path, "w") as fh:
            fh.write("# AUTO-GENERATED from the symbolic simulator. Replace with graphs captured on your robot.\n")
            yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, default_flow_style=None)
        print("wrote", path, len(doc["task_cases"]), "task cases")


if __name__ == "__main__":
    main()
