"""Rule-based stand-in for the LLM (backend: mock). Uses the structured `_ctx` attached to the prompt.

Lets the whole harness run end-to-end without a model, and gives an "oracle reasoning" upper bound
to compare the LLM against.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Dict, List


def _plan_bfs(g, scene, max_depth: int = 8) -> List[str]:
    from halter.reasoning.reasoner import apply_skill_symbolic, preconditions_hold
    from halter.types import ObjectNode, Relation, SceneGraph

    def done(t, a):
        e = SceneGraph(relations=[Relation(*x) for x in t],
                       nodes={k: ObjectNode(name=k, attributes=v) for k, v in a.items()})
        return not scene.target.missing(e)

    start = (frozenset(g.triples()), json.dumps({k: n.attributes for k, n in g.nodes.items()}, sort_keys=True))
    q = deque([(start, [])])
    seen = {start}
    while q:
        (t, a_s), plan = q.popleft()
        a = json.loads(a_s)
        if done(t, a):
            return plan
        if len(plan) >= max_depth:
            continue
        for sk in scene.skills.values():
            if preconditions_hold(t, a, sk):
                continue
            t2, a2 = apply_skill_symbolic(set(t), a, sk)
            key = (frozenset(t2), json.dumps(a2, sort_keys=True))
            if key not in seen:
                seen.add(key)
                q.append((key, plan + [sk.name]))
    return []


def mock_llm(messages: List[Dict]) -> str:
    ctx = next((m.get("_ctx") for m in messages if m.get("_ctx")), None)
    if ctx is None:
        return json.dumps({"error": "mock backend needs _ctx"})
    role = ctx["role"]
    if role == "score":
        g, task = ctx["graph"], ctx["task"]
        done = task.completed_prefix(g)
        return json.dumps({"completed_subtasks": done, "success": len(done) == len(task.subtasks),
                           "rationale": "rule-based"})
    if role == "plan":
        plan = _plan_bfs(ctx["graph"], ctx["scene"])
        return json.dumps({"plan": plan, "feasible": True, "rationale": "BFS over skill effects"})
    if role == "verify_skill":
        sk, after = ctx["skill"], ctx["after"]
        ok = all(after.has(*t) for t in sk.effects_add) and not any(after.has(*t) for t in sk.effects_del)
        for obj, d in sk.attribute_effects.items():
            n = after.nodes.get(obj)
            ok &= n is not None and all(n.attributes.get(k) == v for k, v in d.items())
        return json.dumps({"success": bool(ok), "rationale": "rule-based effect check"})
    if role == "verify_final":
        miss = ctx["scene"].target.missing(ctx["graph"])
        return json.dumps({"success": not miss, "missing": miss, "rationale": "rule-based"})
    return "{}"
