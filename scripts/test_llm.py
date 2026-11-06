#!/usr/bin/env python
"""Smoke-test an LLM backend on HALTER's three prompts with a synthetic scene graph.

  export DASHSCOPE_API_KEY=sk-...; python scripts/test_llm.py --llm configs/llm/api_dashscope.yaml
  bash scripts/serve_qwen_vllm.sh &;  python scripts/test_llm.py --llm configs/llm/local_vllm.yaml
Compares the LLM's answers with the rule-based oracle.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halter.factory import _p, make_llm_from  # noqa: E402
from halter.llm.client import LLMConfig, make_llm  # noqa: E402
from halter.reasoning.reasoner import Reasoner  # noqa: E402
from halter.reasoning.reference_cases import ReferenceSet  # noqa: E402
from halter.reasoning.task_spec import load_scene, load_task  # noqa: E402
from halter.sim.symbolic import SymbolicWorld  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--task", default="configs/tasks/drawer_store.yaml")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    task = load_task(_p(args.task))
    scene = load_scene(_p(f"configs/scenes/{task.scene}.yaml"))
    refs = ReferenceSet.load(task.reference_cases) if os.path.exists(task.reference_cases) else ReferenceSet()
    llm = make_llm_from(args.llm)
    oracle = make_llm(LLMConfig(backend="mock"))
    R, O = Reasoner(llm, scene, task, refs), Reasoner(oracle, scene, task, refs)
    agree = {"score": 0, "plan_valid": 0, "final": 0}
    for i in range(args.n):
        w = SymbolicWorld(scene, seed=100 + i)
        w.rollout(task, p_subtask=0.6)
        g = w.observe()
        t0 = time.time()
        s, so = R.score(g), O.score(g)
        p, _ = R.plan_with_check(g)
        f, fo = R.verify_final(g), O.verify_final(g)
        agree["score"] += sorted(s.completed_subtasks) == sorted(so.completed_subtasks)
        agree["plan_valid"] += p.valid
        agree["final"] += f.ok == fo.ok
        print(f"--- case {i} ({time.time() - t0:.1f}s)\n relations: {sorted(g.triples())}\n"
              f" score: {s.completed_subtasks} (oracle {so.completed_subtasks})\n"
              f" plan: {p.steps} valid={p.valid} {p.invalid_reason}\n final_ok: {f.ok} (oracle {fo.ok})")
    print("agreement:", {k: f"{v}/{args.n}" for k, v in agree.items()},
          f"mean latency {llm.total_latency / max(llm.n_calls, 1):.2f}s/call")


if __name__ == "__main__":
    main()
