#!/usr/bin/env python
"""Run an evaluation campaign with HALTER.

  python scripts/run_evaluation.py --config configs/sim.yaml                       # symbolic sim, mock LLM
  python scripts/run_evaluation.py --config configs/sim.yaml --llm configs/llm/api_dashscope.yaml
  python scripts/run_evaluation.py --config configs/real_franka.yaml --task configs/tasks/drawer_store.yaml
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halter.factory import build_real, build_sim, load_yaml, make_llm_from  # noqa: E402
from halter.harness import metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", action="append", help="override task list (repeatable)")
    ap.add_argument("--llm", help="override LLM config")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--log_dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_yaml(args.config)
    if args.llm:
        cfg["llm"] = args.llm
    if args.log_dir:
        cfg.setdefault("harness", {})["log_dir"] = args.log_dir
    tasks = args.task or cfg["tasks"]
    n = args.episodes or cfg.get("episodes", 25)
    llm = make_llm_from(cfg["llm"])

    all_eps = []
    for i, t in enumerate(tasks):
        if cfg.get("mode", "sim") == "sim":
            h, _ = build_sim(cfg, t, llm=llm, seed=cfg.get("seed", 0) + i)
        else:
            h = build_real(cfg, t, llm=llm)
        if os.path.exists(h.log_path):
            os.remove(h.log_path)
        eps = h.run(n)
        all_eps += [e.to_dict() for e in eps]
        print(f"[{h.task.name}]", json.dumps(metrics.summarize([e.to_dict() for e in eps])))

    summary = {"overall": metrics.summarize(all_eps), "per_task": metrics.per_task(all_eps),
               "llm_calls": llm.n_calls, "llm_mean_latency_s": round(llm.total_latency / max(llm.n_calls, 1), 2)}
    out = os.path.join(h.cfg.log_dir, "summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary["overall"], indent=2))
    print("summary written to", out)


if __name__ == "__main__":
    main()
