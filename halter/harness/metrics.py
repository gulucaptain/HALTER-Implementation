"""Aggregate episode logs into the paper's metrics (mean ± binomial standard error, in %)."""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List


def _pm(xs: List[bool]) -> Dict:
    n = len(xs)
    if n == 0:
        return {"mean": None, "se": None, "n": 0}
    p = sum(xs) / n
    return {"mean": round(100 * p, 1), "se": round(100 * math.sqrt(p * (1 - p) / n), 1), "n": n}


def load(paths: Iterable[str]) -> List[Dict]:
    eps = []
    for p in paths:
        with open(p) as f:
            eps += [json.loads(l) for l in f if l.strip()]
    return eps


def summarize(eps: List[Dict]) -> Dict:
    reset = [e["reset_success"] for e in eps]
    scoring = [e["score"]["correct"] for e in eps if e.get("score") and "correct" in e["score"]]
    plan_valid = [e["plan"]["first_attempt_valid"] for e in eps if e.get("plan")]
    verify = [r["verify_correct"] for e in eps for r in e.get("skill_results", []) if "verify_correct" in r]
    verify += [e["final_verify"]["correct"] for e in eps if e.get("final_verify") and "correct" in e["final_verify"]]
    interventions = sum(e["human_interventions"] for e in eps)
    human_s = sum(e["human_time_s"] for e in eps)
    wall = [e["wall_time_s"] for e in eps]
    return {
        "episodes": len(eps),
        "reset_sr": _pm(reset),
        "scoring_acc": _pm(scoring),
        "verification_acc": _pm(verify),
        "plan_validity": _pm(plan_valid),
        "interventions": interventions,
        "episodes_between_interventions": round(len(eps) / max(interventions, 1), 2),
        "operator_time_min": round(human_s / 60, 1),
        "s_per_episode": round(sum(wall) / max(len(wall), 1), 1),
    }


def per_task(eps: List[Dict]) -> Dict[str, Dict]:
    tasks = sorted({e["task"] for e in eps})
    return {t: summarize([e for e in eps if e["task"] == t]) for t in tasks}
