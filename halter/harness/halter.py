"""HALTER main loop: rollout -> scene graph -> score -> plan -> execute skills with verification -> final check.

For each episode:
  1. run the policy under evaluation, tracking the scene graph online
  2. LLM scores the rollout (completed-subtask fraction)
  3. LLM plans a reset as a sequence of atomic skills (symbolically validated; optional re-plan)
  4. execute skills; after each skill, LLM verifies that the expected effect appears in the new graph
  5. after the plan, LLM checks the final graph against the target initial state
  6. if any check fails -> request human help (counted as an intervention)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from halter.harness.interfaces import GraphSource, HumanInterface, RolloutRunner
from halter.reasoning.reasoner import Reasoner
from halter.reasoning.task_spec import SceneSpec, TaskSpec
from halter.skills.base import SkillLibrary
from halter.types import EpisodeLog, SceneGraph

log = logging.getLogger(__name__)


@dataclass
class HarnessConfig:
    skill_retries: int = 0          # paper: request human help on failure (0 retries)
    replan_on_skill_failure: bool = False
    max_replans: int = 1            # re-prompt the planner with validator feedback
    settle_s: float = 0.0           # wait after a skill before taking the verification snapshot
    log_dir: str = "runs/default"


class HALTER:
    def __init__(self, scene: SceneSpec, task: TaskSpec, reasoner: Reasoner, skills: SkillLibrary,
                 graph: GraphSource, rollout: RolloutRunner, human: HumanInterface,
                 cfg: HarnessConfig = HarnessConfig(),
                 ground_truth: Optional[Callable[[], SceneGraph]] = None):
        self.scene, self.task, self.R, self.skills = scene, task, reasoner, skills
        self.graph, self.rollout, self.human, self.cfg = graph, rollout, human, cfg
        self.gt = ground_truth                     # simulator only; real robot uses operator labels
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.log_path = os.path.join(cfg.log_dir, f"{task.name}.jsonl")

    def _snap(self) -> SceneGraph:
        if self.cfg.settle_s:
            time.sleep(self.cfg.settle_s)
        return self.graph.snapshot()

    def _intervene(self, ep: EpisodeLog, reason: str) -> None:
        ep.human_interventions += 1
        ep.notes.append(f"human: {reason}")
        ep.human_time_s += self.human.request_reset(reason)

    def run_episode(self, idx: int) -> EpisodeLog:
        t0 = time.time()
        ep = EpisodeLog(episode=idx, task=self.task.name)

        # 1) rollout with online scene-graph tracking
        self.graph.start()
        info = self.rollout.run(self.task)
        self.graph.stop()
        g = self._snap()

        # 2) scoring
        sc = self.R.score(g)
        ep.score = {"completed": sc.completed_subtasks, "score": sc.score, "success": sc.success,
                    "rationale": sc.rationale}
        gt_g = self.gt() if self.gt else None
        if gt_g is not None:
            gt_done = self.task.completed_prefix(gt_g)
            ep.score["gt_completed"] = gt_done
            ep.score["correct"] = sorted(gt_done) == sorted(sc.completed_subtasks)
        if "gt_completed" in info and gt_g is None:
            ep.score["gt_completed_count"] = info["gt_completed"]

        # 3) planning
        plan, attempts = self.R.plan_with_check(g)
        ep.plan = {"steps": plan.steps, "valid": plan.valid, "invalid_reason": plan.invalid_reason,
                   "first_attempt_valid": attempts[0].valid, "n_attempts": len(attempts),
                   "rationale": plan.rationale}
        if not plan.valid:
            self._intervene(ep, f"no valid reset plan ({plan.invalid_reason})")
            return self._finish(ep, t0, autonomous=False)

        # 4) execute + per-skill verification
        autonomous = True
        for name in plan.steps:
            before = g
            ok = False
            for attempt in range(self.cfg.skill_retries + 1):
                self.graph.start()
                out = self.skills.run(name, attempt)
                self.graph.stop()
                g = self._snap()
                v = self.R.verify_skill(self.scene.skills[name], before, g)
                rec = {"skill": name, "attempt": attempt, "policy": out.policy_index,
                       "terminated": out.terminated, "duration_s": out.duration_s,
                       "verified": v.ok, "rationale": v.rationale}
                if self.gt:
                    gt_after = self.gt()
                    sk = self.scene.skills[name]
                    gt_ok = all(gt_after.has(*t) for t in sk.effects_add) and all(
                        gt_after.nodes.get(o) is not None and gt_after.nodes[o].attributes.get(k) == val
                        for o, d in sk.attribute_effects.items() for k, val in d.items())
                    rec["gt_success"], rec["verify_correct"] = gt_ok, gt_ok == v.ok
                ep.skill_results.append(rec)
                if v.ok:
                    ok = True
                    break
                before = g
            if not ok:
                if self.cfg.replan_on_skill_failure:
                    plan2, _ = self.R.plan_with_check(g)
                    ep.notes.append(f"replan after {name} failed: {plan2.steps}")
                self._intervene(ep, f"skill '{name}' failed verification")
                autonomous = False
                break

        # 5) final verification
        if autonomous:
            fv = self.R.verify_final(g)
            ep.final_verify = {"ok": fv.ok, "missing": fv.missing, "rationale": fv.rationale}
            if self.gt:
                gt_ok = not self.scene.target.missing(self.gt())
                ep.final_verify.update(gt_ok=gt_ok, correct=gt_ok == fv.ok)
            if not fv.ok:
                self._intervene(ep, f"final state mismatch: {fv.missing}")
                autonomous = False
        return self._finish(ep, t0, autonomous)

    def _finish(self, ep: EpisodeLog, t0: float, autonomous: bool) -> EpisodeLog:
        if self.gt:
            # Reset SR counts an autonomous reset only if the scene is truly reset
            gt_reset = ep.final_verify.get("gt_ok") if ep.final_verify else False
            ep.reset_success = bool(autonomous and gt_reset)
        else:
            ep.reset_success = autonomous
            lab = self.human.label("Was the scene correctly reset? [y/n]") if autonomous else None
            if lab is not None:
                ep.reset_success = lab.lower().startswith("y")
                if ep.final_verify is not None:
                    ep.final_verify["gt_ok"] = ep.reset_success
                    ep.final_verify["correct"] = ep.final_verify["ok"] == ep.reset_success
                if not ep.reset_success:
                    self._intervene(ep, "operator: reset was wrong although verified")
        ep.wall_time_s = time.time() - t0
        with open(self.log_path, "a") as f:
            f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
        return ep

    def run(self, n_episodes: int):
        return [self.run_episode(i) for i in range(n_episodes)]
