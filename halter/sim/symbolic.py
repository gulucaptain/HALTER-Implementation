"""Symbolic tabletop simulator for testing the harness end-to-end without hardware or models.

World state = set of relation triples + object attributes. It provides
  * a rollout policy stand-in (completes a random prefix of the task's subtasks, may drop objects)
  * a skill executor (applies the skill's effects with probability p_success, otherwise a failure mode)
  * a noisy "perception" that returns a SceneGraph (relation dropout / flips)
  * ground truth for computing scoring / verification accuracy
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from halter.reasoning.reasoner import apply_skill_symbolic
from halter.reasoning.task_spec import SceneSpec, TaskSpec
from halter.skills.base import SkillExecutor, SkillOutcome
from halter.types import ObjectNode, Relation, SceneGraph, SkillSpec


class SymbolicWorld:
    def __init__(self, scene: SceneSpec, seed: int = 0, p_skill_success: float = 0.85,
                 p_perception_drop: float = 0.0, p_rollout_drop: float = 0.3):
        self.scene = scene
        self.rng = random.Random(seed)
        self.p_skill = p_skill_success
        self.p_drop = p_perception_drop
        self.p_rollout_drop = p_rollout_drop
        self.triples: Set[Tuple] = set()
        self.attrs: Dict[str, Dict[str, str]] = {}
        self.reset_to_target()

    # ---------- state ----------
    def reset_to_target(self) -> None:
        self.triples = set(self.scene.target.relations)
        self.attrs = {k: dict(v) for k, v in self.scene.target.attributes.items()}

    def _place(self, subj: str, pred: str, obj: str) -> None:
        self.triples = {t for t in self.triples if not (t[0] == subj and t[1] in ("on", "in"))}
        self.triples.add((subj, pred, obj))

    def ground_truth(self) -> SceneGraph:
        nodes = {o: ObjectNode(name=o, attributes=dict(self.attrs.get(o, {}))) for o in self.scene.objects}
        return SceneGraph(nodes=nodes, relations=[Relation(*t) for t in sorted(self.triples)])

    def observe(self) -> SceneGraph:
        g = self.ground_truth()
        if self.p_drop > 0:
            g.relations = [r for r in g.relations if self.rng.random() > self.p_drop]
        g.timestamp = time.time()
        return g

    # ---------- rollout of the policy under evaluation ----------
    def rollout(self, task: TaskSpec, p_subtask: float = 0.7) -> int:
        """Execute subtasks in order; each succeeds w.p. p_subtask; on failure the object may end up on the table."""
        done = 0
        for st in task.subtasks:
            if self.rng.random() < p_subtask:
                for s, p, o in st.goal:
                    self._place(s, p, o)
                for obj, d in st.goal_attributes.items():
                    self.attrs.setdefault(obj, {}).update(d)
                done += 1
            else:
                moved = [s for s, p, o in st.goal if p in ("on", "in")]
                if moved and self.rng.random() < self.p_rollout_drop:
                    self._place(moved[0], "on", self.scene.relation_cfg.table_name)
                break
        return done


class SymbolicSkillExecutor(SkillExecutor):
    def __init__(self, world: SymbolicWorld):
        self.world = world

    def execute(self, spec: SkillSpec, attempt: int = 0) -> SkillOutcome:
        w = self.world
        from halter.reasoning.reasoner import preconditions_hold
        pre_ok = not preconditions_hold(w.triples, w.attrs, spec)
        if pre_ok and w.rng.random() < w.p_skill:
            w.triples, w.attrs = apply_skill_symbolic(w.triples, w.attrs, spec)
            term = "policy_done"
        else:
            # failure mode: a moved object ends up on the table instead of the target
            moved = [s for s, p, o in spec.effects_add if p in ("on", "in")]
            if moved and pre_ok and w.rng.random() < 0.5:
                w._place(moved[0], "on", w.scene.relation_cfg.table_name)
            term = "max_steps"
        return SkillOutcome(spec.name, attempt, 1, 0.0, term)
