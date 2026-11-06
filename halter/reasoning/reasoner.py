"""LLM-based scoring, reset planning and verification over the serialised scene graph."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from halter.llm.client import LLMClient
from halter.llm.parsing import extract_json
from halter.reasoning import prompts as P
from halter.reasoning.reference_cases import ReferenceSet
from halter.reasoning.task_spec import SceneSpec, TaskSpec
from halter.scene_graph.serialize import graph_diff, graph_to_text
from halter.types import ResetPlan, SceneGraph, ScoreResult, SkillSpec, VerifyResult

log = logging.getLogger(__name__)


def _msgs(user: str, ctx: Dict) -> List[Dict]:
    # `_ctx` carries structured info for the mock backend; real backends strip keys starting with "_"
    return [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": user, "_ctx": ctx}]


def _subtasks_text(task: TaskSpec) -> str:
    lines = []
    for i, s in enumerate(task.subtasks, 1):
        cond = ", ".join(f"({a}, {b}, {c})" for a, b, c in s.goal)
        attrs = ", ".join(f"{o}.{k}={v}" for o, d in s.goal_attributes.items() for k, v in d.items())
        lines.append(f"  {i}. {s.name}: goal = {cond}{'; ' + attrs if attrs else ''}")
    return "\n".join(lines)


def _skills_text(skills: Dict[str, SkillSpec]) -> str:
    return "\n".join(s.describe() for s in skills.values())


# ---------------------------------------------------------------- symbolic helpers
def apply_skill_symbolic(triples: Set[Tuple], attrs: Dict[str, Dict], skill: SkillSpec):
    t = set(triples) - set(skill.effects_del)
    # moving an object removes its previous placement relation
    moved = {s for s, p, o in skill.effects_add if p in ("on", "in")}
    t = {x for x in t if not (x[0] in moved and x[1] in ("on", "in"))}
    t |= set(skill.effects_add)
    a = {k: dict(v) for k, v in attrs.items()}
    for obj, d in skill.attribute_effects.items():
        a.setdefault(obj, {}).update(d)
    return t, a


def preconditions_hold(triples, attrs, skill: SkillSpec) -> List[str]:
    miss = [str(p) for p in skill.preconditions if tuple(p) not in triples]
    miss += [f"NOT {p}" for p in skill.neg_preconditions if tuple(p) in triples]
    for obj, d in (skill.attr_preconditions).items():
        for k, v in d.items():
            if attrs.get(obj, {}).get(k) != v:
                miss.append(f"{obj}.{k}={v}")
    return miss


def validate_plan(plan: List[str], g: SceneGraph, scene: SceneSpec) -> Tuple[bool, str]:
    """Plan validity: every skill exists, preconditions hold in sequence, and the end state matches target."""
    triples = g.triples()
    attrs = {k: dict(n.attributes) for k, n in g.nodes.items()}
    for i, name in enumerate(plan):
        if name not in scene.skills:
            return False, f"step {i}: unknown skill '{name}'"
        sk = scene.skills[name]
        miss = preconditions_hold(triples, attrs, sk)
        if miss:
            return False, f"step {i} ({name}): preconditions not met: {miss}"
        triples, attrs = apply_skill_symbolic(triples, attrs, sk)
    end = SceneGraph(relations=[])
    from halter.types import ObjectNode, Relation
    end.relations = [Relation(*t) for t in triples]
    end.nodes = {k: ObjectNode(name=k, attributes=v) for k, v in attrs.items()}
    miss = scene.target.missing(end)
    if miss:
        return False, f"target not reached after plan, missing {miss}"
    return True, ""


# ---------------------------------------------------------------- reasoner
class Reasoner:
    def __init__(self, llm: LLMClient, scene: SceneSpec, task: TaskSpec, refs: ReferenceSet,
                 include_geometry: bool = True, max_replans: int = 1):
        self.llm, self.scene, self.task, self.refs = llm, scene, task, refs
        self.include_geometry = include_geometry
        self.max_replans = max_replans

    def _gt(self, g: SceneGraph) -> str:
        return graph_to_text(g, include_geometry=self.include_geometry)

    # (1) scoring ---------------------------------------------------------
    def score(self, g: SceneGraph) -> ScoreResult:
        user = P.SCORING.format(task_name=self.task.name, instruction=self.task.instruction,
                                subtasks=_subtasks_text(self.task),
                                references=self.refs.format_for_scoring(), graph=self._gt(g))
        ctx = {"role": "score", "graph": g, "task": self.task}
        d = extract_json(self.llm.chat(_msgs(user, ctx)))
        names = [s.name for s in self.task.subtasks]
        done = [s for s in d.get("completed_subtasks", []) if s in names]
        n = len(names)
        return ScoreResult(done, n, len(done) / n if n else 0.0,
                           bool(d.get("success", len(done) == n)), d.get("rationale", ""))

    # (2) planning --------------------------------------------------------
    def plan(self, g: SceneGraph, feedback: str = "") -> ResetPlan:
        user = P.PLANNING.format(task_name=self.task.name, target=self.scene.target.to_text(),
                                 skills=_skills_text(self.scene.skills),
                                 references=self.refs.format_for_planning(), graph=self._gt(g))
        if feedback:
            user += f"\n\nA previous plan was rejected: {feedback}. Produce a corrected plan."
        ctx = {"role": "plan", "graph": g, "scene": self.scene}
        d = extract_json(self.llm.chat(_msgs(user, ctx)))
        steps = [str(s) for s in d.get("plan", [])]
        ok, why = validate_plan(steps, g, self.scene)
        if not d.get("feasible", True):
            ok, why = False, "LLM reported infeasible"
        return ResetPlan(steps, d.get("rationale", ""), ok, why)

    def plan_with_check(self, g: SceneGraph) -> Tuple[ResetPlan, List[ResetPlan]]:
        """Plan, and re-prompt with the validator's feedback if the plan is symbolically invalid.
        Returns the final plan plus all attempts (the first attempt is what 'plan validity' measures)."""
        attempts = [self.plan(g)]
        while not attempts[-1].valid and len(attempts) <= self.max_replans:
            attempts.append(self.plan(g, feedback=attempts[-1].invalid_reason))
        return attempts[-1], attempts

    # (3a) per-skill verification ----------------------------------------
    def verify_skill(self, skill: SkillSpec, before: SceneGraph, after: SceneGraph) -> VerifyResult:
        expected = f"add {skill.effects_add}, remove {skill.effects_del}, attributes {skill.attribute_effects}"
        user = P.VERIFY_SKILL.format(skill_name=skill.name, instruction=skill.instruction, expected=expected,
                                     references=self.refs.format_for_skill(skill.name),
                                     before=self._gt(before), after=self._gt(after),
                                     diff=graph_diff(before, after))
        ctx = {"role": "verify_skill", "skill": skill, "before": before, "after": after}
        d = extract_json(self.llm.chat(_msgs(user, ctx)))
        return VerifyResult(bool(d.get("success", False)), d.get("rationale", ""))

    # (3b) final verification --------------------------------------------
    def verify_final(self, g: SceneGraph) -> VerifyResult:
        user = P.VERIFY_FINAL.format(target=self.scene.target.to_text(),
                                     references=self.refs.format_for_final(), graph=self._gt(g))
        ctx = {"role": "verify_final", "graph": g, "scene": self.scene}
        d = extract_json(self.llm.chat(_msgs(user, ctx)))
        return VerifyResult(bool(d.get("success", False)), d.get("rationale", ""), list(d.get("missing", [])))
