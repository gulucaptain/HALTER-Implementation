"""Reference cases (in-context examples).

Paper: per task, 4 reference cases (1 success + 3 failures) for reset planning and scoring;
per skill, 4 success reference cases for verification. Reset cases can be reused across tasks
that invoke the same atomic skill.

File format (YAML):
task_cases:
  - id: success
    graph: {objects: [...], relations: [[s,p,o], ...]}
    completed_subtasks: [...]
    plan: [...]
    note: "..."
skill_cases:
  <skill_name>:
    - before: {...graph...}
      after: {...graph...}
      note: "..."
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

import yaml

from halter.scene_graph.serialize import graph_diff, graph_to_text
from halter.types import SceneGraph


@dataclass
class ReferenceSet:
    task_cases: List[Dict] = field(default_factory=list)
    skill_cases: Dict[str, List[Dict]] = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> "ReferenceSet":
        if not path:
            return ReferenceSet()
        d = yaml.safe_load(open(path)) or {}
        return ReferenceSet(d.get("task_cases", []) or [], d.get("skill_cases", {}) or {})

    def merge(self, other: "ReferenceSet") -> "ReferenceSet":
        sc = dict(self.skill_cases)
        for k, v in other.skill_cases.items():
            sc.setdefault(k, []).extend(v)
        return ReferenceSet(self.task_cases + other.task_cases, sc)

    # ---------- formatting ----------
    @staticmethod
    def _g(d: Dict) -> str:
        return graph_to_text(SceneGraph.from_dict(d), include_geometry=False)

    def format_for_scoring(self) -> str:
        out = []
        for i, c in enumerate(self.task_cases, 1):
            ans = {"completed_subtasks": c.get("completed_subtasks", []),
                   "success": c.get("success", False), "rationale": c.get("note", "")}
            out.append(f"### Case {i} ({c.get('id', '')})\n{self._g(c['graph'])}\nAnswer: {json.dumps(ans)}")
        return "\n\n".join(out) or "(none)"

    def format_for_planning(self) -> str:
        out = []
        for i, c in enumerate(self.task_cases, 1):
            ans = {"plan": c.get("plan", []), "feasible": True, "rationale": c.get("note", "")}
            out.append(f"### Case {i} ({c.get('id', '')})\n{self._g(c['graph'])}\nAnswer: {json.dumps(ans)}")
        return "\n\n".join(out) or "(none)"

    def format_for_skill(self, skill: str) -> str:
        out = []
        for i, c in enumerate(self.skill_cases.get(skill, []), 1):
            b, a = SceneGraph.from_dict(c["before"]), SceneGraph.from_dict(c["after"])
            out.append(f"### Case {i}\nChange: {graph_diff(b, a)}\nAnswer: "
                       f"{json.dumps({'success': c.get('success', True), 'rationale': c.get('note', '')})}")
        return "\n\n".join(out) or "(none)"

    def format_for_final(self) -> str:
        out = []
        for i, c in enumerate(self.task_cases, 1):
            if c.get("reset_state"):
                ok = c.get("reset_ok", True)
                out.append(f"### Case {i}\n{self._g(c['reset_state'])}\nAnswer: "
                           f"{json.dumps({'success': ok, 'missing': c.get('reset_missing', [])})}")
        return "\n\n".join(out) or "(none)"
