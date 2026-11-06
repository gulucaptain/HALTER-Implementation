"""Task, scene and skill-library specs loaded from YAML (configs/scenes, configs/tasks)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import yaml

from halter.scene_graph.relations import RelationConfig
from halter.types import SceneGraph, SkillSpec

Triple = Tuple[str, str, str]


def _tok(x) -> str:
    # YAML 1.1 parses a bare `on` as boolean True: map it back
    if x is True:
        return "on"
    if x is False:
        return "off"
    return str(x)


def _triples(xs) -> List[Triple]:
    return [tuple(_tok(v) for v in x) for x in (xs or [])]


@dataclass
class Subtask:
    name: str
    goal: List[Triple]
    goal_attributes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    implies: List[str] = field(default_factory=list)   # earlier subtasks necessarily done if this one is

    def satisfied(self, g: SceneGraph) -> bool:
        if not all(g.has(*t) for t in self.goal):
            return False
        for obj, attrs in self.goal_attributes.items():
            n = g.nodes.get(obj)
            if n is None or any(n.attributes.get(k) != v for k, v in attrs.items()):
                return False
        return True


@dataclass
class TargetState:
    relations: List[Triple]
    attributes: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def missing(self, g: SceneGraph) -> List[str]:
        out = [f"({s}, {p}, {o})" for s, p, o in self.relations if not g.has(s, p, o)]
        for obj, attrs in self.attributes.items():
            n = g.nodes.get(obj)
            for k, v in attrs.items():
                if n is None or n.attributes.get(k) != v:
                    out.append(f"{obj}.{k}={v}")
        return out

    def to_text(self) -> str:
        lines = [f"  ({s}, {p}, {o})" for s, p, o in self.relations]
        for obj, attrs in self.attributes.items():
            for k, v in attrs.items():
                lines.append(f"  {obj}.{k} = {v}")
        return "\n".join(lines)


@dataclass
class SceneSpec:
    name: str
    objects: List[str]
    static_objects: Dict[str, Dict[str, List[float]]]
    relation_cfg: RelationConfig
    skills: Dict[str, SkillSpec]
    target: TargetState
    segmenter_aliases: Dict[str, str] = field(default_factory=dict)
    workspace: List[List[float]] = None


@dataclass
class TaskSpec:
    name: str
    scene: str
    instruction: str
    subtasks: List[Subtask]
    held_out: bool = False
    reference_cases: str = ""

    def completed_prefix(self, g: SceneGraph) -> List[str]:
        """Ground-truth scoring rule: largest k such that subtask k holds and every earlier subtask
        either holds or is implied by subtask k. Returns the names of the first k subtasks."""
        for k in range(len(self.subtasks), 0, -1):
            last = self.subtasks[k - 1]
            if not last.satisfied(g):
                continue
            if all(s.satisfied(g) or s.name in last.implies for s in self.subtasks[:k - 1]):
                return [s.name for s in self.subtasks[:k]]
        return []


def load_scene(path: str) -> SceneSpec:
    d = yaml.safe_load(open(path))
    skills = {}
    for s in d["skills"]:
        skills[s["name"]] = SkillSpec(
            name=s["name"], instruction=s["instruction"],
            preconditions=_triples(s.get("preconditions")),
            attr_preconditions=s.get("attr_preconditions", {}) or {},
            neg_preconditions=_triples(s.get("neg_preconditions")),
            effects_add=_triples(s.get("effects_add")), effects_del=_triples(s.get("effects_del")),
            attribute_effects=s.get("attribute_effects", {}) or {},
            checkpoints=s.get("checkpoints", []) or [], max_steps=s.get("max_steps", 600))
    t = d["target"]
    return SceneSpec(
        name=d["name"], objects=d["objects"], static_objects=d.get("static_objects", {}) or {},
        relation_cfg=RelationConfig.from_dict(d.get("relations")), skills=skills,
        target=TargetState(_triples(t.get("relations")), t.get("attributes", {}) or {}),
        segmenter_aliases=d.get("segmenter_aliases", {}) or {}, workspace=d.get("workspace"))


def load_task(path: str) -> TaskSpec:
    d = yaml.safe_load(open(path))
    subs = [Subtask(s["name"], _triples(s.get("goal")), s.get("goal_attributes", {}) or {},
                    s.get("implies", []) or []) for s in d["subtasks"]]
    ref = d.get("reference_cases", "")
    if ref and not os.path.isabs(ref):
        ref = os.path.normpath(os.path.join(os.path.dirname(path), ref))
    return TaskSpec(d["name"], d["scene"], d["instruction"], subs, d.get("held_out", False), ref)
