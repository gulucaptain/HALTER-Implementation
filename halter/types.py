"""Core data structures shared across HALTER modules.

A scene graph G = (V, E):
  V: object nodes (name, 3D point cloud summary, visibility state)
  E: spatial relation triples (subject, predicate, object), predicate in {on, in, under}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

RELATIONS = ("on", "in", "under")


@dataclass
class RGBDFrame:
    """One RGB-D observation from a calibrated camera."""
    camera: str
    rgb: np.ndarray                     # (H, W, 3) uint8
    depth: np.ndarray                   # (H, W) float32, metres
    K: np.ndarray                       # (3, 3) intrinsics
    T_world_cam: np.ndarray             # (4, 4) extrinsics (camera -> world)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ObjectNode:
    name: str
    points: Optional[np.ndarray] = None  # (N, 3) world-frame points (may be dropped when serialising)
    centroid: Optional[np.ndarray] = None
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
    visible: bool = True
    last_seen: float = 0.0
    confidence: float = 1.0
    attributes: Dict[str, str] = field(default_factory=dict)   # e.g. {"state": "open"} for articulated objects

    def update_geometry(self, points: np.ndarray) -> None:
        self.points = points
        self.centroid = points.mean(axis=0)
        self.bbox_min = points.min(axis=0)
        self.bbox_max = points.max(axis=0)

    def to_record(self) -> Dict:
        def r(x):
            return None if x is None else [round(float(v), 3) for v in x]
        rec = {
            "name": self.name,
            "centroid": r(self.centroid),
            "bbox_min": r(self.bbox_min),
            "bbox_max": r(self.bbox_max),
            "visible": self.visible,
        }
        if self.attributes:
            rec["attributes"] = dict(self.attributes)
        return rec


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


@dataclass
class SceneGraph:
    nodes: Dict[str, ObjectNode] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def triples(self) -> set:
        return {r.as_tuple() for r in self.relations}

    def has(self, subject: str, predicate: str, obj: str) -> bool:
        return (subject, predicate, obj) in self.triples()

    def copy_light(self) -> "SceneGraph":
        """Copy without point clouds (for logging / LLM input)."""
        nodes = {}
        for k, n in self.nodes.items():
            nodes[k] = ObjectNode(
                name=n.name, points=None, centroid=n.centroid, bbox_min=n.bbox_min,
                bbox_max=n.bbox_max, visible=n.visible, last_seen=n.last_seen,
                confidence=n.confidence, attributes=dict(n.attributes),
            )
        return SceneGraph(nodes=nodes, relations=list(self.relations), timestamp=self.timestamp)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "objects": [n.to_record() for n in self.nodes.values()],
            "relations": [list(r.as_tuple()) for r in self.relations],
        }

    @staticmethod
    def from_dict(d: Dict) -> "SceneGraph":
        g = SceneGraph(timestamp=d.get("timestamp", time.time()))
        for o in d.get("objects", []):
            n = ObjectNode(name=o["name"], visible=o.get("visible", True),
                           attributes=dict(o.get("attributes", {})))
            for key in ("centroid", "bbox_min", "bbox_max"):
                if o.get(key) is not None:
                    setattr(n, key, np.asarray(o[key], dtype=float))
            g.nodes[n.name] = n
        tok = lambda v: "on" if v is True else ("off" if v is False else str(v))  # YAML `on` -> True
        g.relations = [Relation(*(tok(v) for v in t)) for t in d.get("relations", [])]
        return g


@dataclass
class SkillSpec:
    """An atomic reset skill in the library K."""
    name: str
    instruction: str                         # language instruction passed to the VLA
    preconditions: List[Tuple[str, str, str]] = field(default_factory=list)
    attr_preconditions: Dict[str, Dict[str, str]] = field(default_factory=dict)  # {"lower_drawer": {"state": "open"}}
    neg_preconditions: List[Tuple[str, str, str]] = field(default_factory=list)  # must NOT hold (e.g. lid on pot)
    effects_add: List[Tuple[str, str, str]] = field(default_factory=list)
    effects_del: List[Tuple[str, str, str]] = field(default_factory=list)
    attribute_effects: Dict[str, Dict[str, str]] = field(default_factory=dict)  # {"drawer": {"state": "closed"}}
    checkpoints: List[str] = field(default_factory=list)   # paths/ids of the (two) X-VLA policies
    max_steps: int = 600

    def describe(self) -> str:
        parts = [f"- {self.name}: \"{self.instruction}\""]
        if self.preconditions or self.attr_preconditions:
            parts.append(f"  preconditions: {self.preconditions} attrs={self.attr_preconditions}")
        if self.neg_preconditions:
            parts.append(f"  must NOT hold: {self.neg_preconditions}")
        if self.effects_add or self.attribute_effects:
            parts.append(f"  expected effects: add={self.effects_add} del={self.effects_del} attrs={self.attribute_effects}")
        return "\n".join(parts)


@dataclass
class ScoreResult:
    completed_subtasks: List[str]
    total_subtasks: int
    score: float                         # completed-skill fraction in [0, 1]
    success: bool
    rationale: str = ""


@dataclass
class ResetPlan:
    steps: List[str]                     # ordered list of skill names
    rationale: str = ""
    valid: bool = True
    invalid_reason: str = ""


@dataclass
class VerifyResult:
    ok: bool
    rationale: str = ""
    missing: List[str] = field(default_factory=list)


@dataclass
class EpisodeLog:
    episode: int
    task: str
    score: Optional[Dict] = None
    plan: Optional[Dict] = None
    skill_results: List[Dict] = field(default_factory=list)
    final_verify: Optional[Dict] = None
    reset_success: bool = False
    human_interventions: int = 0
    human_time_s: float = 0.0
    wall_time_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)
