"""'MP skills' baseline: HALTER with learned skills replaced by analytic grasps (GraspNet) + motion planning (cuRobo).

Only the structure is provided; plug in your GraspNet (e.g. graspnet-baseline / AnyGrasp) and cuRobo setup.
Each pick-and-place skill = grasp the `subject` point cloud, lift, move above the target region, place, release.
Articulated skills (open/close drawer) need a handle grasp + a straight-line pull/push along the joint axis.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from halter.skills.base import SkillExecutor, SkillOutcome
from halter.types import SceneGraph, SkillSpec


class MotionPlanningSkillExecutor(SkillExecutor):
    def __init__(self, robot, graph_fn: Callable[[], SceneGraph],
                 grasp_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 plan_fn: Callable[[np.ndarray], list],
                 place_targets: Optional[Dict[str, np.ndarray]] = None,
                 articulation_axes: Optional[Dict[str, np.ndarray]] = None):
        """
        grasp_fn(object_points, scene_points) -> 4x4 grasp pose (GraspNet / AnyGrasp)
        plan_fn(target_pose_4x4) -> list of joint waypoints (cuRobo MotionGen.plan_single)
        place_targets: {target_object_or_region: xyz place position}
        """
        self.robot, self.graph_fn = robot, graph_fn
        self.grasp_fn, self.plan_fn = grasp_fn, plan_fn
        self.place_targets = place_targets or {}
        self.axes = articulation_axes or {}

    def _move(self, T: np.ndarray):
        for q in self.plan_fn(T):
            self.robot.apply_action(np.asarray(q))

    def execute(self, spec: SkillSpec, attempt: int = 0) -> SkillOutcome:
        t0 = time.time()
        g = self.graph_fn()
        if not spec.effects_add and spec.attribute_effects:
            obj = next(iter(spec.attribute_effects))
            raise NotImplementedError(f"articulated MP skill for {obj}: implement handle grasp + linear motion")
        subj, _, target = spec.effects_add[0]
        node = g.nodes[subj]
        scene_pts = np.concatenate([n.points for n in g.nodes.values() if n.points is not None])
        T_grasp = self.grasp_fn(node.points, scene_pts)
        T_pre = T_grasp.copy()
        T_pre[2, 3] += 0.10
        self._move(T_pre)
        self._move(T_grasp)
        self.robot.apply_action(np.array([1.0]))     # close gripper (adapt to your action convention)
        self._move(T_pre)
        place = self.place_targets.get(target)
        if place is None and target in g.nodes and g.nodes[target].centroid is not None:
            place = g.nodes[target].centroid + np.array([0, 0, 0.08])
        T_place = T_grasp.copy()
        T_place[:3, 3] = place
        self._move(T_place)
        self.robot.apply_action(np.array([0.0]))     # open gripper
        return SkillOutcome(spec.name, 0, 0, time.time() - t0, "policy_done")
