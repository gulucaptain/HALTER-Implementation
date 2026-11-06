"""Build a HALTER instance from YAML configs (sim or real)."""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import yaml

from halter.harness.halter import HALTER, HarnessConfig
from halter.harness.interfaces import (BuilderGraphSource, CLIHuman, LeRobotRolloutRunner, SimHuman,
                                       SymbolicRolloutRunner, WorldGraphSource)
from halter.llm.client import LLMConfig, make_llm
from halter.reasoning.reasoner import Reasoner
from halter.reasoning.reference_cases import ReferenceSet
from halter.reasoning.task_spec import load_scene, load_task
from halter.skills.base import SkillLibrary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_yaml(path: str) -> Dict:
    return yaml.safe_load(open(_p(path)))


def make_llm_from(path: str):
    return make_llm(LLMConfig.from_dict(load_yaml(path)))


def _harness_cfg(cfg: Dict, task_name: str) -> HarnessConfig:
    h = dict(cfg.get("harness", {}))
    h["log_dir"] = _p(h.get("log_dir", "runs/default"))
    return HarnessConfig(**h)


def build_sim(cfg: Dict, task_path: str, llm=None, seed: int = 0):
    from halter.sim.symbolic import SymbolicSkillExecutor, SymbolicWorld
    task = load_task(_p(task_path))
    scene = load_scene(_p(f"configs/scenes/{task.scene}.yaml"))
    s = cfg.get("sim", {})
    world = SymbolicWorld(scene, seed=seed, p_skill_success=s.get("p_skill_success", 0.85),
                          p_perception_drop=s.get("p_perception_drop", 0.0))
    llm = llm or make_llm_from(cfg["llm"])
    refs = ReferenceSet.load(task.reference_cases) if os.path.exists(task.reference_cases or "") else ReferenceSet()
    hc = _harness_cfg(cfg, task.name)
    R = Reasoner(llm, scene, task, refs, max_replans=hc.max_replans)
    skills = SkillLibrary(scene.skills, SymbolicSkillExecutor(world))
    h = HALTER(scene, task, R, skills, WorldGraphSource(world),
               SymbolicRolloutRunner(world, s.get("p_subtask", 0.6)), SimHuman(world), hc,
               ground_truth=world.ground_truth)
    return h, world


def build_real(cfg: Dict, task_path: str, llm=None):
    from halter.perception.cameras import build_cameras
    from halter.perception.segmenter import GroundedSAMSegmenter
    from halter.scene_graph.builder import SceneGraphBuilder
    task = load_task(_p(task_path))
    scene = load_scene(_p(f"configs/scenes/{task.scene}.yaml"))

    # robot
    rc = dict(cfg["robot"])
    typ = rc.pop("type")
    if typ == "franka":
        from halter.robot.franka import FrankaRobot
        robot = FrankaRobot(**rc)
    elif typ == "yam":
        from halter.robot.other import YAMRobot
        robot = YAMRobot(**rc)
    else:
        raise ValueError(typ)

    # cameras (wrist camera extrinsics = T_world_ee @ T_ee_cam)
    cam_cfgs = []
    wrist = {}
    for c in cfg["cameras"]:
        c = dict(c)
        if c.pop("wrist", False):
            wrist[c["name"]] = np.load(_p(c["extrinsics"]))
            c["extrinsics"] = None
        elif isinstance(c.get("extrinsics"), str):
            c["extrinsics"] = _p(c["extrinsics"])
        cam_cfgs.append(c)
    cams = build_cameras(cam_cfgs)
    for cam in cams:
        if cam.name in wrist:
            T_ee_cam = wrist[cam.name]
            cam.pose_fn = lambda T=T_ee_cam: robot.ee_pose_matrix() @ T

    sg = cfg.get("segmenter", {})
    seg = GroundedSAMSegmenter(sg.get("dino_id", "IDEA-Research/grounding-dino-base"),
                               sg.get("sam_id", "facebook/sam-vit-huge"), sg.get("device", "cuda"),
                               aliases=scene.segmenter_aliases)
    sgc = cfg.get("scene_graph", {})
    builder = SceneGraphBuilder(cams, seg, scene.objects, scene.relation_cfg, workspace=scene.workspace,
                                voxel=sgc.get("voxel", 0.005), min_points=sgc.get("min_points", 30),
                                rate_hz=sgc.get("rate_hz", 0.5), static_objects=scene.static_objects)

    # skills
    sk = cfg.get("skills", {})
    if sk.get("type", "xvla") == "xvla":
        from halter.skills.xvla_skill import XVLASkillExecutor
        for s in scene.skills.values():
            # local relative paths resolve against the repo root; anything else is treated as a HF repo id
            s.checkpoints = [_p(c) if os.path.exists(_p(c)) else c for c in s.checkpoints]
        executor = XVLASkillExecutor(robot, [c for c in cams if c.name in sk.get("image_keys", {})],
                                     device=sk.get("device", "cuda"), image_keys=sk.get("image_keys"),
                                     two_policy_mode=sk.get("two_policy_mode", "retry"))
    else:
        raise NotImplementedError("MP-skills baseline: construct MotionPlanningSkillExecutor with your "
                                  "GraspNet / cuRobo functions (see halter/skills/mp_skill.py)")

    ro = cfg["rollout"]
    rollout = LeRobotRolloutRunner(robot, cams, _p(ro["checkpoint"]), ro.get("policy_type", "diffusion"),
                                   ro.get("max_steps", 1200), sk.get("device", "cuda"), ro.get("image_keys"))
    llm = llm or make_llm_from(cfg["llm"])
    refs = ReferenceSet.load(task.reference_cases) if os.path.exists(task.reference_cases or "") else ReferenceSet()
    hc = _harness_cfg(cfg, task.name)
    R = Reasoner(llm, scene, task, refs, max_replans=hc.max_replans)
    return HALTER(scene, task, R, SkillLibrary(scene.skills, executor), BuilderGraphSource(builder),
                  rollout, CLIHuman(), hc)
