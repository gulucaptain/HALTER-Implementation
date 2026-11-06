"""Small interfaces that let the harness run on the real robot or the symbolic simulator."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

from halter.reasoning.task_spec import TaskSpec
from halter.types import SceneGraph


class GraphSource(ABC):
    @abstractmethod
    def snapshot(self) -> SceneGraph: ...

    def start(self) -> None:          # begin background tracking (rollout / skill execution)
        pass

    def stop(self) -> None:
        pass


class BuilderGraphSource(GraphSource):
    def __init__(self, builder):
        self.b = builder

    def snapshot(self):
        return self.b.snapshot()

    def start(self):
        self.b.start()

    def stop(self):
        self.b.stop()


class WorldGraphSource(GraphSource):
    def __init__(self, world):
        self.w = world

    def snapshot(self):
        return self.w.observe()


class RolloutRunner(ABC):
    """Runs the policy under evaluation (BESO in the paper) for one episode."""

    @abstractmethod
    def run(self, task: TaskSpec) -> Dict: ...


class LeRobotRolloutRunner(RolloutRunner):
    """Generic closed-loop rollout of any LeRobot policy checkpoint (diffusion / ACT / pi0 / X-VLA ...)."""

    def __init__(self, robot, cameras, checkpoint: str, policy_type: str = "diffusion",
                 max_steps: int = 1200, device: str = "cuda", image_keys=None, state_fn=None):
        from halter.skills.xvla_skill import default_state_fn
        self.robot, self.cameras = robot, cameras
        self.ckpt, self.type, self.max_steps, self.device = checkpoint, policy_type, max_steps, device
        self.image_keys = image_keys or {}
        self.state_fn = state_fn or default_state_fn
        self._policy = None

    def _load(self):
        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        cls = get_policy_class(self.type)
        self.torch = torch
        self._policy = cls.from_pretrained(self.ckpt).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self._policy.config, self.ckpt, preprocessor_overrides={"device_processor": {"device": self.device}})

    def run(self, task: TaskSpec) -> Dict:
        import numpy as np
        if self._policy is None:
            self._load()
        torch = self.torch
        self._policy.reset()
        self.robot.home()
        dt = 1.0 / self.robot.control_hz
        t0 = time.time()
        for step in range(self.max_steps):
            ts = time.time()
            obs = {}
            for c in self.cameras:
                if c.name in self.image_keys:
                    obs[self.image_keys[c.name]] = torch.from_numpy(c.read().rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
            obs["observation.state"] = torch.from_numpy(self.state_fn(self.robot).astype(np.float32)).unsqueeze(0)
            obs["task"] = [task.instruction]
            with torch.inference_mode():
                a = self.post(self._policy.select_action(self.pre(obs)))
            self.robot.apply_action(a.squeeze(0).cpu().numpy())
            time.sleep(max(0.0, dt - (time.time() - ts)))
        self.robot.stop()
        self.robot.home()
        return {"steps": self.max_steps, "duration_s": time.time() - t0}


class SymbolicRolloutRunner(RolloutRunner):
    def __init__(self, world, p_subtask: float = 0.7):
        self.w, self.p = world, p_subtask

    def run(self, task):
        return {"gt_completed": self.w.rollout(task, self.p)}


class HumanInterface(ABC):
    @abstractmethod
    def request_reset(self, reason: str) -> float:
        """Ask the operator to reset the scene manually. Returns operator time in seconds."""

    def label(self, question: str) -> Optional[str]:
        return None


class CLIHuman(HumanInterface):
    def request_reset(self, reason: str) -> float:
        t0 = time.time()
        input(f"\n[HALTER] Human help requested: {reason}\n  Reset the scene manually, then press Enter... ")
        return time.time() - t0

    def label(self, question: str) -> Optional[str]:
        ans = input(f"[HALTER label] {question} (leave empty to skip): ").strip()
        return ans or None


class SimHuman(HumanInterface):
    def __init__(self, world, seconds: float = 53.0):
        self.w, self.s = world, seconds   # 53 s/episode = manual reset time reported in the paper

    def request_reset(self, reason: str) -> float:
        self.w.reset_to_target()
        return self.s
