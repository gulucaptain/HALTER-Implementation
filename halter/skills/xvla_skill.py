"""Learned atomic reset skills with X-VLA (LeRobot implementation).

Paper: "For each atomic skill in K, we train two X-VLA policies using 30 to 50 demonstrations per skill."
The paper does not say how the two policies differ or are combined. Options here (`two_policy_mode`):
  - "retry":    policy 0 on the first attempt, policy 1 on a retry after a failed verification (default)
  - "primary":  always use policy 0
  - "alternate": attempt k uses policy k % 2
Fine-tune each policy with scripts/train_xvla_skill.sh (lerobot-train, --policy.path=lerobot/xvla-base).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from halter.perception.cameras import Camera
from halter.robot.base import Robot
from halter.skills.base import SkillExecutor, SkillOutcome
from halter.types import SkillSpec

log = logging.getLogger(__name__)


class XVLAPolicyRunner:
    """Loads one fine-tuned X-VLA checkpoint and maps robot observations to actions."""

    def __init__(self, checkpoint: str, device: str = "cuda",
                 image_keys: Optional[Dict[str, str]] = None, state_key: str = "observation.state",
                 image_size: Optional[int] = None):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        self.torch = torch
        self.device = torch.device(device)
        self.policy = XVLAPolicy.from_pretrained(checkpoint).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        # camera name -> dataset feature key, must match the keys used when recording demos
        self.image_keys = image_keys or {}
        self.state_key = state_key
        self.image_size = image_size

    def reset(self) -> None:
        self.policy.reset()

    def _obs(self, frames: Dict[str, np.ndarray], state: np.ndarray, instruction: str) -> Dict:
        torch = self.torch
        obs = {}
        for cam, key in self.image_keys.items():
            img = frames[cam]
            if self.image_size:
                from PIL import Image
                img = np.array(Image.fromarray(img).resize((self.image_size, self.image_size)))
            obs[key] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        obs[self.state_key] = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        obs["task"] = [instruction]
        return obs

    def act(self, frames: Dict[str, np.ndarray], state: np.ndarray, instruction: str) -> np.ndarray:
        batch = self.pre(self._obs(frames, state, instruction))
        with self.torch.inference_mode():
            a = self.post(self.policy.select_action(batch))
        return a.squeeze(0).float().cpu().numpy()


def default_state_fn(robot: Robot) -> np.ndarray:
    s = robot.get_state()
    parts = [s[k] for k in ("ee_pos", "ee_rot6d", "gripper") if k in s] or [s.get("joints", np.zeros(7))]
    return np.concatenate(parts)


class XVLASkillExecutor(SkillExecutor):
    def __init__(self, robot: Robot, cameras: List[Camera], device: str = "cuda",
                 image_keys: Optional[Dict[str, str]] = None, two_policy_mode: str = "retry",
                 state_fn: Callable[[Robot], np.ndarray] = default_state_fn,
                 done_fn: Optional[Callable[[SkillSpec, int], bool]] = None, cache_policies: int = 4):
        self.robot, self.cameras = robot, cameras
        self.device, self.image_keys = device, image_keys
        self.mode = two_policy_mode
        self.state_fn = state_fn
        self.done_fn = done_fn       # optional early termination (e.g. gripper open + arm retreated)
        self._cache: Dict[str, XVLAPolicyRunner] = {}
        self.cache_policies = cache_policies

    def _policy_index(self, attempt: int, n: int) -> int:
        if n <= 1 or self.mode == "primary":
            return 0
        if self.mode == "alternate":
            return attempt % n
        return min(attempt, n - 1)   # retry

    def _load(self, ckpt: str) -> XVLAPolicyRunner:
        if ckpt not in self._cache:
            if len(self._cache) >= self.cache_policies:   # simple FIFO eviction to bound GPU memory
                self._cache.pop(next(iter(self._cache)))
            self._cache[ckpt] = XVLAPolicyRunner(ckpt, self.device, self.image_keys)
        return self._cache[ckpt]

    def execute(self, spec: SkillSpec, attempt: int = 0) -> SkillOutcome:
        if not spec.checkpoints:
            raise ValueError(f"skill {spec.name} has no checkpoints configured")
        idx = self._policy_index(attempt, len(spec.checkpoints))
        runner = self._load(spec.checkpoints[idx])
        runner.reset()
        dt = 1.0 / self.robot.control_hz
        t0 = time.time()
        term = "max_steps"
        step = 0
        for step in range(spec.max_steps):
            ts = time.time()
            frames = {c.name: c.read().rgb for c in self.cameras}
            action = runner.act(frames, self.state_fn(self.robot), spec.instruction)
            self.robot.apply_action(action)
            if self.done_fn is not None and self.done_fn(spec, step):
                term = "policy_done"
                break
            time.sleep(max(0.0, dt - (time.time() - ts)))
        self.robot.stop()
        return SkillOutcome(spec.name, idx, step + 1, time.time() - t0, term)
