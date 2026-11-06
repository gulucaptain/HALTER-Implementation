"""Other robot backends: YAM (6-DoF, I2RT) adapter skeleton, LIBERO simulation, and a no-op robot."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from halter.robot.base import Robot


class YAMRobot(Robot):
    """I2RT YAM arm. Fill in with your SDK (e.g. `i2rt`): joint-space control is typical for YAM.
    Action convention here: [q1..q6, gripper]."""

    def __init__(self, channel: str = "can0", control_hz: float = 30.0, home_q=None):
        self.control_hz = control_hz
        self.home_q = np.zeros(6) if home_q is None else np.asarray(home_q, float)
        try:
            from i2rt.robots.get_robot import get_yam_robot  # adjust to your SDK version
            self.arm = get_yam_robot(channel=channel)
        except Exception as e:
            raise ImportError(f"YAM SDK not available or failed to connect: {e}")

    def get_state(self) -> Dict[str, np.ndarray]:
        q = np.asarray(self.arm.get_joint_pos())
        return {"joints": q[:6], "gripper": q[6:7] if len(q) > 6 else np.zeros(1)}

    def apply_action(self, action: np.ndarray) -> None:
        self.arm.command_joint_pos(np.asarray(action, float))

    def home(self) -> None:
        self.arm.command_joint_pos(np.concatenate([self.home_q, [0.0]]))


class LiberoRobot(Robot):
    """Wraps a LIBERO env so skills/policies can be tested in simulation (7-DoF Franka in robosuite).
    The env also gives privileged object poses, which `halter.sim.libero_graph` turns into a scene graph."""

    def __init__(self, env, control_hz: float = 20.0):
        self.env = env
        self.control_hz = control_hz
        self.last_obs: Optional[Dict] = None

    def get_state(self) -> Dict[str, np.ndarray]:
        o = self.last_obs or {}
        return {"ee_pos": o.get("robot0_eef_pos", np.zeros(3)),
                "ee_quat": o.get("robot0_eef_quat", np.zeros(4)),
                "gripper": o.get("robot0_gripper_qpos", np.zeros(2))}

    def apply_action(self, action: np.ndarray) -> None:
        self.last_obs, _, _, _ = self.env.step(np.asarray(action, float))

    def home(self) -> None:
        self.last_obs = self.env.reset()


class NullRobot(Robot):
    """Does nothing; used with the symbolic simulator."""

    def get_state(self):
        return {}

    def apply_action(self, action):
        pass

    def home(self):
        pass
