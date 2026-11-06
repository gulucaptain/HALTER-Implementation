"""Franka Panda + Robotiq 2F-85 via `franky` (libfranka wrapper) and `pyrobotiqgripper`.

Install: pip install franky-control pyRobotiqGripper
Action convention (matches X-VLA ee6d mode): [x, y, z, rot6d(6), gripper(1)] absolute EEF pose in base frame,
gripper in [0, 1] (1 = closed). Adjust `apply_action` if your policy outputs deltas.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from halter.robot.base import Robot, matrix_to_rot6d, rot6d_to_matrix


class FrankaRobot(Robot):
    def __init__(self, host: str, gripper_port: str = "/dev/ttyUSB0", control_hz: float = 15.0,
                 dynamics_factor: float = 0.15, home_q=None):
        import franky
        self.franky = franky
        self.robot = franky.Robot(host)
        self.robot.relative_dynamics_factor = dynamics_factor
        self.control_hz = control_hz
        self.home_q = home_q or [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        try:
            from pyrobotiqgripper import RobotiqGripper
            self.gripper = RobotiqGripper(gripper_port)
            self.gripper.activate()
        except Exception:
            self.gripper = None
        self._g = 0.0

    def get_state(self) -> Dict[str, np.ndarray]:
        s = self.robot.current_cartesian_state.pose.end_effector_pose
        T = np.array(s.matrix)
        q = np.array(self.robot.current_joint_state.position)
        return {"ee_pos": T[:3, 3], "ee_rot6d": matrix_to_rot6d(T[:3, :3]),
                "gripper": np.array([self._g]), "joints": q}

    def ee_pose_matrix(self) -> np.ndarray:
        return np.array(self.robot.current_cartesian_state.pose.end_effector_pose.matrix)

    def apply_action(self, action: np.ndarray) -> None:
        f = self.franky
        pos, R = action[:3], rot6d_to_matrix(action[3:9])
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(R).as_quat()  # xyzw
        self.robot.move(f.CartesianMotion(f.Affine(pos.tolist(), quat.tolist())), asynchronous=True)
        g = float(np.clip(action[9], 0, 1))
        if self.gripper is not None and abs(g - self._g) > 0.3:
            self.gripper.goTo(int(g * 255))
        self._g = g

    def home(self) -> None:
        self.robot.move(self.franky.JointMotion(self.home_q))
        if self.gripper is not None:
            self.gripper.open()
        self._g = 0.0

    def stop(self) -> None:
        try:
            self.robot.join_motion(timeout=1.0)
        except Exception:
            pass
