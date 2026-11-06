"""Robot interfaces. The paper uses a Franka Panda + Robotiq 2F-85; the same harness can drive other arms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Robot(ABC):
    control_hz: float = 15.0

    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]:
        """{"ee_pos": (3,), "ee_rot6d": (6,), "gripper": (1,), "joints": (n,)}"""

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Execute one action from the policy (convention set by the policy wrapper, e.g. abs EEF pose + gripper)."""

    @abstractmethod
    def home(self) -> None: ...

    def stop(self) -> None:
        pass

    def ee_pose_matrix(self) -> np.ndarray:
        """4x4 world pose of the end effector (used for the wrist camera extrinsics)."""
        raise NotImplementedError


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]])
