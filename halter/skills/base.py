"""Atomic reset skills and the skill library K."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from halter.types import SkillSpec

log = logging.getLogger(__name__)


@dataclass
class SkillOutcome:
    skill: str
    policy_index: int
    steps: int
    duration_s: float
    terminated: str = "max_steps"   # max_steps | policy_done | error
    error: str = ""


class SkillExecutor(ABC):
    """Executes one atomic skill on the robot (learned VLA, motion planning, or simulated)."""

    @abstractmethod
    def execute(self, spec: SkillSpec, attempt: int = 0) -> SkillOutcome: ...


class SkillLibrary:
    def __init__(self, specs: Dict[str, SkillSpec], executor: SkillExecutor):
        self.specs = specs
        self.executor = executor

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def run(self, name: str, attempt: int = 0) -> SkillOutcome:
        spec = self.specs[name]
        t0 = time.time()
        try:
            out = self.executor.execute(spec, attempt)
        except Exception as e:
            log.exception("skill %s failed", name)
            out = SkillOutcome(name, attempt, 0, time.time() - t0, "error", str(e))
        return out
