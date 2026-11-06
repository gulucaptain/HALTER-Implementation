"""Online scene-graph builder.

Paper: graph built online at 0.5 Hz from RGB-D keyframes; GroundedSAM segments task-relevant objects;
masks are projected into depth to get per-object point clouds; relations come from geometry;
occluded objects keep their last valid observation.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from halter.perception.cameras import Camera
from halter.perception.pointcloud import fuse_object_points
from halter.perception.segmenter import Segmenter
from halter.scene_graph.relations import RelationConfig, infer_relations
from halter.types import ObjectNode, RGBDFrame, SceneGraph

log = logging.getLogger(__name__)


class SceneGraphBuilder:
    def __init__(self, cameras: List[Camera], segmenter: Segmenter, objects: List[str],
                 relation_cfg: RelationConfig, workspace=None, voxel: float = 0.005,
                 min_points: int = 30, rate_hz: float = 0.5,
                 static_objects: Optional[Dict[str, Dict[str, List[float]]]] = None):
        self.cameras = cameras
        self.segmenter = segmenter
        self.objects = list(objects)
        self.rel_cfg = relation_cfg
        self.workspace = workspace
        self.voxel = voxel
        self.min_points = min_points
        self.period = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self.memory: Dict[str, ObjectNode] = {}       # last valid observation per object
        # fixed regions that are not segmented (e.g. "pot_home" mat area): {"name": {"min": [...], "max": [...]}}
        self.static_nodes: Dict[str, ObjectNode] = {}
        for name, box in (static_objects or {}).items():
            n = ObjectNode(name=name)
            n.bbox_min, n.bbox_max = np.asarray(box["min"], float), np.asarray(box["max"], float)
            n.centroid = (n.bbox_min + n.bbox_max) / 2
            self.static_nodes[name] = n
        self.history: List[SceneGraph] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---------- single update ----------
    def capture(self) -> List[RGBDFrame]:
        return [c.read() for c in self.cameras]

    def update(self, frames: Optional[List[RGBDFrame]] = None) -> SceneGraph:
        frames = frames if frames is not None else self.capture()
        masks_per_frame = []
        for f in frames:
            dets = self.segmenter.segment(f.rgb, self.objects)
            masks_per_frame.append({d.label: d.mask for d in dets})
        pts = fuse_object_points(frames, masks_per_frame, voxel=self.voxel,
                                 workspace=self.workspace, min_points=self.min_points)
        now = time.time()
        with self._lock:
            for name in self.objects:
                if name in pts:
                    node = self.memory.get(name) or ObjectNode(name=name)
                    node.update_geometry(pts[name])
                    node.visible, node.last_seen = True, now
                    self.memory[name] = node
                elif name in self.memory:
                    # occluded / not detected: keep last valid observation
                    self.memory[name].visible = False
            g = self._compose(now)
            self.history.append(g.copy_light())
        return g

    def _compose(self, ts: float) -> SceneGraph:
        nodes = {k: ObjectNode(name=v.name, points=v.points, centroid=v.centroid, bbox_min=v.bbox_min,
                               bbox_max=v.bbox_max, visible=v.visible, last_seen=v.last_seen,
                               attributes=dict(v.attributes))
                 for k, v in self.memory.items()}
        nodes.update({k: v for k, v in self.static_nodes.items() if k not in nodes})
        rels = infer_relations(nodes, self.rel_cfg)
        for k, n in nodes.items():                 # persist articulated states into memory
            if k in self.memory:
                self.memory[k].attributes.update(n.attributes)
        return SceneGraph(nodes=nodes, relations=rels, timestamp=ts)

    def latest(self) -> Optional[SceneGraph]:
        with self._lock:
            return self._compose(time.time()) if self.memory else None

    def reset_memory(self) -> None:
        with self._lock:
            self.memory.clear()
            self.history.clear()

    # ---------- background loop during rollouts / skill execution ----------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    self.update()
                except StopIteration:
                    break
                except Exception as e:  # keep running on transient camera/segmenter errors
                    log.warning("scene graph update failed: %s", e)
                self._stop.wait(max(0.0, self.period - (time.time() - t0)))

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None

    def snapshot(self) -> SceneGraph:
        """Fresh update (synchronous). Used after a rollout / skill finishes."""
        return self.update()
