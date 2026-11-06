"""Mask + depth -> per-object world-frame point clouds, with multi-view fusion and cleanup."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from halter.types import RGBDFrame


def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray,
                min_depth: float = 0.05, max_depth: float = 3.0) -> np.ndarray:
    """Return (N, 3) world points for pixels inside `mask` with valid depth."""
    v, u = np.nonzero(mask)
    z = depth[v, u]
    ok = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    u, v, z = u[ok], v[ok], z[ok]
    if z.size == 0:
        return np.zeros((0, 3))
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (T_world_cam @ pc.T).T[:, :3]


def erode(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    """Erode mask borders (depth at mask edges is unreliable). Pure numpy 4-neighbourhood erosion."""
    m = mask.copy()
    for _ in range(iters):
        n = m.copy()
        n[1:, :] &= m[:-1, :]
        n[:-1, :] &= m[1:, :]
        n[:, 1:] &= m[:, :-1]
        n[:, :-1] &= m[:, 1:]
        m = n
    return m


def voxel_downsample(points: np.ndarray, voxel: float = 0.005) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def remove_outliers(points: np.ndarray, k_sigma: float = 2.5) -> np.ndarray:
    """Simple statistical outlier removal around the median (robust, no KD-tree needed)."""
    if len(points) < 20:
        return points
    med = np.median(points, axis=0)
    d = np.linalg.norm(points - med, axis=1)
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    keep = d < np.median(d) + k_sigma * 1.4826 * mad
    return points[keep]


def crop_workspace(points: np.ndarray, lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo), np.asarray(hi)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def fuse_object_points(frames: List[RGBDFrame], masks_per_frame: List[Dict[str, np.ndarray]],
                       voxel: float = 0.005, workspace=None, erode_iters: int = 2,
                       min_points: int = 30) -> Dict[str, np.ndarray]:
    """Fuse per-object points across views. Returns {label: (N,3)} only for labels with enough points."""
    acc: Dict[str, List[np.ndarray]] = {}
    for f, masks in zip(frames, masks_per_frame):
        for label, m in masks.items():
            pts = backproject(f.depth, erode(m, erode_iters), f.K, f.T_world_cam)
            if workspace is not None and len(pts):
                pts = crop_workspace(pts, workspace[0], workspace[1])
            if len(pts):
                acc.setdefault(label, []).append(pts)
    out = {}
    for label, chunks in acc.items():
        pts = remove_outliers(voxel_downsample(np.concatenate(chunks), voxel))
        if len(pts) >= min_points:
            out[label] = pts
    return out
