"""RGB-D camera backends.

The paper uses 3 RGB-D cameras (2x OAK-D Lite: table side + wrist, 1x ZED: opposite side).
All backends return `RGBDFrame` in metres, with calibrated intrinsics K and extrinsics T_world_cam.
Hardware SDKs are imported lazily so the rest of the package runs without them.
"""
from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np

from halter.types import RGBDFrame


class Camera(ABC):
    def __init__(self, name: str, K: Optional[np.ndarray] = None, T_world_cam: Optional[np.ndarray] = None):
        self.name = name
        self.K = K
        self.T_world_cam = np.eye(4) if T_world_cam is None else np.asarray(T_world_cam, dtype=float)
        # For a wrist camera, extrinsics change with the arm. Provide a callable returning T_world_cam.
        self.pose_fn: Optional[Callable[[], np.ndarray]] = None

    def current_pose(self) -> np.ndarray:
        return self.pose_fn() if self.pose_fn is not None else self.T_world_cam

    @abstractmethod
    def read(self) -> RGBDFrame: ...

    def close(self) -> None:
        pass


class OakDCamera(Camera):
    """Luxonis OAK-D Lite via depthai (stereo depth aligned to RGB)."""

    def __init__(self, name: str, T_world_cam=None, mxid: Optional[str] = None, fps: int = 15,
                 size=(640, 400)):
        super().__init__(name, None, T_world_cam)
        import depthai as dai  # lazy
        self.dai = dai
        p = dai.Pipeline()
        cam = p.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setPreviewSize(*size)
        cam.setInterleaved(False)
        cam.setFps(fps)
        left, right = p.create(dai.node.MonoCamera), p.create(dai.node.MonoCamera)
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        stereo = p.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(*size)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        xo_rgb, xo_d = p.create(dai.node.XLinkOut), p.create(dai.node.XLinkOut)
        xo_rgb.setStreamName("rgb")
        xo_d.setStreamName("depth")
        cam.preview.link(xo_rgb.input)
        stereo.depth.link(xo_d.input)
        info = dai.DeviceInfo(mxid) if mxid else None
        self.dev = dai.Device(p, info) if info else dai.Device(p)
        calib = self.dev.readCalibration()
        self.K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, *size))
        self.q_rgb = self.dev.getOutputQueue("rgb", 2, False)
        self.q_d = self.dev.getOutputQueue("depth", 2, False)

    def read(self) -> RGBDFrame:
        rgb = self.q_rgb.get().getCvFrame()[..., ::-1].copy()  # BGR -> RGB
        depth = self.q_d.get().getFrame().astype(np.float32) / 1000.0
        return RGBDFrame(self.name, rgb, depth, self.K, self.current_pose())

    def close(self):
        self.dev.close()


class ZedCamera(Camera):
    """Stereolabs ZED via pyzed."""

    def __init__(self, name: str, T_world_cam=None, serial: Optional[int] = None):
        super().__init__(name, None, T_world_cam)
        import pyzed.sl as sl  # lazy
        self.sl = sl
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        if serial is not None:
            init.set_from_serial_number(serial)
        self.zed = sl.Camera()
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED")
        c = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]])
        self._img, self._dep = sl.Mat(), sl.Mat()

    def read(self) -> RGBDFrame:
        sl = self.sl
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab failed")
        self.zed.retrieve_image(self._img, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self._dep, sl.MEASURE.DEPTH)
        rgb = self._img.get_data()[..., :3][..., ::-1].copy()
        depth = np.nan_to_num(self._dep.get_data().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return RGBDFrame(self.name, rgb, depth, self.K, self.current_pose())

    def close(self):
        self.zed.close()


class RealSenseCamera(Camera):
    """Intel RealSense (common alternative rig)."""

    def __init__(self, name: str, T_world_cam=None, serial: Optional[str] = None, size=(640, 480), fps=30):
        super().__init__(name, None, T_world_cam)
        import pyrealsense2 as rs  # lazy
        self.rs = rs
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, *size, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, *size, rs.format.z16, fps)
        self.pipe = rs.pipeline()
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

    def read(self) -> RGBDFrame:
        f = self.align.process(self.pipe.wait_for_frames())
        rgb = np.asanyarray(f.get_color_frame().get_data()).copy()
        depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * self.scale
        return RGBDFrame(self.name, rgb, depth, self.K, self.current_pose())

    def close(self):
        self.pipe.stop()


class ReplayCamera(Camera):
    """Replays recorded frames from disk: <dir>/<idx>_rgb.png, <idx>_depth.npy, plus calib.npz (K, T_world_cam)."""

    def __init__(self, name: str, directory: str, loop: bool = False):
        calib = np.load(os.path.join(directory, "calib.npz"))
        super().__init__(name, calib["K"], calib["T_world_cam"])
        self.rgbs = sorted(glob.glob(os.path.join(directory, "*_rgb.png")))
        self.idx, self.loop = 0, loop

    def read(self) -> RGBDFrame:
        from PIL import Image
        if self.idx >= len(self.rgbs):
            if not self.loop:
                raise StopIteration
            self.idx = 0
        p = self.rgbs[self.idx]
        self.idx += 1
        rgb = np.array(Image.open(p).convert("RGB"))
        depth = np.load(p.replace("_rgb.png", "_depth.npy")).astype(np.float32)
        return RGBDFrame(self.name, rgb, depth, self.K, self.current_pose())


def build_cameras(cfg: List[Dict]) -> List[Camera]:
    """cfg: list of {name, type: oakd|zed|realsense|replay, extrinsics: path-to-4x4.npy or list, ...}."""
    cams = []
    for c in cfg:
        c = dict(c)
        typ, name = c.pop("type"), c.pop("name")
        ext = c.pop("extrinsics", None)
        T = None
        if isinstance(ext, str):
            T = np.load(ext)
        elif ext is not None:
            T = np.asarray(ext, dtype=float)
        if typ == "oakd":
            cams.append(OakDCamera(name, T, **c))
        elif typ == "zed":
            cams.append(ZedCamera(name, T, **c))
        elif typ == "realsense":
            cams.append(RealSenseCamera(name, T, **c))
        elif typ == "replay":
            cams.append(ReplayCamera(name, **c))
        else:
            raise ValueError(f"unknown camera type {typ}")
    return cams
