from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import os
from pathlib import Path
from ..utils.frame_tools import FrameTools
from ..utils.gaussian_renderer import GaussianRenderer
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import XFormPrim
from scipy.spatial.transform import Rotation as R

try:
    from ..utils.gaussian_renderer_new_new import ForegroundGSRenderer
except:
    print("ForegroundGSRenderer not found")


class Background(ABC):
    def __init__(
        self,
        asset_root: str,
        root_path: str,
        marker_to_isaacsim: np.ndarray = None,
        params: Dict[str, Any] = None,
    ):
        self.usd_path = os.path.join(asset_root, params.get("usd_path"))
        self.prim_path = root_path + "/background" + params.get("prim_path", "/Table")
        self.mesh_transform_matrix = np.load(
            Path(self.usd_path).parent / "mesh_to_marker.npy"
        )

        self.gs_path = os.path.join(asset_root, params.get("gs_path"))
        self.gs_transform_matrix = np.load(Path(self.gs_path) / "gs_to_marker.npy")
        self.gs_renderer = GaussianRenderer(
            gaussian_model_path=os.path.join(
                self.gs_path, "point_cloud/iteration_30000/point_cloud.ply"
            )
        )

        self.frames = FrameTools("marker")
        self.frames.add_frame_transform_relative_to(
            "mesh", "marker", self.mesh_transform_matrix
        )
        self.frames.add_frame_transform_relative_to(
            "gs", "marker", self.gs_transform_matrix
        )
        self.frames.add_frame_transform_relative_to(
            "isaacsim", "marker", np.linalg.inv(marker_to_isaacsim)
        )
        self.bg_prim = None

    def load(self):
        create_prim(
            prim_path=self.prim_path,
            usd_path=self.usd_path,
        )
        self.bg_prim = XFormPrim(self.prim_path)
        self.bg_prim.set_local_pose(
            translation=self.frames.get_frame_translation_relative_to(
                "mesh", "isaacsim"
            ),
            orientation=self.frames.get_frame_rotation_relative_to(
                "mesh", "isaacsim", "quat"
            ),
        )
        self.bg_prim.set_local_scale(
            scale=np.array([1.0] * 3)
            * self.frames.get_frame_scale_relative_to("mesh", "isaacsim")
        )

    def render(self, cam_pose, width, height, fx, fy, camera_pose_frame="marker"):
        """
        Params:
            cam_pose: np.ndarray, shape (4, 4), in camera_pose_frame
            width: int
            height: int
            fx: float
            fy: float
            camera_pose_frame: str, default "marker"
        """
        cam_pose = (
            self.frames.get_frame_transform_relative_to(camera_pose_frame, "gs")
            @ cam_pose
        )
        rotation_matrix = cam_pose[:3, :3] / np.abs(
            (np.linalg.det(cam_pose[:3, :3])) ** (1 / 3)
        )
        translation = cam_pose[:3, 3]
        x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
        cam_pose = [translation, [w, x, y, z]]
        return self.gs_renderer.render(
            cam_pose=cam_pose, width=width, height=height, fx=fx, fy=fy
        )


class Foreground(ABC):
    def __init__(
        self,
        asset_root: str,
        root_path: str,
        marker_to_isaacsim: np.ndarray = None,
        params: Dict[str, Any] = None,
    ):
        self.usd_path = os.path.join(asset_root, params.get("usd_path"))
        self.prim_path = root_path + "/foreground" + params.get("prim_path")
        self.mesh_transform_matrix = np.load(
            Path(self.usd_path).parent / "mesh_to_marker.npy"
        )

        self.gs_path = os.path.join(asset_root, params.get("gs_path"))
        self.gs_transform_matrix = np.load(Path(self.gs_path) / "gs_to_marker.npy")

        self.fore_gs_renderer = ForegroundGSRenderer(foreground_gs=self.gs_path)

        self.gs_usds = {"name": "/prim_path"}
        self.gs_usds = ["/prim_path1", "/prim_path2"]

        self.frames = FrameTools("marker")
        self.frames.add_frame_transform_relative_to(
            "mesh", "marker", self.mesh_transform_matrix
        )
        self.frames.add_frame_transform_relative_to(
            "gs", "marker", self.gs_transform_matrix
        )
        self.frames.add_frame_transform_relative_to(
            "isaacsim", "marker", np.linalg.inv(marker_to_isaacsim)
        )
        self.bg_prim = None

    def load(self):
        create_prim(
            prim_path=self.prim_path,
            usd_path=self.usd_path,
        )
        self.bg_prim = XFormPrim(self.prim_path)
        self.bg_prim.set_local_pose(
            translation=self.frames.get_frame_translation_relative_to(
                "mesh", "isaacsim"
            ),
            orientation=self.frames.get_frame_rotation_relative_to(
                "mesh", "isaacsim", "quat"
            ),
        )
        self.bg_prim.set_local_scale(
            scale=np.array([1.0] * 3)
            * self.frames.get_frame_scale_relative_to("mesh", "isaacsim")
        )

    def render(
        self,
        cam_pose,
        width,
        height,
        fx,
        fy,
        camera_pose_frame="marker",
        poses: Dict[str, np.ndarray] = None,
    ):
        """
        Params:
            cam_pose: np.ndarray, shape (4, 4), in camera_pose_frame
            width: int
            height: int
            fx: float
            fy: float
            camera_pose_frame: str, default "marker",
            poses: Dict[str, np.ndarray], gs model poses
        """
        cam_pose = (
            self.frames.get_frame_transform_relative_to(camera_pose_frame, "gs")
            @ cam_pose
        )
        rotation_matrix = cam_pose[:3, :3] / np.abs(
            (np.linalg.det(cam_pose[:3, :3])) ** (1 / 3)
        )
        translation = cam_pose[:3, 3]
        x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
        cam_pose = [translation, [w, x, y, z]]
        return self.fore_gs_renderer.render(
            cam_pose=cam_pose,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            gaussian_poses=poses,
        )
