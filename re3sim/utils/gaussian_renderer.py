import os
import torch
import numpy as np

from real2sim2real.gaussian_splatting.gaussian_renderer import render
import json
from real2sim2real.gaussian_splatting.utils.graphics_utils import (
    getWorld2View2,
    focal2fov,
    fov2focal,
)
from real2sim2real.gaussian_splatting.utils.camera_utils import cameraList_from_camInfos
import cv2
import time
from tqdm import tqdm
from real2sim2real.gaussian_splatting.scene.cameras import Camera
from multiprocessing import shared_memory
from scipy.spatial.transform import Rotation as Rota
from real2sim2real.gaussian_splatting.gaussian_renderer import GaussianModel

try:
    from real2sim2real.gaussian_splatting.gaussian_renderer import GaussianModel_3dgsr
except:
    print("GaussianModel_3dgsr not found")
    pass
import roboticstoolbox as rtb


class SimplePipeline:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def render_set(views, gaussians, pipeline, background):

    res = []
    for idx, view in enumerate(views):
        rendering = render(view, gaussians, pipeline, background)["render"]
        res.append(rendering.permute(1, 2, 0).nan_to_num().clamp(min=0, max=1))
    return res


class GaussianRenderer:
    def __init__(
        self,
        gaussian_model_path="/home/pjlab/main/real2sim/assets/data/new/mix2/gs-output/1/point_cloud/iteration_30000/point_cloud.ply",
        device="cuda",
        is_3dgsr=False,
    ):
        if is_3dgsr:
            self.gaussians = GaussianModel_3dgsr(3)
        else:
            self.gaussians = GaussianModel(3)
        self.gaussians.load_ply(gaussian_model_path)
        self.pipeline = SimplePipeline()
        self.tmp_transform = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
        self.tmp_transform = np.linalg.inv(self.tmp_transform)
        self.device = torch.device(device)
        pass

    def render(self, cam_pose, height, width, fx, fy):
        camera_translation, camera_orientation = cam_pose
        w, x, y, z = camera_orientation
        camera_orientation = np.array([x, y, z, w])
        camera_orientation = Rota.from_quat(camera_orientation).as_matrix()
        W2C = np.eye(4)
        W2C[:3, :3] = camera_orientation
        W2C[:3, 3] = camera_translation
        W2C = (
            W2C
            @ rtb.ET.Rz(np.pi / 2).A()
            @ rtb.ET.Ry(np.pi / 2).A()
            @ self.tmp_transform
        )
        Rt = np.linalg.inv(W2C)
        R = Rt[:3, :3].transpose()
        T = Rt[:3, 3]
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        img_name = "000"
        height = int(height)
        width = int(width)
        sample_image = torch.zeros(
            (3, height, width), dtype=torch.float32, device=self.device
        )
        cam = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=sample_image,
            gt_alpha_mask=None,
            image_name=img_name,
            uid=0,
            data_device=self.device,
        )
        cams = [cam]
        with torch.no_grad():
            res = render_set(
                cams,
                self.gaussians,
                self.pipeline,
                torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device),
            )
        image = res[0].detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        return image
