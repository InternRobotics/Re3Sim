import numpy as np
import requests
import pickle
import cv2
import os
from scipy.spatial.transform import Rotation as R
from typing import List, Any, Dict, Union, Optional, Tuple

try:
    from pxr.Usd import Prim
except:
    print("No pxr found")
    Prim = None
os.environ["NO_PROXY"] = (
    os.environ["NO_PROXY"] + "\," + "localhost"
    if "NO_PROXY" in os.environ
    else "localhost"
)
import logging

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
from multiprocessing.shared_memory import SharedMemory
import random
import time
import torch

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix

    Args:
        q: List or numpy array of 4 elements representing quaternion [qx, qy, qz, qw]

    Returns:
        R: 3x3 numpy array representing rotation matrix
    """
    qx, qy, qz, qw = q

    R = np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )

    return R


def euler_to_quaternion(yaw, pitch, roll):
    """
    Convert Euler angles to quaternion

    Args:
        yaw: Rotation around z-axis in radians
        pitch: Rotation around y-axis in radians
        roll: Rotation around x-axis in radians

    Returns:
        q: Numpy array with 4 elements representing quaternion [qx, qy, qz, qw]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return np.array([qx, qy, qz, qw])


def compute_fx_fy(camera, height, width):
    focal_length = camera.get_focal_length()
    horiz_aperture = camera.get_horizontal_aperture()
    vert_aperture = camera.get_vertical_aperture()
    near, far = camera.get_clipping_range()
    fov = 2 * np.arctan(0.5 * horiz_aperture / focal_length)

    focal_x = height * focal_length / vert_aperture
    focal_y = width * focal_length / horiz_aperture
    return focal_x, focal_y


def get_intrinsic_matrix(camera):
    fx, fy = compute_fx_fy(
        camera, camera.get_resolution()[1], camera.get_resolution()[0]
    )
    cx, cy = camera.get_resolution()[0] / 2, camera.get_resolution()[1] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def set_semantic_label(prim: Prim, label):
    from omni.isaac.core.utils.semantics import add_update_semantics

    if prim.GetTypeName() == "Mesh":
        add_update_semantics(prim, semantic_label=label, type_label="class")
    all_children = prim.GetAllChildren()
    for child in all_children:
        set_semantic_label(child, label)


def wait_for_data(width, height, control_data: np.ndarray, data_shm: SharedMemory):
    while True:
        try:
            if control_data[0] <= 0.5:
                buffer = np.ndarray(
                    (height, width, 3), dtype=np.uint8, buffer=data_shm.buf
                )
                image = np.array(buffer)
                return image
        except Exception as e:
            print(control_data)
            raise e


def send_by_shm(control_data: np.ndarray, cam_pose, fx, fy, width, height):
    cam_trans, cam_rot = cam_pose
    control_data[1:4] = cam_trans
    control_data[4:8] = cam_rot
    control_data[8] = fx
    control_data[9] = fy
    control_data[10] = width
    control_data[11] = height
    control_data[0] = 1.0


def get_rendered_images(cam_pose, height, width, fx, fy):
    control_shm = SharedMemory(name="control_psm_08d5dd701")
    data_shm = SharedMemory(name="data_psm_08d5dd701")
    control_data = np.ndarray((12,), dtype=np.float64, buffer=control_shm.buf)
    position = cam_pose[0]
    rotation = cam_pose[1]
    send_by_shm(control_data, cam_pose, fx, fy, width, height)
    image = wait_for_data(width, height, control_data, data_shm)
    return image


def remove_ndarray_in_dict(dict):
    for key in dict.keys():
        if isinstance(dict[key], np.ndarray):
            dict[key] = dict[key].tolist()
        if isinstance(dict[key], Dict):
            remove_ndarray_in_dict(dict[key])
    return dict


def pose_to_transform(pose):
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return transform


def _transform_grasp(
    grasp_position: np.ndarray,
    grasp_rotation_matrix: np.ndarray,
    camera_position,
    camera_quaternion,
):
    camera_to_world_transform = pose_to_transform((camera_position, camera_quaternion))
    grasp_to_camera_transform = pose_to_transform(
        (
            grasp_position,
            R.from_matrix(grasp_rotation_matrix).as_quat(scalar_first=True),
        )
    )
    grasp_to_world_transform = camera_to_world_transform @ grasp_to_camera_transform
    grasp_rotation = grasp_to_world_transform[:3, :3]
    temp_rotation = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    grasp_rotation = grasp_rotation @ temp_rotation
    return grasp_to_world_transform[:3, 3], R.from_matrix(grasp_rotation).as_quat(
        scalar_first=True
    )


def adjust_translation_along_quaternion(translation, quaternion, distance):
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    direction_vector = rotation.apply([0, 0, 1])
    reverse_direction = -direction_vector
    new_translation = translation + reverse_direction * distance
    return new_translation


def get_anygrasp_pose(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_params,
    url="http://10.6.8.89:5001/process",
    camera_pose=None,
    bias=0.1,
):

    debug = False  # Can be set to False as needed
    if debug:
        # Ensure save directory exists
        save_dir = "/tmp/anygrasp_debug"
        os.makedirs(save_dir, exist_ok=True)

        # Save RGB image
        rgb_filename = os.path.join(save_dir, "rgb_debug.png")
        cv2.imwrite(rgb_filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Save depth image 
        depth_filename = os.path.join(save_dir, "depth_debug.png")
        cv2.imwrite(depth_filename, (depth * 255).astype(np.uint8))

        print(f"Debug images saved to {save_dir}")
    fx, fy, cx, cy = camera_params
    depth = np.nan_to_num(depth)
    depth = np.clip(depth, 0, 1)
    data = {
        "colors": rgb.tolist(),
        "depths": (depth * 1000).tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "scale": 1000.0,
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        processed_data = response.json()
        grasp_list = []
        if "grasp_groups" not in processed_data:
            print(processed_data.get("error", "Unknown error"))
            return []
        for grasp in processed_data["grasp_groups"]:
            translation = np.array(grasp["translation"])
            rotation_matrix = np.array(grasp["rotation_matrix"])
            translation, orientation = _transform_grasp(
                translation, rotation_matrix, camera_pose[0], camera_pose[1]
            )
            translation = adjust_translation_along_quaternion(
                translation, orientation, bias
            )
            grasp_list.append(
                {
                    "translation": translation,
                    "orientation": orientation,
                }
            )
        return grasp_list
    else:
        print("Failed to process data:", response.text)
        return None


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unset_seed():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    t = int(1000 * time.time()) % (2**31 - 1)
    np.random.seed(t)
    random.seed(t)
    torch.manual_seed(t)
    torch.cuda.manual_seed_all(t)
