import numpy as np
import os
import sys
import open3d as o3d
import cv2
from tqdm import trange
import roboticstoolbox as rtb

parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from calibration.hand_in_eye import HandinEyeCalibrator
from calibration.utils import read_data


def tcp_to_hand(pose):
    panda = rtb.models.Panda()
    joints = panda.ik_LM(pose)[0]
    hand_pose = panda.fkine(joints, end="panda_hand").A
    return hand_pose


def joint_to_hand(joints):
    panda = rtb.models.Panda()
    hand_pose = panda.fkine(joints, end="panda_hand").A
    return hand_pose


# Read data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, help="path to data root directory")
args = parser.parse_args()
base_dir = args.data_root
(
    rgb_list,
    depth_list,
    pose_list,
    rgb_intrinsics,
    rgb_coeffs,
    depth_intrinsics,
    depth_coeffs,
    depth_scale,
    joints_list,
) = read_data(base_dir)
if joints_list is not None:
    pose_list = [joint_to_hand(joints) for joints in joints_list]
else:
    pose_list = [tcp_to_hand(pose) for pose in pose_list]
print(f"{len(rgb_list)} poses found")
print(f"Camera matrix: {rgb_intrinsics}")

# Calibrate
charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, charuco_dict)

calibrator = HandinEyeCalibrator(rgb_intrinsics, rgb_coeffs, charuco_dict, board)
R_cam2hand_avg, t_cam2hand_avg = calibrator.perform(rgb_list, pose_list)

print("Average Camera to hand rotation matrix:")
print(R_cam2hand_avg)
print("Average Camera to hand translation vector:")
print(t_cam2hand_avg)
cam_to_hand_pose = np.eye(4)
cam_to_hand_pose[:3, :3] = R_cam2hand_avg
cam_to_hand_pose[:3, 3] = t_cam2hand_avg.squeeze()
print(f"Camera to hand pose:\n{cam_to_hand_pose}")
np.save(f"{base_dir}/cam_to_hand_pose.npy", cam_to_hand_pose)
