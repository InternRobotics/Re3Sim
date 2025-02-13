import os
import cv2
import numpy as np


def read_data(base_dir):
    rgb_folder = os.path.join(base_dir, "rgb")
    depth_folder = os.path.join(base_dir, "depth")
    pose_folder = os.path.join(base_dir, "poses")
    joints_folder = os.path.join(base_dir, "joints")

    rgb_list, depth_list, pose_list, joints_list = None, None, None, None

    # Check if RGB folder exists
    if os.path.exists(rgb_folder):
        # Read RGB images
        rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith(".png")]
        rgb_files.sort()
        rgb_list = []
        for f in rgb_files:
            img = cv2.imread(os.path.join(rgb_folder, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_list.append(img)

    # Check if depth folder exists
    if os.path.exists(depth_folder):
        # Read depth images
        depth_files = [f for f in os.listdir(depth_folder) if f.endswith(".npy")]
        depth_files.sort()
        depth_list = [np.load(os.path.join(depth_folder, f)) for f in depth_files]

    # Check if pose folder exists
    if os.path.exists(pose_folder):
        # Read poses
        pose_files = [f for f in os.listdir(pose_folder) if f.endswith(".npy")]
        pose_files.sort()
        pose_list = [np.load(os.path.join(pose_folder, f)) for f in pose_files]

    if os.path.exists(joints_folder):
        # Read joints
        joints_files = [f for f in os.listdir(joints_folder) if f.endswith(".npy")]
        joints_files.sort()
        joints_list = [np.load(os.path.join(joints_folder, f)) for f in joints_files]

    # Check if camera parameters exist
    rgb_params_file = os.path.join(base_dir, "rgb_intrinsics.npz")
    if os.path.exists(rgb_params_file):
        # Load the intrinsic parameters
        camera_params = np.load(rgb_params_file)
        fx = camera_params["fx"]
        fy = camera_params["fy"]
        ppx = camera_params["ppx"]
        ppy = camera_params["ppy"]
        rgb_coeffs = camera_params["coeffs"]
        rgb_intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    else:
        rgb_intrinsics, rgb_coeffs = None, None

    depth_params_file = os.path.join(base_dir, "depth_intrinsics.npz")
    if os.path.exists(depth_params_file):
        # Load the intrinsic parameters
        camera_params = np.load(depth_params_file)
        fx = camera_params["fx"]
        fy = camera_params["fy"]
        ppx = camera_params["ppx"]
        ppy = camera_params["ppy"]
        depth_coeffs = camera_params["coeffs"]
        depth_intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        depth_scale = camera_params["depth_scale"]
    else:
        depth_intrinsics, depth_coeffs, depth_scale = None, None, None

    return (
        rgb_list,
        depth_list,
        pose_list,
        rgb_intrinsics,
        rgb_coeffs,
        depth_intrinsics,
        depth_coeffs,
        depth_scale,
        joints_list,
    )
