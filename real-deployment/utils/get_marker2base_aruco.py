import roboticstoolbox as rtb
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2
import json
import os
from pathlib import Path
import re


def estimate_pose(image, charuco_dict, intrinsics_matrix, dist_coeffs, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_dict)

    if len(corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )  # can not pass
        if charuco_ids is not None and len(charuco_corners) > 3:
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                board,
                intrinsics_matrix,
                dist_coeffs,
                None,
                None,
            )
            if valid:
                R_target2cam = cv2.Rodrigues(rvec)[0]
                t_target2cam = tvec.reshape(3, 1)
                target2cam = np.eye(4)
                target2cam[:3, :3] = R_target2cam
                target2cam[:3, 3] = t_target2cam.reshape(-1)
                return np.linalg.inv(target2cam)
    return None

def create_camera_model(size=0.1):
    # Create a simple camera model (Frustum shape)
    mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0, 0, 0]
    )
    # mesh_camera.paint_uniform_color([0.9, 0.1, 0.1])
    return mesh_camera


def show_pose(camera_pose, size=0.1):
    camera_pose = np.array(camera_pose)
    camera_pose[:3, :3] = camera_pose[:3, :3] / np.abs(
        (np.linalg.det(camera_pose[:3, :3]))
    ) ** (1 / 3)
    # Apply camera pose transformation
    tmp_tans = np.eye(4) 
    tmp_tans[2, 2] = -1
    # camera_pose = camera_pose @ tmp_tans
    camera_model = create_camera_model(size)
    camera_model.transform(camera_pose)
    return camera_model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="path to data root directory")
    args = parser.parse_args()
    data_root = args.data_root
    data_root = Path(data_root)
    intrinsic_path = data_root / "rgb_intrinsics.npz"
    robot = rtb.models.Panda()
    item_to_show = []
    marker_2_base_list = []
    ee_2_base_list = []
    cam_2_base_list = []
    camera_2_marker_list = []
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, charuco_dict)
    for joints_path in data_root.glob("joints/joints_*.npy"):
        uuid = re.search(r"joints_(\w+).npy", joints_path.name).group(1)
        image = cv2.imread(str(data_root / f"rgb/{uuid}.png"))

        intrinsic_zip = np.load(intrinsic_path)
        fx = intrinsic_zip["fx"]
        fy = intrinsic_zip["fy"]
        cx = intrinsic_zip["ppx"]
        cy = intrinsic_zip["ppy"]
        camera_params = [fx, fy, cx, cy]
        cam_2_ee = np.load(data_root / "cam_to_hand_pose.npy")
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        tag_size = 0.031
        cam_2_marker = estimate_pose(
            image, charuco_dict, intrinsic_matrix, np.zeros(5), board
        )
        if cam_2_marker is None:
            print(f"image_{uuid} has no marker")
            continue
        camera_2_marker_list.append(cam_2_marker)
        qpos = np.load(joints_path)
        ee_2_base = np.array(robot.fkine(qpos, end="panda_hand").A)
        ee_2_base_list.append(ee_2_base)
        cam_2_base = ee_2_base @ cam_2_ee
        marker_2_cam = np.linalg.inv(cam_2_marker)
        marker_2_base = cam_2_base @ marker_2_cam
        marker_2_base_list.append(marker_2_base)
        cam_2_base_list.append(cam_2_base)
        # item_to_show.append(show_pose(cam_2_marker))
        item_to_show.append(show_pose(marker_2_base))
    frame_base = show_pose(np.eye(4))
    item_to_show.append(frame_base)
    print(f"Marker to base:\n{np.mean(marker_2_base_list, axis=0)}")
    o3d.visualization.draw_geometries(item_to_show)
    np.save(f"{data_root}/marker_2_base.npy", np.mean(marker_2_base_list, axis=0))
    print("saved")


if __name__ == "__main__":
    main()
