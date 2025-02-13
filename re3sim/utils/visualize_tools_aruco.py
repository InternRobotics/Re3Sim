import open3d as o3d
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
from real2sim2real.utils.colmap_tools import read_cameras_binary, read_images_binary
import cv2
from real2sim2real.utils.frame_tools import (
    FrameTools,
    get_transform_between_two_frames,
    get_transform_between_two_frames2,
)
from pathlib import Path
import roboticstoolbox as rtb
from typing import List
from real2sim2real.frankapy.src.utils.arcuo_marker import estimate_pose


def align_frame(res_list):
    def get_one_pose(res_list):
        for id, cam_dict in enumerate(res_list):
            estimated_pose = cam_dict["estimated_pose"]
            if estimated_pose is None:
                continue
            tmp_cam_to_marker = estimated_pose
            return tmp_cam_to_marker, id

    cam_to_marker, cam_id = get_one_pose(res_list)
    print(cam_id)
    frame_tools = FrameTools("GS")
    for id, cam_dict in enumerate(res_list):
        cam_pose = cam_dict["cam_pose"]
        frame_tools.add_frame(f"cam_{id}", cam_pose)
    mark_to_gs = np.linalg.inv(cam_to_marker) @ res_list[cam_id]["cam_pose"]
    frame_tools.add_frame("marker", mark_to_gs)
    frame_tools.change_base_frame("marker")
    return frame_tools


def read_polycam(
    polycam_folder,
    charuco_dict,
    board,
    dist_coeffs=None,
    RECOMPUTE=False,
    marker_size=0.1,
    sample_num=1000,
    mesh=None,
):
    corrected_cameras_path = polycam_folder / "keyframes/corrected_cameras"
    corrected_images = polycam_folder / "keyframes/corrected_images"
    if not corrected_cameras_path.exists():
        corrected_cameras_path = polycam_folder / "keyframes/cameras"
        corrected_images = polycam_folder / "keyframes/images"
    ids = [x.stem for x in corrected_cameras_path.iterdir()]
    if os.path.exists(polycam_folder / "to_marker_transform.npy") and not RECOMPUTE:
        polycam2marker = np.load(polycam_folder / "to_marker_transform.npy")
        camera_poses_in_marker = []
        for camera_id in ids:
            with open(corrected_cameras_path / f"{camera_id}.json", "r") as f:
                camera_params = json.load(f)
            polycam_transform = np.array(
                [
                    [
                        camera_params["t_00"],
                        camera_params["t_01"],
                        camera_params["t_02"],
                        camera_params["t_03"],
                    ],
                    [
                        camera_params["t_10"],
                        camera_params["t_11"],
                        camera_params["t_12"],
                        camera_params["t_13"],
                    ],
                    [
                        camera_params["t_20"],
                        camera_params["t_21"],
                        camera_params["t_22"],
                        camera_params["t_23"],
                    ],
                    [0, 0, 0, 1],
                ]
            )
            camera_poses_in_marker.append(polycam2marker @ polycam_transform)
        marker_camera_poses = None
    else:
        polycam_transforms = []
        marker_transforms = []
        for camera_id in ids:
            image = cv2.imread(str(corrected_images / f"{camera_id}.jpg"))
            with open(corrected_cameras_path / f"{camera_id}.json", "r") as f:
                camera_params = json.load(f)
            params = [
                camera_params["fx"],
                camera_params["fy"],
                camera_params["cx"],
                camera_params["cy"],
            ]
            polycam_transform = np.array(
                [
                    [
                        camera_params["t_00"],
                        camera_params["t_01"],
                        camera_params["t_02"],
                        camera_params["t_03"],
                    ],
                    [
                        camera_params["t_10"],
                        camera_params["t_11"],
                        camera_params["t_12"],
                        camera_params["t_13"],
                    ],
                    [
                        camera_params["t_20"],
                        camera_params["t_21"],
                        camera_params["t_22"],
                        camera_params["t_23"],
                    ],
                    [0, 0, 0, 1],
                ]
            )
            intrinsics_matrix = np.array(
                [[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]]
            )
            dist_coeffs = (
                np.zeros((5,)) if dist_coeffs is None else np.array(dist_coeffs)
            )
            marker_pose = estimate_pose(
                image, charuco_dict, intrinsics_matrix, dist_coeffs, board
            )
            polycam_transforms.append(polycam_transform @ rtb.ET.Rx(np.pi).A())
            marker_transforms.append(marker_pose)
        polycam2marker, camera_poses_in_marker = get_transform_between_two_frames2(
            polycam_transforms,
            marker_transforms,
            transform_type=True,
            sample_num=sample_num,
        )

        np.save(polycam_folder / "polycam_to_marker.npy", polycam2marker)
        marker_camera_poses = marker_transforms

    mesh.transform(polycam2marker)

    return camera_poses_in_marker, marker_camera_poses, mesh


def read_colmap(
    colmap_root_path,
    charuco_dict,
    board,
    RECOMPUTE=False,
    sample_num=1000,
    camera_params=None,
    dist_coeffs=None,
    reg=None,
):
    # transform colmap camera to marker
    colmap_cameras = read_cameras_binary(colmap_root_path / "sparse/0/cameras.bin")

    raw_colmap_camera_poses = read_images_binary(
        colmap_root_path / "sparse/0/images.bin"
    )

    if RECOMPUTE or not os.path.exists(
        colmap_root_path / "sparse/0/colmap_to_marker.npy"
    ):
        colmap_camera_poses = []
        marker_camera_poses = []
        image_names = []
        for camera_id, raw_colmap_camera_pose in raw_colmap_camera_poses.items():
            quat = raw_colmap_camera_pose.qvec
            translation = raw_colmap_camera_pose.tvec
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R.from_quat(
                [quat[0], quat[1], quat[2], quat[3]], scalar_first=True
            ).as_matrix()
            camera_pose[:3, 3] = translation
            camera_pose = np.linalg.inv(camera_pose)
            colmap_camera_poses.append(camera_pose)
            image = cv2.imread(
                str(colmap_root_path / "images" / raw_colmap_camera_pose.name)
            )
            if camera_params is None:
                tmp_camera_params = colmap_cameras[
                    raw_colmap_camera_pose.camera_id
                ].params
            else:
                tmp_camera_params = camera_params
            intrinsics_matrix = np.array(
                [
                    [tmp_camera_params[0], 0, tmp_camera_params[2]],
                    [0, tmp_camera_params[1], tmp_camera_params[3]],
                    [0, 0, 1],
                ]
            )
            dist_coeffs = (
                np.zeros((5,)) if dist_coeffs is None else np.array(dist_coeffs)
            )
            marker_pose = estimate_pose(
                image, charuco_dict, intrinsics_matrix, dist_coeffs, board
            )
            marker_camera_poses.append(marker_pose)
            image_names.append(raw_colmap_camera_pose.name)
        colmap2marker, camera_poses_in_marker = get_transform_between_two_frames2(
            colmap_camera_poses,
            marker_camera_poses,
            transform_type=True,
            sample_num=sample_num,
            reg=reg,
            img_names=image_names,
        )
        np.save(colmap_root_path / "sparse/0/colmap_to_marker.npy", colmap2marker)
    else:
        colmap_camera_poses = []
        camera_poses_in_marker = []
        colmap2marker = np.load(colmap_root_path / "sparse/0/colmap_to_marker.npy")
        for camera_id, raw_colmap_camera_pose in raw_colmap_camera_poses.items():
            quat = raw_colmap_camera_pose.qvec
            translation = raw_colmap_camera_pose.tvec
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R.from_quat(
                [quat[0], quat[1], quat[2], quat[3]], scalar_first=True
            ).as_matrix()
            camera_pose[:3, 3] = translation
            colmap_camera_poses.append(camera_pose)
            camera_poses_in_marker.append(colmap2marker @ camera_pose)
        marker_camera_poses = None
    return camera_poses_in_marker, marker_camera_poses, colmap2marker


def read_gaussian(
    gaussian_folder,
    charuco_dict,
    board,
    dist_coeffs,
    RECOMPUTE=False,
    reg=None,
    marker_size=0.1,
    sample_num=1000,
    mesh=None,
):
    gaussian_cameras_json_path = gaussian_folder / "cameras.json"
    with open(gaussian_cameras_json_path, "r") as f:
        gaussian_cameras = json.load(f)
    if RECOMPUTE or not os.path.exists(
        gaussian_folder / "to_marker_22222_transform.npy"
    ):
        gaussian_camera_poses = []
        marker_camera_poses = []
        image_names = []
        for gaussian_camera in gaussian_cameras:
            position = gaussian_camera["position"]
            rotation = gaussian_camera["rotation"]
            fx = gaussian_camera["fx"]
            fy = gaussian_camera["fy"]
            cx = gaussian_camera["width"] / 2
            cy = gaussian_camera["height"] / 2
            gaussian_camera_pose = np.eye(4)
            gaussian_camera_pose[:3, :3] = np.array(rotation)
            gaussian_camera_pose[:3, 3] = np.array(position)
            # gaussian_camera_pose = np.linalg.inv(gaussian_camera_pose)
            camera_params = [fx, fy, cx, cy]
            image_path = f"{gaussian_camera['image_path']}"
            image = cv2.imread(str(image_path))
            intrinsics_matrix = np.array(
                [
                    [camera_params[0], 0, camera_params[2]],
                    [0, camera_params[1], camera_params[3]],
                    [0, 0, 1],
                ]
            )
            dist_coeffs = (
                np.zeros((5,)) if dist_coeffs is None else np.array(dist_coeffs)
            )
            marker_pose = estimate_pose(
                image, charuco_dict, intrinsics_matrix, dist_coeffs, board
            )
            gaussian_camera_poses.append(gaussian_camera_pose)
            marker_camera_poses.append(marker_pose)
            image_names.append(gaussian_camera["img_name"])
        gaussian2marker, camera_poses_in_marker = get_transform_between_two_frames2(
            gaussian_camera_poses,
            marker_camera_poses,
            transform_type=True,
            img_names=image_names,
            reg=reg,
            sample_num=sample_num,
        )
        np.save(gaussian_folder / "gs_to_marker.npy", gaussian2marker)
    else:
        gaussian_camera_poses = []
        camera_poses_in_marker = []
        gaussian2marker = np.load(gaussian_folder / "gs_to_marker.npy")
        for gaussian_camera in gaussian_cameras:
            position = gaussian_camera["position"]
            rotation = gaussian_camera["rotation"]
            fx = gaussian_camera["fx"]
            fy = gaussian_camera["fy"]
            cx = gaussian_camera["width"] / 2
            cy = gaussian_camera["height"] / 2
            gaussian_camera_pose = np.eye(4)
            gaussian_camera_pose[:3, :3] = np.array(rotation)
            gaussian_camera_pose[:3, 3] = np.array(position)
            camera_params = [fx, fy, cx, cy]
            gaussian_camera_poses.append(gaussian_camera_pose)
            camera_poses_in_marker.append(gaussian2marker @ gaussian_camera_pose)
        marker_camera_poses = None
    if mesh is not None:
        mesh.transform(gaussian2marker)
    np.save(gaussian_folder / "gs_to_marker.npy", gaussian2marker)
    return camera_poses_in_marker, marker_camera_poses, mesh
    # return gaussian_camera_poses, marker_camera_poses, mesh


def read_3dgsr(
    gsr_folder,
    charuco_dict,
    board,
    RECOMPUTE=False,
    sample_num=1000,
    camera_params=None,
    dist_coeffs=None,
    reg=None,
):
    # transform colmap camera to marker
    gsr_folder = Path(gsr_folder)
    gsr_images_path = gsr_folder / "images"
    gsr_poses_path = gsr_folder / "poses"
    if RECOMPUTE or not os.path.exists(gsr_folder / "gs_to_marker.npy"):
        gsr_camera_poses = []
        marker_camera_poses = []
        image_names = []
        for image_path in gsr_images_path.iterdir():
            camera_pose_path = (
                gsr_poses_path / f"pose_{int(image_path.stem.split('_')[-1])}.npy"
            )
            if not os.path.exists(camera_pose_path):
                print(f"Camera pose path {camera_pose_path} does not exist")
                continue
            camera_pose = np.load(camera_pose_path)
            image = cv2.imread(str(gsr_images_path / image_path.name))
            if camera_params is None:
                colmap_camera_params_path = gsr_folder / f"sparse/0/cameras.bin"
                colmap_camera_params = read_cameras_binary(colmap_camera_params_path)
                tmp_camera_params = colmap_camera_params[1].params
            else:
                tmp_camera_params = camera_params
            intrinsics_matrix = np.array(
                [
                    [tmp_camera_params[0], 0, tmp_camera_params[2]],
                    [0, tmp_camera_params[1], tmp_camera_params[3]],
                    [0, 0, 1],
                ]
            )
            dist_coeffs = (
                np.zeros((5,)) if dist_coeffs is None else np.array(dist_coeffs)
            )
            marker_pose = estimate_pose(
                image, charuco_dict, intrinsics_matrix, dist_coeffs, board
            )
            gsr_camera_poses.append(camera_pose)
            marker_camera_poses.append(marker_pose)
            image_names.append(image_path.name)
        gsr2marker, camera_poses_in_marker = get_transform_between_two_frames2(
            gsr_camera_poses,
            marker_camera_poses,
            transform_type=True,
            sample_num=sample_num,
            reg=reg,
            img_names=image_names,
        )
        np.save(gsr_folder / "gs_to_marker.npy", gsr2marker)
        np.save(gsr_folder / "mesh_to_marker.npy", gsr2marker)
    else:
        gsr_camera_poses = []
        camera_poses_in_marker = []
        gsr2marker = np.load(gsr_folder / "gs_to_marker.npy")
        for image_path in gsr_images_path.iterdir():
            camera_pose_path = (
                gsr_poses_path / f"poses_{int(image_path.stem.split('_')[-1])}.npy"
            )
            camera_pose = np.load(camera_pose_path)
            gsr_camera_poses.append(camera_pose)
            camera_poses_in_marker.append(gsr2marker @ camera_pose)
        marker_camera_poses = None
    return camera_poses_in_marker, marker_camera_poses, gsr2marker, gsr_camera_poses


if __name__ == "__main__":
    main3()
