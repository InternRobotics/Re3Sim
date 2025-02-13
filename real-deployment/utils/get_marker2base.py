import roboticstoolbox as rtb
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
import cv2
import json
import os
from pathlib import Path
import re


def get_to_marker_pose(image, camera_params, tag_size, detector, min_num=4):
    """
    get the pose: camera -> marker
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(
        gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size
    )
    if len(results) < min_num:
        return None
    res = None
    for result in results:
        if result.tag_id == 22222:
            res = result
            break
    if res is None:
        return None
    pose = np.eye(4)
    pose[:3, :3] = res.pose_R
    pose[:3, 3] = res.pose_t[:, 0]
    pose = np.linalg.inv(pose)
    hope2now = np.eye(4)
    hope2now[1, 1] = -1
    hope2now[2, 2] = -1
    now2hope = np.linalg.inv(hope2now)
    pose = now2hope @ pose
    return pose


def create_camera_model(size=0.1):
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
    tmp_tans = np.eye(4)
    tmp_tans[2, 2] = -1
    # camera_pose =  camera_pose @ tmp_tans
    camera_model = create_camera_model(size)
    camera_model.transform(camera_pose)
    return camera_model


def test_marker():
    data_root = "/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/frankapy/tests/test-marker/data"
    data_root = Path(data_root)
    calibration_root = (
        "/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/assets/calibration"
    )
    robot = rtb.models.Panda()
    item_to_show = []
    marker_2_base_list = []
    ee_2_base_list = []
    cam_2_base_list = []
    camera_2_marker_list = []
    for joints_path in data_root.glob("joints_*.npy"):
        uuid = re.search(r"joints_(\w+).npy", joints_path.name).group(1)
        image = cv2.imread(str(data_root / f"color_{uuid}.png"))
        image_np = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detector = Detector(
            families="tagStandard52h13",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        with open(
            os.path.join(calibration_root, "intrinsic/wrist_camera.json"), "r"
        ) as f:
            intrinsic = json.load(f)["cameraMatrix"]
        with open(
            os.path.join(calibration_root, "easy_handeye/eye_on_hand/10-12.json"), "r"
        ) as f:
            calibration_results = json.load(f)
        cam_2_ee = np.eye(4)
        quat = np.array(
            [
                calibration_results["rotation"]["x"],
                calibration_results["rotation"]["y"],
                calibration_results["rotation"]["z"],
                calibration_results["rotation"]["w"],
            ]
        )
        cam_2_ee[:3, :3] = R.from_quat(quat, scalar_first=False).as_matrix()
        cam_2_ee[:3, 3] = np.array(
            [
                calibration_results["translation"]["x"],
                calibration_results["translation"]["y"],
                calibration_results["translation"]["z"],
            ]
        )

        # camera_params = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
        camera_params = [
            604.0827026367188,
            602.91064453125,
            330.3056640625,
            239.24717712402344,
        ]
        tag_size = 0.031
        cam_2_marker = get_to_marker_pose(
            image, camera_params, tag_size, detector, min_num=1
        )
        if cam_2_marker is None:
            print(f"image_{uuid} has no marker")
            continue
        camera_2_marker_list.append(cam_2_marker)
        qpos = np.load(joints_path)
        ee_2_base = np.array(robot.fkine(qpos, end="panda_hand"))
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
    print(
        "marker_2_base:",
        np.linalg.norm(marker_2_base_list[0][:3, 3] - marker_2_base_list[1][:3, 3]),
    )
    print(
        "ee_2_base:",
        np.linalg.norm(ee_2_base_list[0][:3, 3] - ee_2_base_list[1][:3, 3]),
    )
    print(
        "camera_2_marker:",
        np.linalg.norm(camera_2_marker_list[0][:3, 3] - camera_2_marker_list[1][:3, 3]),
    )
    print(
        "cam_2_base:",
        np.linalg.norm(cam_2_base_list[0][:3, 3] - cam_2_base_list[1][:3, 3]),
    )
    print("-" * 80)
    print("cam_2_base:", cam_2_base_list[0][:3, 3] - cam_2_base_list[1][:3, 3])
    print(
        "cam_2_marker:", camera_2_marker_list[0][:3, 3] - camera_2_marker_list[1][:3, 3]
    )
    print(np.mean(marker_2_base_list, axis=0))
    o3d.visualization.draw_geometries(item_to_show)
    np.save(
        "/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/assets/marker_2_base.npy",
        np.mean(marker_2_base_list, axis=0),
    )
    print("saved")


if __name__ == "__main__":
    test_marker()
