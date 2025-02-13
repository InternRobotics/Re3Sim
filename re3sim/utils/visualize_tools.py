import open3d as o3d
import numpy as np
import json
import os

try:
    from pupil_apriltags import Detector
except ImportError:
    Detector = None
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


def visualize_pcds_poses(
    pcds: List[np.ndarray] = [],
    poses: List[np.ndarray] = [],
    window_name="pcd",
    show_frame=True,
):
    item_to_show = []
    for pcd in pcds:
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        item_to_show.append(o3d_pcd)
    for pose in poses:
        camera_model = show_pose(pose)
        item_to_show.append(camera_model)
    if show_frame:
        frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]
        )
        item_to_show.append(frame_base)
    o3d.visualization.draw_geometries(item_to_show, window_name=window_name)


def create_camera_model(size=0.1):
    # 创建一个简单的相机模型（Frustum形状）
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
    camera_model = create_camera_model(size)
    camera_model.transform(camera_pose)
    return camera_model


def load_cam_poses(cam_poses_file):
    with open(cam_poses_file, "r") as f:
        datas = json.load(f)
    cam_list = []
    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    for id, frame in enumerate(datas["frames"]):
        tmp_cam_dict = {}
        tmp_cam_dict["cam_pose"] = frame["transform_matrix"]
        tmp_cam_dict["estimated_pose"] = detect_apriltags(frame, detector, 0.1)
        cam_list.append(tmp_cam_dict)
    return cam_list


def detect_apriltags(frame, detector, tag_size=0.05):
    base_path = "/home/pjlab/main/real2sim/Gaussian_Recon/data/mydata/table-603-polycam/nerfstudio-output"
    image_path = os.path.join(base_path, frame["file_path"])
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fx = frame["fl_x"]
    fy = frame["fl_y"]
    cx = frame["cx"]
    cy = frame["cy"]
    tag_size = tag_size
    cam2marker = get_to_marker_pose(img, [fx, fy, cx, cy], tag_size, detector)
    return cam2marker


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
    for i, result in enumerate(results):
        if result.tag_id == 22222 or i == len(results) - 1:
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


def read_polycam(
    polycam_folder, detector, mesh, RECOMPUTE=False, marker_size=0.1, sample_num=1000
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
            marker_transform = get_to_marker_pose(image, params, marker_size, detector)
            polycam_transforms.append(polycam_transform @ rtb.ET.Rx(np.pi).A())
            marker_transforms.append(marker_transform)
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
    detector,
    RECOMPUTE=False,
    marker_size=0.1,
    sample_num=1000,
    camera_params=None,
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
            marker_pose = get_to_marker_pose(
                image, tmp_camera_params, marker_size, detector
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
    detector,
    mesh,
    RECOMPUTE=False,
    reg=None,
    marker_size=0.1,
    sample_num=1000,
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
            marker_pose = get_to_marker_pose(
                image, camera_params, marker_size, detector
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
        np.save(gaussian_folder / "to_marker_22222_transform.npy", gaussian2marker)
    else:
        gaussian_camera_poses = []
        camera_poses_in_marker = []
        gaussian2marker = np.load(gaussian_folder / "to_marker_22222_transform.npy")
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

def main():
    # Create point cloud
    pcd = o3d.io.read_point_cloud(
        "/home/pjlab/main/real2sim/Gaussian_Recon/data/mydata/table-603-polycam/nerfstudio-output/point_cloud.ply"
    )

    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    # Camera poses, simple example here
    cam_list = load_cam_poses(
        "/home/pjlab/main/real2sim/Gaussian_Recon/data/mydata/table-603-polycam/nerfstudio-output/transforms.json"
    )

    item_to_show = []
    for cam_dict in cam_list:
        # Transform camera model
        camera_model = show_pose(cam_dict["cam_pose"]) 
        item_to_show.append(camera_model)

    item_to_show.append(frame_base)
    # Visualization
    item_to_show.append(pcd)
    o3d.visualization.draw_geometries(
        item_to_show, window_name="Camera and Mesh Visualization"
    )


def main2():
    pcd = o3d.io.read_point_cloud(
        "/home/pjlab/main/real2sim/Gaussian_Recon/data/mydata/table-603-polycam/nerfstudio-output/point_cloud.ply"
    )
    cam_list = load_cam_poses(
        "/home/pjlab/main/real2sim/Gaussian_Recon/data/mydata/table-603-polycam/nerfstudio-output/transforms.json"
    )
    frame_tools = align_frame(cam_list)
    pcd.transform(frame_tools.get_frame("GS"))
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    item_to_show = [frame_base]
    for id in range(len(cam_list)):
        cam_pose = frame_tools.get_frame(f"cam_{id}")
        cam_model = show_pose(cam_pose)
        item_to_show.append(cam_model)
    item_to_show.append(pcd)
    o3d.visualization.draw_geometries(
        item_to_show, window_name="Camera and Mesh Visualization"
    )


def main3():
    RECOMPUTE = True
    mesh_type = "gaussian"
    # mesh_type = "polycam"

    root_path = Path("/home/pjlab/main/real2sim/assets")
    mesh = o3d.io.read_triangle_mesh(
        str(root_path / "meshes/processed/0909-mix-2/0909-mix-2.obj")
    )
    mesh = o3d.io.read_triangle_mesh(
        str(root_path / "meshes/processed/polycam-0909-1/raw.glb")
    )
    polycam_folder = "/home/pjlab/main/real2sim/assets/meshes/processed/polycam-0909-1"
    polycam_folder = Path(polycam_folder)

    polycam_mesh = o3d.io.read_triangle_mesh(
        str(root_path / "meshes/processed/polycam-0909-1/raw.glb")
    )
    polycam_mesh_to_marker = np.load(polycam_folder / "to_marker_transform.npy")
    polycam_mesh.transform(polycam_mesh_to_marker)
    colmap_root_path = root_path / "data/mix2" 
    gaussian_folder = root_path / "data/mix5-sugar/gs-output/1"

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    # Transform polycam mesh to marker
    if mesh_type == "polycam":
        camera_poses_in_marker, marker_camera_poses, mesh = read_polycam(
            polycam_folder, detector, mesh, RECOMPUTE
        )
    elif mesh_type == "colmap":
        camera_poses_in_marker, marker_camera_poses, mesh = read_colmap(
            colmap_root_path, detector, mesh, RECOMPUTE
        )
    elif mesh_type == "gaussian":
        camera_poses_in_marker, marker_camera_poses, mesh = read_gaussian(
            gaussian_folder, detector, mesh, RECOMPUTE, reg=r"^\d{12}$"
        )

    def show_window(items, window_name):
        o3d.visualization.draw_geometries(items, window_name=window_name)

    item_to_show1 = []
    for cam_pose in camera_poses_in_marker:
        if cam_pose is None:
            continue
        camera_model = show_pose(cam_pose)
        item_to_show1.append(camera_model)
    # item_to_show1.append(mesh)
    item_to_show1.append(polycam_mesh)
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    item_to_show1.append(frame_base)

    if RECOMPUTE or not os.path.exists(
        colmap_root_path / "sparse/0/to_marker_22222_transform.npy"
    ):
        item_to_show2 = []
        for cam_pose in marker_camera_poses:
            if cam_pose is None:
                continue
            camera_model = show_pose(cam_pose)
            item_to_show2.append(camera_model)
        item_to_show2.append(mesh)
        item_to_show2.append(frame_base)
    import threading

    # Create thread
    thread1 = threading.Thread(
        target=show_window,
        args=(item_to_show1, "Colmap Camera in Marker and Mesh Visualization"),
    )

    # Start thread
    thread1.start()

    # Wait for thread completion
    thread1.join()
    print("Exit")


def test_marker():
    camera_params = [2799.3406691660498, 2662.5620746736886, 1152, 2048]
    image_path = "/home/pjlab/main/real2sim/assets/data/test-marker/images"
    image_path = Path(image_path)
    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    item_to_show = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB
    i = 0
    for image in image_path.iterdir():
        print(image)
        img = cv2.imread(str(image))
        pose = get_to_marker_pose(img, camera_params, 0.1, detector, min_num=1)
        camera_model = show_pose(pose)
        camera_model.paint_uniform_color(colors[i])
        i += 1
        item_to_show.append(camera_model)
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    item_to_show.append(base_frame)
    o3d.visualization.draw_geometries(
        item_to_show, window_name="Camera and Mesh Visualization"
    )


def test_polycam():
    root_path = Path("/home/pjlab/main/real2sim/assets")
    mesh = o3d.io.read_triangle_mesh(
        str(root_path / "meshes/processed/polycam-0909-1/raw.glb")
    )
    polycam_folder = "/home/pjlab/main/real2sim/assets/meshes/processed/polycam-0909-1"
    polycam_folder = Path(polycam_folder)
    with open(polycam_folder / "mesh_info.json", "r") as f:
        mesh_info = json.load(f)
    alignmentTransform = np.array(mesh_info["alignmentTransform"]).reshape(4, 4).T
    mesh.transform(np.linalg.inv(alignmentTransform))

    corrected_cameras_path = polycam_folder / "keyframes/corrected_cameras"
    corrected_images = polycam_folder / "keyframes/corrected_images"
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


if __name__ == "__main__":
    main3()
