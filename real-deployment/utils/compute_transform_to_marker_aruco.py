import argparse
import numpy as np
import os
from real2sim2real.utils.visualize_tools_aruco import (
    read_gaussian,
    read_polycam,
    read_colmap,
    read_3dgsr,
)
from real2sim2real.utils.visualize_tools import show_pose
from pathlib import Path
import open3d as o3d
import json
import cv2

charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, charuco_dict)


def process_polycam_data(
    polycam_folder,
    headless=True,
    detector_family="tagStandard52h13",
    marker_size=0.1,
    detector_threads=1,
    sample_num=1000,
):
    polycam_folder = Path(polycam_folder)
    mesh = o3d.io.read_triangle_mesh(str(polycam_folder / "raw.glb"))
    export_success = o3d.io.write_triangle_mesh(
        str(polycam_folder / "raw.obj"),
        mesh,
        write_ascii=True,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True,
        print_progress=True,
    )

    if export_success:
        print("Export obj successful")
    else:
        print("Export obj failed")
    with open(polycam_folder / "mesh_info.json", "r") as f:
        mesh_info = json.load(f)
    alignmentTransform = np.array(mesh_info["alignmentTransform"]).reshape(4, 4).T
    np.save(polycam_folder / "mesh_to_polycam.npy", np.linalg.inv(alignmentTransform))
    mesh.transform(np.linalg.inv(alignmentTransform))

    camera_poses_in_marker, marker_camera_poses, mesh = read_polycam(
        polycam_folder,
        charuco_dict,
        board,
        RECOMPUTE=True,
        marker_size=marker_size,
        sample_num=sample_num,
        mesh=mesh,
    )
    if headless:
        return camera_poses_in_marker, marker_camera_poses, mesh
    item_to_show = []
    for camera_pose in camera_poses_in_marker:
        camera_model = show_pose(camera_pose)
        item_to_show.append(camera_model)
    item_to_show.append(mesh)
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    item_to_show.append(frame_base)
    o3d.visualization.draw_geometries(
        item_to_show, window_name="Polycam Camera in Marker and Mesh Visualization"
    )
    mesh2polycam = np.load(polycam_folder / "mesh_to_polycam.npy")
    polycam2marker = np.load(polycam_folder / "polycam_to_marker.npy")
    mesh2marker = np.dot(polycam2marker, mesh2polycam)
    np.save(polycam_folder / "mesh_to_marker.npy", mesh2marker)
    print(f"Scale: {np.linalg.det(mesh2marker) ** (1/3)}")
    return camera_poses_in_marker, marker_camera_poses, mesh


def process_gaussian_data(
    gs_folder,
    headless=True,
    detector_family="tagStandard52h13",
    marker_size=0.1,
    detector_threads=1,
    reg=None,
    sample_num=1000,
):
    gs_folder = Path(gs_folder)
    camera_poses_in_marker, marker_camera_poses, mesh = read_gaussian(
        gs_folder,
        charuco_dict,
        board,
        None,
        RECOMPUTE=True,
        marker_size=marker_size,
        reg=reg,
        sample_num=sample_num,
    )
    if headless:
        return camera_poses_in_marker, marker_camera_poses, mesh
    item_to_show = []
    for camera_pose in camera_poses_in_marker:
        camera_model = show_pose(camera_pose)
        item_to_show.append(camera_model)
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    item_to_show.append(frame_base)
    o3d.visualization.draw_geometries(
        item_to_show, window_name="GS Camera in Marker and Mesh Visualization"
    )
    return camera_poses_in_marker, marker_camera_poses, mesh


def process_openmvs_data(
    openmvs_folder,
    headless=True,
    detector_family="tagStandard52h13",
    marker_size=0.1,
    detector_threads=1,
    sample_num=1000,
    reg=None,
    use_predefined_camera_params=False,
):
    openmvs_folder = Path(openmvs_folder)
    mesh = o3d.io.read_triangle_mesh(
        str(openmvs_folder / "scene_dense_mesh_refine_texture.ply")
    )

    if use_predefined_camera_params:
        camera_params = [912.89596470432798, 908.08585399271976, 640, 360]
        rgb_coeffs = [0, 0, 0, 0, 0]
    else:
        camera_params = None
    camera_poses_in_marker, marker_camera_poses, colmap2marker = read_colmap(
        openmvs_folder,
        charuco_dict=charuco_dict,
        board=board,
        RECOMPUTE=True,
        sample_num=sample_num,
        camera_params=camera_params,
        reg=reg,
    )
    mesh.transform(colmap2marker)
    if headless:
        return camera_poses_in_marker, marker_camera_poses, mesh
    item_to_show = []
    for camera_pose in camera_poses_in_marker:
        camera_model = show_pose(camera_pose)
        item_to_show.append(camera_model)
    item_to_show.append(mesh)
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    item_to_show.append(frame_base)
    o3d.visualization.draw_geometries(
        item_to_show, window_name="OpenMVS Camera in Marker and Mesh Visualization"
    )
    return camera_poses_in_marker, marker_camera_poses, mesh


def process_3dgsr_data(
    gs_folder,
    headless=True,
    detector_family="tagStandard52h13",
    marker_size=0.1,
    detector_threads=1,
    sample_num=1000,
    reg=None,
    use_predefined_camera_params=False,
):
    gs_folder = Path(gs_folder)
    mesh = o3d.io.read_triangle_mesh(str(gs_folder / "lego_test_mesh_high.ply"))
    if use_predefined_camera_params:
        camera_params = [912.89596470432798, 908.08585399271976, 640, 360]
        rgb_coeffs = [0, 0, 0, 0, 0]
    else:
        camera_params = None
    camera_poses_in_marker, marker_camera_poses, gsr2marker, gsr_camera_poses = (
        read_3dgsr(
            gs_folder,
            charuco_dict=charuco_dict,
            board=board,
            RECOMPUTE=True,
            sample_num=sample_num,
            camera_params=camera_params,
            reg=reg,
        )
    )
    mesh.transform(gsr2marker)
    if headless:
        return camera_poses_in_marker, marker_camera_poses, mesh
    item_to_show = []
    for camera_pose in camera_poses_in_marker:
        # for camera_pose in marker_camera_poses:
        if camera_pose is None:
            continue
        camera_model = show_pose(camera_pose)
        item_to_show.append(camera_model)
    item_to_show.append(mesh)
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    item_to_show.append(frame_base)
    o3d.visualization.draw_geometries(
        item_to_show, window_name="OpenMVS Camera in Marker and Mesh Visualization"
    )
    return camera_poses_in_marker, marker_camera_poses, mesh


def main():
    arg_parser = argparse.ArgumentParser("Process data to get tranfrom to marker")
    arg_parser.add_argument(
        "--data_type",
        type=str,
        help="Type of data to process",
        choices=["polycam", "gaussian", "openmvs", "3dgsr"],
        default="3dgsr",
    )
    arg_parser.add_argument(
        "--data_folder",
        type=str,
        help="Path to the folder containing the data",
        default="/isaac-sim/src/assets/data/3dgsr/new-1024-2/",
    )
    arg_parser.add_argument(
        "--headless", action="store_true", help="Whether to show the visualization"
    )
    arg_parser.add_argument(
        "--detector_family",
        type=str,
        default="tagStandard52h13",
        help="Family of the detector",
    )
    arg_parser.add_argument(
        "--marker_size", type=float, default=0.031, help="Size of the marker"
    )
    arg_parser.add_argument(
        "--detector_threads",
        type=int,
        default=1,
        help="Number of threads to use for the detector",
    )
    arg_parser.add_argument(
        "--reg",
        type=str,
        default=None,
        help="Regularization parameter for the transformation",
    )
    arg_parser.add_argument(
        "--sample_num",
        type=int,
        default=50000,
        help="Number of samples to use for the transformation",
    )
    arg_parser.add_argument(
        "--use_predefined_camera_params",
        action="store_true",
        help="Whether to use predefined camera parameters",
    )
    args = arg_parser.parse_args()
    if args.data_type == "polycam":
        process_polycam_data(
            args.data_folder,
            args.headless,
            args.detector_family,
            args.marker_size,
            args.detector_threads,
            args.sample_num,
        )
    elif args.data_type == "gaussian":
        process_gaussian_data(
            args.data_folder,
            args.headless,
            args.detector_family,
            args.marker_size,
            args.detector_threads,
            args.reg,
            args.sample_num,
        )
    elif args.data_type == "openmvs":
        process_openmvs_data(
            args.data_folder,
            args.headless,
            args.detector_family,
            args.marker_size,
            args.detector_threads,
            args.sample_num,
            args.reg,
            args.use_predefined_camera_params,
        )
    elif args.data_type == "3dgsr":
        process_3dgsr_data(
            args.data_folder,
            args.headless,
            args.detector_family,
            args.marker_size,
            args.detector_threads,
            args.sample_num,
            args.reg,
            args.use_predefined_camera_params,
        )
    else:
        print("Invalid data type")


if __name__ == "__main__":
    main()
