import open3d as o3d
import numpy as np
from PIL import Image
from utils.colmap_tools import qvec2rotmat, read_cameras_binary, read_images_binary
from utils.visualize_tools import show_pose, get_to_marker_pose
import cv2
import os
from utils.frame_tools import update_colmap_scale
import random
from pupil_apriltags import Detector

def downsample_point_cloud(pcd, target_number_of_points):
    """对点云进行下采样"""
    # 如果原始点云大于目标点数
    if np.asarray(pcd.points).shape[0] > target_number_of_points:
        # 随机选择
        choice = np.random.choice(np.asarray(pcd.points).shape[0], target_number_of_points, replace=False)
        downsampled_pcd = pcd.select_by_index(choice)
        return downsampled_pcd
    else:
        return pcd
    
def read_image(image_path):
    """读取图像文件"""
    return np.array(Image.open(image_path))

def create_point_cloud(depth_image, color_image, camera_intrinsics, camera_pose, width, height, fx, fy, cx, cy):
    """根据深度图、RGB图和相机位姿生成彩色点云"""
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_o3d = o3d.geometry.Image(color_image)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    
    # 应用相机位姿
    pcd.transform(camera_pose)
    camera_pose_model = show_pose(camera_pose)
    pcd += camera_pose_model.sample_points_uniformly(number_of_points=10000)

    return pcd

def merge_point_clouds(point_clouds, target_number_of_points=200000):
    """合并多个点云"""
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        merged_pcd += pcd
    merged_pcd = downsample_point_cloud(merged_pcd, target_number_of_points)
    return merged_pcd

def smooth_point_cloud(pcd):
    """对点云进行平滑处理"""
    # 使用半径邻域平滑
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    return pcd

def select_items(iterables, num):
    """
    从一组长度相同的iterable中随机均匀选出num个项。
    Args:
    - iterables: 一个包含多个iterable的列表，每个iterable的长度必须相同。
    - num: 要从每个iterable中选出的元素数量。
    
    Returns:
    - 一个列表，包含筛选后的iterables。
    """
    if not iterables:
        return []
    
    # 确保所有iterable的长度相同
    length = len(iterables[0])
    for iterable in iterables:
        if len(iterable) != length:
            raise ValueError("所有iterable的长度必须相同")
    
    # 如果请求的数量大于等于iterable的长度，直接返回原iterables
    if num >= length:
        return iterables
    
    # 生成随机但固定的索引列表
    selected_indices = sorted(random.sample(range(length), num))
    
    # 使用选定的索引从每个iterable中选取元素
    selected_iterables = []
    for iterable in iterables:
        selected_iterable = [iterable[i] for i in selected_indices]
        selected_iterables.append(selected_iterable)
    
    return selected_iterables


def main():
    from pathlib import Path
    root_dir = '/home/pjlab/main/real2sim/gaussian-splatting/data/mix'
    root_dir = Path(root_dir)
    depth_root_dir = root_dir / "depth"
    image_root_dir = root_dir / "input"
    cameras = read_cameras_binary(root_dir / "sparse/0/cameras.bin")
    images = read_images_binary(root_dir / "sparse/0/images.bin")
    
    camera_info = cameras[1]
    width = camera_info.width
    height = camera_info.height
    fx = camera_info.params[0]
    fy = camera_info.params[1]
    cx = camera_info.params[2]
    cy = camera_info.params[3]
    depths = []
    rgbs = []
    poses = []
    marker_poses = []
    detector = Detector(families='tagStandard52h13',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)
    camera_params = list(cameras[1].params[0:4])
    for id, image in images.items():
        depth_name = image.name.split('.')[0].split('_')[0] + ".npy"
        depth_path = depth_root_dir / depth_name
        if not os.path.exists(depth_path):
            continue
        image_path = image_root_dir / image.name
        qvec = image.qvec
        tvec = image.tvec
        rotmat = qvec2rotmat(qvec)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = rotmat
        cam_pose[:3, 3] = tvec
        cam_pose = np.linalg.inv(cam_pose)
        depths.append(depth_path)
        rgbs.append(image_path)
        poses.append(cam_pose)
        image = cv2.imread(str(image_path))
        marker_pose = get_to_marker_pose(image, camera_params, 0.1, detector)
        marker_poses.append(marker_pose)
    
    depths, rgbs, poses, marker_poses = select_items([depths, rgbs, poses, marker_poses], 30)

    poses = update_colmap_scale(poses, marker_poses)
    point_clouds = []
    for depth_path, rgb_path, pose in zip(depths, rgbs, poses):
        depth_image = np.load(depth_path)
        color_image = read_image(rgb_path)
        pcd = create_point_cloud(depth_image, color_image, None, pose, width, height, fx, fy, cx, cy)
        point_clouds.append(pcd)
    
    print("Point clouds created.")
    merged_pcd = merge_point_clouds(point_clouds, 1000000)
    # smoothed_pcd = smooth_point_cloud(merged_pcd)
    
    print("Point cloud after downsample has {} points.".format(len(merged_pcd.points)))

    merged_pcd
    # 可视化
    o3d.visualization.draw_geometries([merged_pcd])


def main_for_polycam():
    from pathlib import Path
    import json
    root_dir = "/home/pjlab/main/real2sim/gaussian-splatting/data/items/orange-bottle/item-polycam-9-21-1/keyframes"
    root_dir = Path(root_dir)
    depth_root_dir = root_dir / "depth"
    image_root_dir = root_dir / "corrected_images"
    camera_dir = root_dir / "corrected_cameras"
    names = [i.name for i in image_root_dir.iterdir()]
    names.sort(key=lambda x: int(os.path.splitext(x)[0]))
    for name in names:
        image_path = image_root_dir / name
        depth_path = depth_root_dir / name.replace("jpg", "png")
        camera_path = camera_dir / name.replace("jpg", "json")
        with open(camera_path, 'r') as file:
            camera_info = json.load(file)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        image = cv2.imread(str(image_path))
        print(depth.shape)
        print(image.shape)


if __name__ == "__main__":
    main_for_polycam()
