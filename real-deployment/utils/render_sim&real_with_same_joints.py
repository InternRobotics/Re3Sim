from panda_py import libfranka
import panda_py
import h5py
from pathlib import Path
import time
from src.frankapy.src.realsense_reader import MultiRealSenseCamera
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import open3d as o3d
import imageio


def draw_contrast_image(sim_image, real_image, save_dir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(sim_image)
    ax1.set_title("模拟图像")
    ax1.axis("off")

    ax2.imshow(real_image)
    ax2.set_title("真实图像")
    ax2.axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir.parent, exist_ok=True)
        plt.savefig(save_dir)
    else:
        plt.show()

    plt.close()


def depth_to_pointcloud(depth, intrinsics):
    height, width = depth.shape
    fx, fy, cx, cy = intrinsics

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    xyz = np.stack([x, y, z], axis=-1)
    return xyz.reshape(-1, 3)


def draw_contrast_depth(sim_depth, real_depth, save_dir=None, idx=None):
    min_depth = min(sim_depth.min(), real_depth.min())
    max_depth = max(sim_depth.max(), real_depth.max())
    sim_depth_normalized = (sim_depth - min_depth) / (max_depth - min_depth)
    real_depth_normalized = (real_depth - min_depth) / (max_depth - min_depth)

    sim_depth_normalized = (sim_depth_normalized * 255).astype(np.uint8)
    real_depth_normalized = (real_depth_normalized * 255).astype(np.uint8)

    sim_depth_normalized = Image.fromarray(sim_depth_normalized)
    real_depth_normalized = Image.fromarray(real_depth_normalized)

    sim_depth_normalized.save(save_dir / f"sim_depth_{idx}.png")
    real_depth_normalized.save(save_dir / f"real_depth_{idx}.png")

    intrinsics = [907.3532104492188, 907.6549682617188, 655.828125, 362.16796875]

    sim_pcd = depth_to_pointcloud(sim_depth, intrinsics)
    real_pcd = depth_to_pointcloud(real_depth, intrinsics)

    sim_o3d = o3d.geometry.PointCloud()
    sim_o3d.points = o3d.utility.Vector3dVector(sim_pcd)
    sim_o3d.paint_uniform_color([1, 0, 0]) 

    real_o3d = o3d.geometry.PointCloud()
    real_o3d.points = o3d.utility.Vector3dVector(real_pcd)
    real_o3d.paint_uniform_color([0, 0, 1]) 

    o3d.visualization.draw_geometries([sim_o3d])
    o3d.visualization.draw_geometries([real_o3d])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(sim_o3d)
    vis.add_geometry(real_o3d)

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])

    vis.run()

    if save_dir:
        os.makedirs(save_dir.parent, exist_ok=True)
        vis.capture_screen_image(str(save_dir / f"sim_real_depth_pcd_{idx}.png"))

    vis.destroy_window()


def save_images(images, save_dir, step):
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        image.save(save_dir / f"color_{step}_{i}.png")


def get_image_by_key(dataset_root_path, key, idx):
    image_path = dataset_root_path / f"observations/{key}.jpg"
    image = imageio.imread(image_path)
    return image


def get_mask_by_key(dataset_root_path, key, idx):
    mask_path = dataset_root_path / f"observations/{key}.png"
    mask = imageio.imread(mask_path)
    return mask


def my_clip(depth, min_depth, max_depth):
    return np.where(depth < min_depth, 0, np.where(depth > max_depth, 0, depth))


hdf5_file_path = Path(
    "/isaac-sim/src/logs/collect_data_render_1110_low_res_continuous/2024-11-11_13-56-18_972230/log-000001-8525/traj.hdf5"
)
hdf5_file = h5py.File(hdf5_file_path, "r")

idxes = list(range(0, len(hdf5_file[f"observations/qpos"]), 5))[2:]
camera_name = "wrist_camera"

# real_camera_name = {
#     "right_frames": 0,
#     "wrist_camera": 2,
# }
real_camera_name = {
    "camera_0": 0,
    "wrist_camera": 1,
}

joints = []
images = []
gripper_widths = []
for idx in idxes:
    joints.append(hdf5_file[f"observations/qpos"][idx])
    # images.append(hdf5_file[f"observations/images/{camera_name}"][idx])
    gripper_widths.append(sum(hdf5_file[f"action"][idx][7:]))

franka = panda_py.Panda("172.16.0.2")
gripper = libfranka.Gripper("172.16.0.2")
multicamera = MultiRealSenseCamera(fps=30, image_width=640, image_height=480)

for i in range(20):
    real_image, real_depth = multicamera.undistorted_rgbd()
real_depths = []
real_images = []
global_gripper_width = 0.08
for i, (idx, joint) in enumerate(zip(idxes, joints)):
    franka.move_to_joint_position(joint[:7])
    time.sleep(1)
    gripper_width = gripper_widths[i]
    if gripper_width > 0.04:
        gripper_width = 0.08
    else:
        gripper_width = 0.0
    if abs(gripper_width - global_gripper_width) > 0.01:
        gripper.grasp(width=gripper_width, speed=0.1, force=40, epsilon_outer=0.08)
        global_gripper_width = gripper_width
    real_images, real_depth = multicamera.undistorted_rgbd()
    real_image = real_images[real_camera_name[camera_name]]
    real_depth = my_clip(real_depth[real_camera_name[camera_name]], 0.15, 3)
    os.makedirs(hdf5_file_path.parent / "rgb", exist_ok=True)
    save_images(real_images, save_dir=(hdf5_file_path.parent / "rgb"), step=idx)
    real_images.append(real_image)
    real_depths.append(real_depth)

np.savez(
    hdf5_file_path.parent / "real.npz",
    joints=joints,
    images=real_images,
    depths=real_depths,
)
