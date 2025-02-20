import open3d as o3d
import numpy as np
import json
import os
from pupil_apriltags import Detector
import cv2
from frame_tools import FrameTools, R
from copy import deepcopy
from pathlib import Path

def create_axis_cylinders(radius=0.02, length=0.6):
    # 创建一个x轴圆柱（红色）
    cylinder_x = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder_x.paint_uniform_color([1, 0, 0])  # 红色
    # 旋转90度并平移到正确的位置
    transform_x = np.array([[0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]])
    cylinder_x.transform(transform_x)

    # 创建一个z轴圆柱（蓝色）
    cylinder_z = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder_z.paint_uniform_color([0, 0, 1])  # 蓝色
    # 无需旋转，但需要平移到x轴圆柱的一个端点
    transform_z = np.array([[1, 0, 0, length / 2],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    cylinder_z.transform(transform_z)

    # 合并两个圆柱体
    combined_mesh = cylinder_x + cylinder_z

    return combined_mesh

def create_camera_model(size=0.1):
    # mesh_camera = create_axis_cylinders()
    mesh_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    # mesh_camera.paint_uniform_color([0.9, 0.1, 0.1])
    return mesh_camera

def show_pose(camera_pose, size=0.1):
    # 应用相机位姿变换
    tmp_tans = np.eye(4)
    tmp_tans[2, 2] = -1
    camera_pose =  camera_pose @ tmp_tans
    camera_model = create_camera_model(size=size)
    camera_model.transform(camera_pose)
    return camera_model

def load_cam_poses(cam_poses_file, save_mark_pose=False):
    if save_mark_pose:
        with open(cam_poses_file, 'r') as f:
            datas = deepcopy(json.load(f))
        cam_list = []
        detector = Detector(families='tagStandard52h13',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
        for id, frame in enumerate(datas["frames"]):
            tmp_cam_dict = {}
            tmp_cam_dict["cam_pose"] = frame["transform_matrix"]
            tmp_cam_dict['file_name'] = frame["file_path"].split("/")[-1]
            tmp_cam_dict["marker_pose"] = detect_apriltags(frame, detector, 0.1)
            if tmp_cam_dict["marker_pose"] is None:
                del datas["frames"][id]
                continue
            datas["frames"][id]["transform_matrix"] = tmp_cam_dict["marker_pose"].tolist()
            cam_list.append(tmp_cam_dict)
        cam_poses_file = Path(cam_poses_file)
        cam_poses_output_file = cam_poses_file.parent / f"{cam_poses_file.stem}_marker.json"
        print(f"Save to {cam_poses_output_file.absolute()}")
        with open(cam_poses_output_file, "wt") as t:
            json.dump(datas, t)
        return cam_list
    else:
        with open(cam_poses_file, 'r') as f:
            datas = json.load(f)
        cam_list = []
        detector = Detector(families='tagStandard52h13',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
        for id, frame in enumerate(datas["frames"]):
            tmp_cam_dict = {}
            tmp_cam_dict["cam_pose"] = frame["transform_matrix"]
            tmp_cam_dict['file_name'] = frame["file_path"].split("/")[-1]
            tmp_cam_dict["marker_pose"] = detect_apriltags(frame, detector, 0.1)
            cam_list.append(tmp_cam_dict)
        return cam_list

def detect_apriltags(frame, detector, tag_size=0.05):
    base_path = "/home/pjlab/main/real2sim/gaussian-splatting/data/image_data/photo-marker/nerfstudio-data/"
    image_path = os.path.join(base_path, frame["file_path"])
    img = cv2.imread(image_path)
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
            marker_pose = cam_dict["marker_pose"]
            if marker_pose is None:
                continue
            tmp_cam_to_marker = marker_pose
            return tmp_cam_to_marker, id
    cam_to_marker, cam_id = get_one_pose(res_list)
    print(cam_id)
    frame_tools = FrameTools("polycam")
    for id, cam_dict in enumerate(res_list):
        cam_pose = cam_dict["cam_pose"]
        frame_tools.add_frame(f"cam_{id}", cam_pose)
    mark_to_gs = np.linalg.inv(cam_to_marker) @ res_list[cam_id]["cam_pose"]
    frame_tools.add_frame("marker", mark_to_gs)
    frame_tools.change_base_frame("marker")
    return frame_tools

def get_to_marker_pose(image, camera_params, tag_size, detector):
    '''
    get the pose: camera -> marker
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    if len(results) < 4:
        return None
    pose = np.eye(4)
    pose[:3, :3] = results[0].pose_R
    pose[:3, 3] = results[0].pose_t[:, 0]
    pose = np.linalg.inv(pose)
    hope2now = np.eye(4)
    hope2now[1, 1] = -1
    hope2now[2, 2] = -1
    now2hope = np.linalg.inv(hope2now)
    tmp_trans = np.eye(4)
    tmp_trans[2, 2] = -1
    pose = now2hope @ pose @ tmp_trans
    # pose[:3, 3] /= 2
    return pose

def main2():
    # pcd = o3d.io.read_point_cloud("/home/pjlab/main/real2sim/gaussian-splatting/data/image_data/photo-marker/nerfstudio-data/point_cloud.ply")
    cam_list = load_cam_poses("/home/pjlab/main/real2sim/gaussian-splatting/data/image_data/photo-marker/nerfstudio-data/transforms.json")
    frame_tools = align_frame(cam_list)
    # pcd.transform(frame_tools.get_frame("polycam"))
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    item_to_show = [frame_base]
    for id in range(len(cam_list)):
        cam_pose = frame_tools.get_frame(f"cam_{id}")
        cam_model = show_pose(cam_pose)
        item_to_show.append(cam_model)
    # item_to_show.append(pcd)
    o3d.visualization.draw_geometries(item_to_show, window_name="Camera and Mesh Visualization")

from utils.colmap_tools import * 
def main():
    # 创建点云
    # pcd = o3d.io.read_point_cloud("/home/pjlab/main/real2sim/Gaussian_Recon/data/mydata/table-603-polycam/nerfstudio-output/point_cloud.ply")

    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # 相机的位姿，这里简单设置一个例子
    cam_list = load_cam_poses("/home/pjlab/main/real2sim/gaussian-splatting/data/image_data/photo-marker/nerfstudio-data/transforms.json", save_mark_pose=False)

    cameras = read_cameras_binary("/home/pjlab/main/real2sim/gaussian-splatting/data/realsense-data/0907-2/20240907_135213/distorted/sparse/0/cameras.bin")
    images = read_images_binary("/home/pjlab/main/real2sim/gaussian-splatting/data/realsense-data/0907-2/20240907_135213/distorted/sparse/0/images.bin")
    
    item_to_show = []
    
    for image in images.values():
        qvec = image.qvec
        tvec = image.tvec
        rotmat = qvec2rotmat(qvec)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = rotmat
        cam_pose[:3, 3] = tvec
        cam_pose = np.linalg.inv(cam_pose)
        camera_model = show_pose(cam_pose)
        item_to_show.append(camera_model)
        
    
    # for cam_dict in cam_list:
    #     if cam_dict["marker_pose"] is None:
    #         continue
    #     # 变换相机模型
    #     camera_model = show_pose(cam_dict["marker_pose"])
    #     item_to_show.append(camera_model)
    
    # ft = FrameTools("ns")
    # for cam_dict in cam_list:
    #     ft.add_frame(cam_dict['file_name'], cam_dict["cam_pose"])
    
    # for cam_dict in cam_list:
    #     camera_model = show_pose(ft.get_frame(cam_dict["file_name"]))
    #     item_to_show.append(camera_model)
    
    item_to_show.append(frame_base)
    # 可视化
    # item_to_show.append(pcd)
    o3d.visualization.draw_geometries(item_to_show, window_name="Camera and Mesh Visualization")

    
if __name__ == "__main__":
    main()
