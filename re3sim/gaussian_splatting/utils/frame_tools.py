import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

## R.from_quat() [x, y, z, w]
class FrameTools:
    def __init__(self, base_frame_name) -> None:
        self.frames = {base_frame_name: np.eye(4)}
        self.base_frame_name = base_frame_name
    
    def add_frame_relative_to(self, frame_name, relative_to, transform_matrix):
        if relative_to not in self.frames:
            raise ValueError(f"Frame {relative_to} not found.")
        self.frames[frame_name] = self.frames[relative_to] @ transform_matrix
    
    def add_frame(self, frame_name, transform_matrix):
        self.frames[frame_name] = transform_matrix
    
    def get_frame(self, frame_name):
        return self.frames[frame_name]

    def get_frame_relative_to(self, frame_name, relative_to):
        return np.linalg.inv(self.frames[relative_to]) @ self.frames[frame_name]

    def change_base_frame(self, new_base_frame_name):
        if new_base_frame_name not in self.frames:
            raise ValueError(f"Frame {new_base_frame_name} not found.")
        new_base_frame = self.frames[new_base_frame_name]
        for frame_name, frame_matrix in self.frames.items():
            self.frames[frame_name] = np.linalg.inv(new_base_frame) @ frame_matrix
        self.base_frame_name = new_base_frame_name
        
def update_colmap_scale(colmap_poses, marker_poses):
    colmap_pose1 = None
    marker_pose1 = None
    colmap_pose2 = None
    marker_pose2 = None
    for colmap_pose, marker_pose in zip(colmap_poses, marker_poses):
        if marker_pose is None:
            continue
        if colmap_pose1 is None:
            colmap_pose1 = colmap_pose
            marker_pose1 = marker_pose
            continue
        if colmap_pose2 is None:
            colmap_pose2 = colmap_pose
            marker_pose2 = marker_pose
            break

    tmp_colmap_translations = []
    tmp_marker_translations = []
    for colmap_pose, marker_pose in zip(colmap_poses, marker_poses):
        if marker_pose is None:
            continue
        tmp_colmap_translations.append(colmap_pose[:3, 3])
        tmp_marker_translations.append(marker_pose[:3, 3])
    
    tmp_colmap_translations = np.array(tmp_colmap_translations)
    tmp_marker_translations = np.array(tmp_marker_translations)
    def error_function(params, colmap_poses, marker_poses):
        scale = params[0]
        quat = params[1:5]  # 四元数表示旋转
        translation = params[5:8]  # 平移
        rotation = R.from_quat(quat)
        
        scaled_colmap = scale * colmap_poses
        rotated_colmap = rotation.apply(scaled_colmap) + translation
        
        error = np.sum((rotated_colmap - marker_poses) ** 2)
        return error
    
    initial_params = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 使用scipy.optimize的minimize函数进行优化
    result = minimize(error_function, initial_params, args=(tmp_colmap_translations, tmp_marker_translations))
    colmap_delta_translation = colmap_pose2[:3, 3] - colmap_pose1[:3, 3]
    marker_delta_translation = marker_pose2[:3, 3] - marker_pose1[:3, 3]
    scale = np.linalg.norm(marker_delta_translation) / np.linalg.norm(colmap_delta_translation)
    print("Simple scale:", scale)
    print("Scale from optimization:", result.x)
    print("Final Error:", result.fun)
    for colmap_pose in colmap_poses:
        colmap_pose[:3, 3] *= result.x[0]
    return colmap_poses 
