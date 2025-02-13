import numpy as np
from scipy.spatial.transform import Rotation as R
import pyquaternion as pyq
from scipy.optimize import minimize
import json
import os
import torch


## R.from_quat() [x, y, z, w]
class FrameTools:
    def __init__(self, base_frame_name) -> None:
        self.frames = {base_frame_name: np.eye(4)}
        self.base_frame_name = base_frame_name

    def add_frame_transform_relative_to(
        self, frame_name, relative_to, transform_matrix, ignore_exist=False
    ):
        if relative_to not in self.frames:
            raise ValueError(f"Frame {relative_to} not found.")
        if frame_name in self.frame_names and not ignore_exist:
            raise ValueError(
                f"Frame {frame_name} already exists in {self.base_frame_name}"
            )
        self.frames[frame_name] = self.frames[relative_to] @ transform_matrix

    def add_frame_transform(self, frame_name, transform_matrix, ignore_exist=False):
        if frame_name in self.frame_names and not ignore_exist:
            raise ValueError(
                f"Frame {frame_name} already exists in {self.base_frame_name}"
            )
        self.frames[frame_name] = transform_matrix

    def get_frame_transform(self, frame_name):
        return self.frames[frame_name]

    def get_frame_transform_relative_to(self, frame_name, relative_to):
        return np.linalg.inv(self.frames[relative_to]) @ self.frames[frame_name]

    def change_base_frame(self, new_base_frame_name):
        if new_base_frame_name not in self.frames:
            raise ValueError(f"Frame {new_base_frame_name} not found.")
        new_base_frame = self.frames[new_base_frame_name]
        for frame_name, frame_matrix in self.frames.items():
            self.frames[frame_name] = np.linalg.inv(new_base_frame) @ frame_matrix
        self.base_frame_name = new_base_frame_name

    def get_frame_translation(self, frame_name):
        return self.frames[frame_name][:3, 3]

    def get_frame_scale(self, frame_name):
        return (np.linalg.det(self.frames[frame_name][:3, :3])) ** (1 / 3)

    def get_frame_rotation(self, frame_name, type="matrix"):
        """
        Get rotation matrix or quaternion of the frame
        type: "matrix" or "quat"(w x y z)
        """
        rotation_matrix = self.frames[frame_name][:3, :3] / self.get_frame_scale(
            frame_name
        )
        if type == "matrix":
            return rotation_matrix
        elif type == "quat":
            x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
            return np.array([w, x, y, z])
        else:
            raise ValueError(f"Unknown type {type}")

    def get_frame_translation_relative_to(self, frame_name, relative_to):
        relative_transform = self.get_frame_transform_relative_to(
            frame_name, relative_to
        )
        return relative_transform[:3, 3]

    def get_frame_scale_relative_to(self, frame_name, relative_to):
        relative_transform = self.get_frame_transform_relative_to(
            frame_name, relative_to
        )
        return (np.linalg.det(relative_transform[:3, :3])) ** (1 / 3)

    def get_frame_rotation_relative_to(self, frame_name, relative_to, type="matrix"):
        """
        Get rotation matrix or quaternion of the frame
        type: "matrix" or "quat"
        """
        relative_transform = self.get_frame_transform_relative_to(
            frame_name, relative_to
        )
        rotation_matrix = relative_transform[:3, :3] / (
            np.linalg.det(relative_transform[:3, :3])
        ) ** (1 / 3)
        if type == "matrix":
            return rotation_matrix
        elif type == "quat":
            x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
            return np.array([w, x, y, z])

    @property
    def frame_names(self):
        return self.frames.keys()

    def merge_frames(self, frame1_name, frame1_transform, frame1):
        frame1: FrameTools
        self.add_frame_transform(frame1_name, frame1_transform)
        for frame_name in frame1.frame_names:
            self.add_frame_transform_relative_to(
                frame_name, frame1_name, frame1.get_frame_transform(frame_name)
            )

    def apply_scale_to(self, name, scale):
        self.frames[name][:3, :3] *= scale

    def apply_translation_to(self, name, translation):
        if isinstance(translation, list):
            translation = np.array(translation)
        self.frames[name][:3, 3] += translation

    def save(self, file_path):
        """
        Save the frames to a JSON file.

        Parameters:
            file_path (str): The path to the file where frames will be saved.
        """
        data = {"base_frame_name": self.base_frame_name, "frames": {}}
        for frame_name, transform in self.frames.items():
            data["frames"][
                frame_name
            ] = transform.tolist()  # Convert numpy array to list for JSON serialization

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Frames successfully saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        """
        Load frames from a JSON file and create a FrameTools instance.

        Parameters:
            file_path (str): The path to the file from which frames will be loaded.

        Returns:
            FrameTools: An instance of FrameTools with loaded frames.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "r") as f:
            data = json.load(f)

        base_frame_name = data.get("base_frame_name")
        if not base_frame_name:
            raise ValueError("Base frame name is missing in the file.")

        frames_data = data.get("frames")
        if not frames_data:
            raise ValueError("Frames data is missing in the file.")

        frame_tools = cls(base_frame_name)
        for frame_name, transform_list in frames_data.items():
            transform_matrix = np.array(transform_list)
            frame_tools.frames[frame_name] = transform_matrix

        print(f"Frames successfully loaded from {file_path}")
        return frame_tools


def _optimize_transform_1(tmp_frame1_translations, tmp_frame2_translations):
    def error_function(params, frame1_poses, frame2_poses):
        scale = params[0]
        quat = params[1:5]  
        quat = quat / np.linalg.norm(quat)
        translation = params[5:8]  
        quat_scale_last = np.array([quat[1], quat[2], quat[3], quat[0]])
        rotation = R.from_quat(quat_scale_last)  # x y z w Here

        scaled_colmap = scale * frame1_poses
        rotated_colmap = rotation.apply(scaled_colmap) + translation

        error = np.sum((rotated_colmap - frame2_poses) ** 2)
        return error

    initial_params = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = minimize(
        error_function,
        initial_params,
        args=(tmp_frame1_translations, tmp_frame2_translations),
    )
    scale = result.x[0]
    quat = result.x[1:5]
    translation = result.x[5:8]
    rotation_matrix = pyq.Quaternion(quat).rotation_matrix
    print(f"Using {len(tmp_frame1_translations)} translations to optimized.")
    print("result from optimization:", result.x)
    print("Final Error:", result.fun)
    return scale, rotation_matrix, quat, translation


def get_transform_between_two_frames(
    frame1_poses, frame2_poses, transform_type=True, img_names=None, reg=None
):
    """
    Get transform from frame1 to frame2
    Parameters:
    frame1_poses: list of np.ndarray, shape=(4, 4)
    frame2_poses: list of np.ndarray, shape=(4, 4)
    Return:
    Scale, Rotation, Translation, frame1_poses_in_frame2
    Rotation is represented by a quaternion [w, x, y, z]
    """
    if isinstance(frame1_poses, FrameTools) and isinstance(frame2_poses, FrameTools):
        frame1_names = frame1_poses.frame_names
        frame2_names = frame2_poses.frame_names
        common_names = list(set(frame1_names) & set(frame2_names))
        frame1_poses_lst = [
            frame1_poses.get_frame_transform(frame_name) for frame_name in common_names
        ]
        frame2_poses_lst = [
            frame2_poses.get_frame_transform(frame_name) for frame_name in common_names
        ]
        return get_transform_between_two_frames(
            frame1_poses_lst, frame2_poses_lst, transform_type
        )
    else:
        reg_match_idx = 0
        tmp_frame1_translations = []
        tmp_frame2_translations = []
        for i, (frame1_pose, frame2_pose) in enumerate(zip(frame1_poses, frame2_poses)):
            if reg is not None:
                assert isinstance(reg, str), "reg should be a string"
                assert img_names is not None, "img_names should be provided"
                import re

                if re.match(reg, img_names[i]) is None:
                    continue
                reg_match_idx += 1
            if frame1_pose is None or frame2_pose is None:
                continue
            tmp_frame1_translations.append(frame1_pose[:3, 3])
            tmp_frame2_translations.append(frame2_pose[:3, 3])

        tmp_frame1_translations = np.array(tmp_frame1_translations)
        tmp_frame2_translations = np.array(tmp_frame2_translations)

        if reg is not None:
            print(f"Reg match {reg_match_idx} images.")
        scale, rotation_matrix, quat, translation = _optimize_transform_1(
            tmp_frame1_translations, tmp_frame2_translations
        )
        new_frame1_poses = []
        for frame1_pose in frame1_poses:
            original_translation = frame1_pose[:3, 3]
            original_rotation = frame1_pose[:3, :3]
            new_translation = (
                scale * rotation_matrix @ original_translation + translation
            )
            new_orientation = rotation_matrix @ original_rotation
            new_pose = np.eye(4)
            new_pose[:3, :3] = new_orientation
            new_pose[:3, 3] = new_translation
            new_frame1_poses.append(new_pose)

        if transform_type:
            transform = np.eye(4)
            transform[:3, :3] = scale * rotation_matrix
            transform[:3, 3] = translation
            return transform, new_frame1_poses
        else:
            return scale, quat, translation, new_frame1_poses


def _optimize_transform_2(tmp_frame1_poses, tmp_frame2_poses):
    """
    Use scipy.optimize.minimize to optimize the transformation error
    """

    def normalize_transform(transform):
        scale = np.linalg.det(transform[:3, :3]) ** (1 / 3)
        rotation = transform[:3, :3] / scale
        translation = transform[:3, 3]
        res = np.eye(4)
        res[:3, :3] = rotation
        res[:3, 3] = translation
        return res

    def error_function(params, frame1_poses, frame2_poses):
        scale = params[0]
        quat = params[1:5]  
        quat = quat / np.linalg.norm(quat)
        translation = params[5:8]  
        quat_scale_last = np.array([quat[1], quat[2], quat[3], quat[0]])
        rotation = R.from_quat(quat_scale_last)  # x y z w Here

        transformation = np.eye(4)
        transformation[:3, :3] = scale * rotation.as_matrix()
        transformation[:3, 3] = translation

        quat_self = params[8:12]
        rotation_self = R.from_quat(quat_self).as_matrix()
        transformation_self = np.eye(4)
        transformation_self[:3, :3] = rotation_self

        frame1_poses_after_transform = np.array(
            [
                normalize_transform(transformation @ frame1_pose @ transformation_self)
                for frame1_pose in frame1_poses
            ]
        )
        normalized_frame2_poses = np.array(
            [normalize_transform(frame2_pose) for frame2_pose in frame2_poses]
        )
        error = np.sum((frame1_poses_after_transform - normalized_frame2_poses) ** 2)
        return error + 0.001 * np.sum(params**2)

    initial_params = np.array(
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    result = minimize(
        error_function, initial_params, args=(tmp_frame1_poses, tmp_frame2_poses)
    )  # TODO: tmp_frame2_poses is wrong, det == -1
    scale = result.x[0]
    quat = result.x[1:5]
    translation = result.x[5:8]
    rotation_matrix = pyq.Quaternion(quat).rotation_matrix
    print(f"Using {len(tmp_frame1_poses)} translations to optimized.")
    print("result from optimization:", result.x)
    print("Final Error:", result.fun)
    return scale, rotation_matrix, quat, translation


def _optimize_transform_3(
    tmp_frame1_poses: np.ndarray, tmp_frame2_poses: np.ndarray, sample_num=1000
):
    """
    Solve by Sample transformation. TODO: 可能有自旋未处理
    """

    def normalize_transform(transform):
        scale = np.linalg.det(transform[:3, :3]) ** (1 / 3)
        rotation = transform[:3, :3] / scale
        translation = transform[:3, 3]
        res = np.eye(4)
        res[:3, :3] = rotation
        res[:3, 3] = translation
        return res

    def error_function(params, frame1_poses, frame2_poses):
        scale = params[0]
        quat = params[1:5]  
        quat = quat / np.linalg.norm(quat)
        translation = params[5:8]  
        quat_scale_last = np.array([quat[1], quat[2], quat[3], quat[0]])
        rotation = R.from_quat(quat_scale_last)  # x y z w Here

        transformation = np.eye(4)
        transformation[:3, :3] = scale * rotation.as_matrix()
        transformation[:3, 3] = translation

        frame1_poses_after_transform = np.array(
            [
                normalize_transform(transformation @ frame1_pose)
                for frame1_pose in frame1_poses
            ]
        )
        normalized_frame2_poses = np.array(
            [normalize_transform(frame2_pose) for frame2_pose in frame2_poses]
        )
        error = np.sum((frame1_poses_after_transform - normalized_frame2_poses) ** 2)
        return error

    def solve_with_2_transforms(frame1_poses, frame2_poses):
        """
        A batch-processing solve_with_2_transforms function, where the 0th dimension is the batch dimension.

        :param frame1_poses: Tensor of shape (B, 2, 4, 4), where B is batch size, 2 is two poses, 4x4 is transform matrix
        :param frame2_poses: Tensor of shape (B, 2, 4, 4), same as above
        :return: scale, rotation_matrix, quaternion, translation 
        """
        frame1_pose_1 = frame1_poses[:, 0]
        frame1_pose_2 = frame1_poses[:, 1]
        frame2_pose_1 = frame2_poses[:, 0]
        frame2_pose_2 = frame2_poses[:, 1]

        scale = torch.norm(
            frame2_pose_1[:, :3, 3] - frame2_pose_2[:, :3, 3], dim=1
        ) / torch.norm(frame1_pose_1[:, :3, 3] - frame1_pose_2[:, :3, 3], dim=1)

        rotation_matrix_1 = frame2_pose_1[:, :3, :3] @ torch.inverse(
            frame1_pose_1[:, :3, :3]
        )
        rotation_matrix_2 = frame2_pose_2[:, :3, :3] @ torch.inverse(
            frame1_pose_2[:, :3, :3]
        )
        rotation_matrix = (rotation_matrix_1 + rotation_matrix_2) / 2
        translation_1 = frame2_pose_1[:, :3, 3] - scale.unsqueeze(1) * (
            rotation_matrix @ frame1_pose_1[:, :3, 3].unsqueeze(2)
        ).squeeze(2)
        translation_2 = frame2_pose_2[:, :3, 3] - scale.unsqueeze(1) * (
            rotation_matrix @ frame1_pose_2[:, :3, 3].unsqueeze(2)
        ).squeeze(2)
        rotation_diff = torch.norm(rotation_matrix_1 - rotation_matrix_2, dim=(1, 2))
        _, indices = torch.topk(
            rotation_diff, int(0.9 * rotation_diff.size(0)), largest=False
        )
        rotation_matrix = (rotation_matrix_1[indices] + rotation_matrix_2[indices]) / 2
        translation_diff = torch.norm(translation_1 - translation_2, dim=1)
        _, indices = torch.topk(
            translation_diff, int(0.9 * translation_diff.size(0)), largest=False
        )
        translation = (translation_1[indices] + translation_2[indices]) / 2

        quat = rotation_matrix_to_quaternion(rotation_matrix)

        scale = scale.sort()[0][int(0.05 * scale.size(0)) : -int(0.05 * scale.size(0))]

        return scale, rotation_matrix, quat, translation

    def rotation_matrix_to_quaternion(matrix):
        """
        Convert rotation matrix to quaternion, assuming matrix shape is (B, 3, 3)
        :param matrix: Rotation matrix tensor 
        :return: Quaternion tensor of shape (B, 4)
        """
        m = matrix
        B = m.shape[0]
        quat = torch.zeros((B, 4), dtype=m.dtype, device=m.device)

        quat[:, 0] = torch.sqrt(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]) / 2
        quat[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) / (4 * quat[:, 0])
        quat[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) / (4 * quat[:, 0])
        quat[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) / (4 * quat[:, 0])

        return quat

    def generate_unique_pairs(numbers, num_pairs, extra_factor=1.5):
        """
        Generate num_pairs pairs of non-repeating numbers from the given list,
        where the two numbers in each pair are different, but pairs can share numbers.

        :param numbers: a list or tensor of numbers
        :param num_pairs: number of pairs to generate
        :param extra_factor: factor for generating extra data, default 1.5
        :return: tensor containing num_pairs number pairs
        """
        N = len(numbers)  # size of number pool
        numbers = torch.tensor(numbers)  # convert input to tensor 

        # generate more pairs at once
        total_pairs = int(num_pairs * extra_factor) 
        indices = torch.randint(0, N, (total_pairs, 2))

        # filter out pairs where the two numbers are different
        mask = indices[:, 0] != indices[:, 1]
        valid_pairs = indices[mask]

        # if not enough valid pairs, generate more
        while valid_pairs.size(0) < num_pairs:
            new_indices = torch.randint(0, N, (total_pairs, 2))
            new_mask = new_indices[:, 0] != new_indices[:, 1]
            valid_pairs = torch.cat((valid_pairs, new_indices[new_mask]), dim=0)

        # select only num_pairs pairs
        final_indices = valid_pairs[:num_pairs]

        # use indices to select final number pairs from original list
        final_pairs = numbers[final_indices]

        return final_pairs

    # Solve by Sample in batch
    sample_idx = generate_unique_pairs(range(len(tmp_frame1_poses)), sample_num)
    sample_frame1_poses = torch.tensor(tmp_frame1_poses)[sample_idx]
    sample_frame2_poses = torch.tensor(tmp_frame2_poses)[sample_idx]
    scale, rotation_matrix, quat, translation = solve_with_2_transforms(
        sample_frame1_poses, sample_frame2_poses
    )
    print("Std of Scale:", scale.std())
    print("Std of Rotation:", rotation_matrix.std(dim=0))
    print("Std of Translation:", translation.std(dim=0))
    rotation_matrix = rotation_matrix.mean(dim=0)
    quat = quat.mean(dim=0)
    translation = translation.mean(dim=0)
    scale = scale.mean()
    print("Final Scale:", scale)
    print("Final Rotation:", rotation_matrix)
    print("Final Quat:", quat)
    print("Final Translation:", translation)
    print(
        f"Error of {len(tmp_frame1_poses)}:",
        error_function(
            [scale, *(quat.tolist()), *(translation.tolist())],
            tmp_frame1_poses,
            tmp_frame2_poses,
        ),
    )

    return scale, rotation_matrix, quat, translation


def get_transform_between_two_frames2(
    frame1_poses,
    frame2_poses,
    transform_type=True,
    img_names=None,
    reg=None,
    sample_num=1000,
):
    """
    Get transform from frame1 to frame2
    Parameters:
    frame1_poses: list of np.ndarray, shape=(4, 4)
    frame2_poses: list of np.ndarray, shape=(4, 4)
    Return:
    Scale, Rotation, Translation, frame1_poses_in_frame2
    Rotation is represented by a quaternion [w, x, y, z]
    """
    if isinstance(frame1_poses, FrameTools) and isinstance(frame2_poses, FrameTools):
        frame1_names = frame1_poses.frame_names
        frame2_names = frame2_poses.frame_names
        common_names = list(set(frame1_names) & set(frame2_names))
        frame1_poses_lst = [
            frame1_poses.get_frame_transform(frame_name) for frame_name in common_names
        ]
        frame2_poses_lst = [
            frame2_poses.get_frame_transform(frame_name) for frame_name in common_names
        ]
        return get_transform_between_two_frames2(
            frame1_poses_lst, frame2_poses_lst, transform_type
        )
    else:
        reg_match_idx = 0
        tmp_frame1_poses = []
        tmp_frame2_poses = []
        for i, (frame1_pose, frame2_pose) in enumerate(zip(frame1_poses, frame2_poses)):
            if reg is not None:
                assert isinstance(reg, str), "reg should be a string"
                assert img_names is not None, "img_names should be provided"
                import re

                if re.match(reg, img_names[i]) is None:
                    print(f"Reg {reg} not match {img_names[i]}")
                    continue
                reg_match_idx += 1
            if frame1_pose is None or frame2_pose is None:
                continue
            tmp_frame1_poses.append(frame1_pose)
            tmp_frame2_poses.append(frame2_pose)

        tmp_frame1_poses = np.array(tmp_frame1_poses)
        tmp_frame2_poses = np.array(tmp_frame2_poses)

        if reg is not None:
            print(f"Reg match {reg_match_idx} images.")
        scale, rotation_matrix, quat, translation = _optimize_transform_3(
            tmp_frame1_poses, tmp_frame2_poses, sample_num=sample_num
        )
        new_frame1_poses = []
        for frame1_pose in frame1_poses:
            original_translation = frame1_pose[:3, 3]
            original_rotation = frame1_pose[:3, :3]
            new_translation = (
                scale * rotation_matrix @ original_translation + translation
            )
            new_orientation = rotation_matrix @ original_rotation
            new_pose = np.eye(4)
            new_pose[:3, :3] = new_orientation
            new_pose[:3, 3] = new_translation
            new_frame1_poses.append(new_pose)

        if transform_type:
            transform = np.eye(4)
            transform[:3, :3] = scale * rotation_matrix
            transform[:3, 3] = translation
            return transform, new_frame1_poses
        else:
            return scale, quat, translation, new_frame1_poses
