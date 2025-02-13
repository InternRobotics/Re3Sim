import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
import IPython
import traceback
from .datasets_utils import *

e = IPython.embed


class RealRobotDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_norm_stats(dataset_path_list, relative_control=False):
        all_qpos_data = []
        all_action_data = []
        all_episode_len = []
        error_occurred = False
        for dataset_path in dataset_path_list:
            try:
                with h5py.File(dataset_path, "r") as root:
                    raw_qpos = root["/observations/qpos"][()]
                    raw_ee_pose = root["/observations/ee_pose"][()]
                    raw_ee_translation = raw_ee_pose[:, :3, 3]
                    raw_ee_euler = R.from_matrix(raw_ee_pose[:, :3, :3]).as_euler("xyz")
                    raw_qpos = np.concatenate(
                        [raw_qpos, raw_ee_translation, raw_ee_euler], axis=-1
                    )  # (qpos(9), ee_translation(3), ee_euler(3))
                    qpos = raw_qpos[:-1]  # remove last dim
                    # action
                    action = raw_qpos[1:, -8:]
                    gripper_action = root["/action"][:-1, -2:]
                    action[:, :2] = gripper_action
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate(
                        [action, dummy_base_action], axis=-1
                    )  # gripper_action(2) + panda_hand_pos(3) + panda_hand_euler(3) + dummy_base_action(2)[zero_padding]
                    action = torch.from_numpy(action)
                    if relative_control:
                        action[:, 2:8] = action[:, 2:8] - qpos[:, 9:15]

            except Exception as e:
                print(f"Error loading {dataset_path} in get_norm_stats")
                print(e)
                traceback.print_exc()
                error_occurred = True
                # quit()
            if not error_occurred:
                assert len(qpos) == len(action)
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(action)
                all_episode_len.append(len(qpos))
        if error_occurred:
            quit()
        all_qpos_data = torch.cat(all_qpos_data, dim=0)
        all_action_data = torch.cat(all_action_data, dim=0)

        # normalize action data
        action_mean = all_action_data.mean(dim=[0]).float()
        action_std = all_action_data.std(dim=[0]).float()
        action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

        # normalize qpos data
        qpos_mean = all_qpos_data.mean(dim=[0]).float()
        qpos_std = all_qpos_data.std(dim=[0]).float()
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

        action_min = all_action_data.min(dim=0).values.float()
        action_max = all_action_data.max(dim=0).values.float()

        eps = 0.0001
        stats = {
            "action_mean": action_mean.numpy(),
            "action_std": action_std.numpy(),
            "action_min": action_min.numpy() - eps,
            "action_max": action_max.numpy() + eps,
            "qpos_mean": qpos_mean.numpy(),
            "qpos_std": qpos_std.numpy(),
            "example_qpos": qpos,
        }

        return stats, all_episode_len

    def __init__(
        self,
        dataset_path_list,
        camera_names,
        norm_stats,
        episode_ids,
        episode_len,
        chunk_size,
        relative_control,
        transform=None,
    ):
        super(RealRobotDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.relative_control = relative_control
        self.augment_images = transform is not None
        self.transform = transform
        self.__getitem__(0)  # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, "r") as root:
                is_sim = True
                # get observation at start_ts only
                raw_qpos = root["/observations/qpos"][()]
                raw_ee_pose = root["/observations/ee_pose"][()]
                gripper_action = root["/action"][:-1, -2:]
                raw_ee_translation = raw_ee_pose[:, :3, 3]
                raw_ee_euler = R.from_matrix(raw_ee_pose[:, :3, :3]).as_euler("xyz")
                raw_qpos = np.concatenate(
                    [raw_qpos, raw_ee_translation, raw_ee_euler], axis=-1
                )  # (qpos(9), ee_translation(3), ee_euler(3))
                qpos = raw_qpos[start_ts]
                # action = raw_qpos[1:, -8:] - raw_qpos[:-1, -8:] if self.relative_control else raw_qpos[1:, -8:]
                action = raw_qpos[1:, -8:]
                if self.relative_control:
                    action[:, -6:] = action[:, -6:] - raw_qpos[:-1, -6:]
                action[:, :2] = gripper_action
                # action input
                # action = raw_qpos[1:]
                dummy_base_action = np.zeros([action.shape[0], 2])
                action = np.concatenate([action, dummy_base_action], axis=-1)
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # image input
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                        start_ts
                    ]

                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[
                        max(0, start_ts - 1) :
                    ]  # hack, to make timesteps more aligned
                    action_len = episode_len - max(
                        0, start_ts - 1
                    )  # hack, to make timesteps more aligned

            # self.is_sim = is_sim
            padded_action = np.zeros(
                (self.max_episode_len, original_action_shape[1]), dtype=np.float32
            )
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[: self.chunk_size]
            is_pad = is_pad[: self.chunk_size]

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                # augmentation
                if self.transform is not None:
                    tmp_img = image_dict[cam_name]
                    tmp_img = self.transform(image=tmp_img)["image"]
                    all_cam_images.append(tmp_img)
                else:
                    all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # channel last
            image_data = torch.einsum("k h w c -> k c h w", image_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0
            # normalize to mean 0 std 1
            action_data = (
                action_data - self.norm_stats["action_mean"]
            ) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
                "qpos_std"
            ]

        except Exception as e:
            print(f"Error loading {dataset_path} in __getitem__")
            raise e

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, qpos_data, action_data, is_pad
