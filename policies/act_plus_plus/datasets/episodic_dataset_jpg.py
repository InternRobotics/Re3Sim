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
import IPython
import traceback
import random
from .datasets_utils import preprocess_base_action
from pathlib import Path
import time
import json

try:
    import torch_utils.distributed as dist
except:
    from src.act_plus_plus.torch_utils import distributed as dist
from .jpg_manager import JpgToShm

e = IPython.embed


class EpisodicDatasetJpg(torch.utils.data.Dataset):
    def __init__(
        self,
        original_dataset_root_path,
        dataset_path_list,
        camera_names,
        norm_stats,
        episode_ids,
        episode_len,
        chunk_size,
        relative_control,
        transform=None,
        all_data_in_mem=False,
        background_names={"ground", "table", "BACKGROUND"},
    ):
        super(EpisodicDatasetJpg).__init__()
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
        self.is_sim = False
        self.items = []
        self.background_names = background_names
        self.background_ids = {}
        self.original_dataset_root_path = original_dataset_root_path
        self._initialized = False
        self.multi_item = False
        self.mask_free_items = 0
        self.all_data_in_mem = all_data_in_mem
        if all_data_in_mem:
            # 如果是DDP训练，等待所有进程初始化完成
            if torch.distributed.is_initialized():
                self.video_transfer = JpgToShm(self.original_dataset_root_path)
                if dist.get_rank() == 0:
                    self.video_transfer.copy_videos()
                torch.distributed.barrier()
            else:
                self.video_transfer = JpgToShm(self.original_dataset_root_path)
                self.video_transfer.copy_videos()

    def initialize(self):
        """初始化视频管理器，确保在DDP训练时正确共享实例"""
        if not self._initialized:
            self._initialized = True
            self.__getitem__(0)  # initialize self.is_sim and self.transformations

    # def initialize(self):
    #     """初始化视频管理器，确保在DDP训练时正确共享实例"""
    #     if not self._initialized:
    #         self._initialized = True

    #         # 如果是DDP训练，等待所有进程初始化完成
    #         if torch.distributed.is_initialized():
    #             if dist.get_rank() == 0:
    #                 video_manager = FastVideoReader(self.original_dataset_root_path)
    #             torch.distributed.barrier()
    #             self.video_manager = video_manager
    #         else:
    #             self.video_manager = FastVideoReader(self.original_dataset_root_path)

    #         self.__getitem__(0)  # initialize self.is_sim and self.transformations

    # def get_image(self, dataset_path, key, start_ts):
    #     video_abs_path = Path(os.path.realpath(dataset_path)).parent.absolute() / f"{key}.mp4"
    #     video_rel_path = video_abs_path.relative_to(os.path.realpath(self.video_transfer.folder_path))
    #     return self.video_manager.get_frame(video_rel_path, start_ts)

    def get_image(self, dataset_path, key, start_ts):
        if self.all_data_in_mem:
            jpg_path = (
                Path(os.path.realpath(dataset_path)).parent.absolute()
                / f"{key}/{start_ts}.jpg"
            )
            image_path = os.path.join(self.video_transfer.get_shm_path(jpg_path))
            return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        else:
            jpg_path = (
                Path(os.path.realpath(dataset_path)).parent.absolute()
                / f"{key}/{start_ts}.jpg"
            )
            return cv2.cvtColor(cv2.imread(str(jpg_path)), cv2.COLOR_BGR2RGB)

    def get_image_png(self, dataset_path, key, start_ts):
        jpg_path = (
            Path(os.path.realpath(dataset_path)).parent.absolute()
            / f"{key}/{start_ts}.png"
        )
        return cv2.cvtColor(cv2.imread(str(jpg_path)), cv2.COLOR_BGR2RGB)

    def __len__(self):
        if hasattr(self, "multi_item") and self.multi_item:
            return sum(self.episode_len) * (2**self.mask_free_items)
        else:
            return sum(self.episode_len)

    def _locate_transition(self, index):
        assert sum(self.episode_len) == self.cumulative_len[-1]
        assert index < sum(self.episode_len) or self.multi_item
        if self.multi_item:
            mask_id = index // sum(self.episode_len)
            index = index % sum(self.episode_len)
        else:
            mask_id = 0
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts, mask_id

    def get_item_single_item(self, start_ts, dataset_path):
        try:
            with h5py.File(dataset_path, "r") as root:
                try:  # some legacy data does not have this attribute
                    is_sim = root.attrs["sim"]
                except:
                    is_sim = False
                compressed = root.attrs.get("compress", False)
                if "/base_action" in root:
                    base_action = root["/base_action"][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root["/action"][()], base_action], axis=-1)
                else:
                    action = root["/action"][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                qpos = root["/observations/qpos"][start_ts]
                qvel = root["/observations/qvel"][start_ts]
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = self.get_image(
                        dataset_path, f"observations/images/{cam_name}", start_ts
                    )

                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)

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

            try:
                if self.relative_control:
                    action_data = (
                        action_data - torch.cat([qpos_data, torch.zeros((2,))], axis=0)
                    ).view(action_data.dtype)
            except Exception as e:
                dist.print0(e)
            # normalize to mean 0 std 1
            action_data = (
                action_data - self.norm_stats["action_mean"]
            ) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
                "qpos_std"
            ]

        except Exception as e:
            dist.print0(f"Error loading {dataset_path} in __getitem__")
            traceback.print_exc()
            dist.print0(e)
            quit()

        # dist.print0(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, qpos_data, action_data, is_pad

    def random_choice(self, list):
        choice_num = random.randint(0, len(list))
        return random.sample(list, choice_num)

    def get_mix_images(
        self, render_image, sim_image, mask, picking_item_name, camera_name, mask_id
    ):
        items_to_random = [item for item in self.items if item != picking_item_name]
        items_to_mask = [
            items_to_random[i]
            for i in range(len(items_to_random))
            if mask_id & (1 << i)
        ]

        masks = [
            mask == self._name_to_id[camera_name][item]
            for item in items_to_mask
            if item in self._name_to_id[camera_name]
        ]
        masks += [mask == id for id in self.background_ids[camera_name]]
        masks = np.logical_or.reduce(np.array(masks), axis=0)
        image = masks * render_image + (1 - masks) * sim_image
        return image

    def get_item_multi_item(self, start_ts, dataset_path, mask_id):
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, "r") as root:
                try:  # some legacy data does not have this attribute
                    is_sim = root.attrs["sim"]
                except:
                    is_sim = False
                compressed = root.attrs.get("compress", False)
                if "/base_action" in root:
                    base_action = root["/base_action"][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root["/action"][()], base_action], axis=-1)
                else:
                    action = root["/action"][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                qpos = root["/observations/qpos"][start_ts]
                qvel = root["/observations/qvel"][start_ts]
                image_dict = dict()
                for cam_name in self.camera_names:
                    if cam_name in root["observations/fix_render_images"].keys():
                        render_image = root[
                            f"/observations/fix_render_images/{cam_name}"
                        ]
                    else:
                        render_image = self.get_image(
                            dataset_path,
                            f"observations/render_images/{cam_name}",
                            start_ts,
                        )

                    sim_image = self.get_image(
                        (dataset_path), f"observations/sim_images/{cam_name}", start_ts
                    )
                    mask = self.get_image_png(
                        dataset_path, f"observations/mask/{cam_name}", start_ts
                    )
                    with open(
                        Path(os.path.realpath(dataset_path)).parent / "info.json", "r"
                    ) as f:
                        picking_item_name = json.load(f)["picking_item_name"]
                    mix_image = self.get_mix_images(
                        render_image,
                        sim_image,
                        mask,
                        picking_item_name,
                        cam_name,
                        mask_id,
                    )
                    image_dict[cam_name] = np.uint8(mix_image)

                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)

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

            try:
                if self.relative_control:
                    action_data = (
                        action_data - torch.cat([qpos_data, torch.zeros((2,))], axis=0)
                    ).view(action_data.dtype)
            except Exception as e:
                dist.print0(e)
            # normalize to mean 0 std 1
            action_data = (
                action_data - self.norm_stats["action_mean"]
            ) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
                "qpos_std"
            ]

        except Exception as e:
            dist.print0(f"Error loading {dataset_path} in __getitem__")
            traceback.print_exc()
            dist.print0(e)
            quit()

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, qpos_data, action_data, is_pad

    def _init_name_to_id(self, idToLabels):
        self._name_to_id = {}
        background_id = {}
        for camera_name in self.camera_names:
            background_id[camera_name] = []
            self._name_to_id[camera_name] = {}
            for id, data in idToLabels[camera_name].items():
                name = data["class"]
                if name in self.background_names:
                    background_id[camera_name].append(int(id))
                elif name in self.items:
                    self._name_to_id[camera_name][name] = int(id)

        self.background_ids = background_id

    def __getitem__(self, index):
        episode_id, start_ts, mask_id = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        dataset_origin_path = os.path.realpath(dataset_path)
        with h5py.File(dataset_path, "r") as root:
            if not os.path.exists(
                Path(dataset_origin_path).parent / "observations/images"
            ):
                self.multi_item = True
                if not hasattr(self, "_name_to_id"):
                    with open(Path(dataset_origin_path).parent / "info.json", "r") as f:
                        data = json.load(f)
                        self.items = data["item_names"]
                        self.mask_free_items = len(self.items) - 1
                        dist.print0(
                            f"The number of mask-free items: {self.mask_free_items}"
                        )
                        idToLabels = data["mask_idToLabels"]
                    self._init_name_to_id(idToLabels)
        if self.multi_item:
            return self.get_item_multi_item(start_ts, dataset_path, mask_id)
        else:
            return self.get_item_single_item(start_ts, dataset_path)
