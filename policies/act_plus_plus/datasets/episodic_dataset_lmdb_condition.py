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
import lmdb
import av

try:
    import torch_utils.distributed as dist
except:
    from src.act_plus_plus.torch_utils import distributed as dist
from .jpg_manager import JpgToShm

e = IPython.embed


from transformers import AutoTokenizer, CLIPTextModel

pretrained_dir = os.path.expanduser("~/.cache/torch/hub/clip-vit-base-patch32")
lang_encoder = CLIPTextModel.from_pretrained(pretrained_dir)
tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)


class EpisodicDatasetLmdbCondition(torch.utils.data.Dataset):
    @staticmethod
    def get_norm_stats(dataset_path_list, relative_control=False):
        all_qpos_data = []
        all_action_data = []
        all_episode_len = []
        error_occurred = False
        for dataset_path in dataset_path_list:
            try:
                with lmdb.open(
                    str(Path(os.path.realpath(dataset_path)).parent / "lmdb"),
                    lock=False,
                ) as lmdb_env:
                    with lmdb_env.begin(write=False) as txn:
                        qpos = np.array(
                            pickle.loads(txn.get("observations/qpos".encode("utf-8")))
                        )
                        # qvel = np.array(pickle.loads(txn.get("observations/qvel".encode("utf-8"))))
                        # 读取action数据
                        action = np.array(
                            pickle.loads(txn.get("action".encode("utf-8")))
                        )
                        dummy_base_action = np.zeros([action.shape[0], 2])
                        action = np.concatenate([action, dummy_base_action], axis=-1)
            except Exception as e:
                dist.print0(
                    f'Error loading {str(Path(os.path.realpath(dataset_path)).parent / "lmdb")} in get_norm_stats'
                )
                dist.print0(e)
                traceback.print_exc()
                error_occurred = True
                # quit()
            if not error_occurred:
                all_qpos_data.append(torch.tensor(qpos))
                if relative_control:
                    all_action_data.append(
                        torch.from_numpy(action)
                        - torch.from_numpy(
                            np.concatenate(
                                [qpos, np.zeros((qpos.shape[0], 2))], axis=-1
                            )
                        )
                    )
                else:
                    all_action_data.append(torch.from_numpy(action))
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
        all_data_in_mem=False,
        background_names={"ground", "table", "BACKGROUND"},
    ):
        super(EpisodicDatasetLmdbCondition).__init__()
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
        self._initialized = False
        self.multi_item = False
        self.mask_free_items = 0
        self.all_data_in_mem = all_data_in_mem
        self.lang_emb_cache = {}

    def initialize(self):
        """初始化视频管理器，确保在DDP训练时正确共享实例"""
        if not self._initialized:
            self._initialized = True
            self.__getitem__(0)  # initialize self.is_sim and self.transformations

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert sum(self.episode_len) == self.cumulative_len[-1]
        assert index < sum(self.episode_len) or self.multi_item
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
        breakpoint()
        try:
            with lmdb.open(
                str(Path(os.path.realpath(dataset_path)).parent / "lmdb"), lock=False
            ) as lmdb_env:
                with lmdb_env.begin(write=False) as txn:
                    is_sim = False
                    compressed = False
                    action = pickle.loads(txn.get("action".encode("utf-8")))
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                    original_action_shape = action.shape
                    episode_len = original_action_shape[0]
                    # get observation at start_ts only
                    qpos = pickle.loads(txn.get("observations/qpos".encode("utf-8")))[
                        start_ts
                    ]
                    # qvel = pickle.loads(txn.get("observations/qvel".encode("utf-8")))[start_ts]
                    image_dict = dict()
                    for cam_name in self.camera_names:
                        image_dict[cam_name] = cv2.imdecode(
                            pickle.loads(
                                txn.get(
                                    f"observations/images/{cam_name}/{start_ts}".encode(
                                        "utf-8"
                                    )
                                )
                            ),
                            cv2.IMREAD_COLOR,
                        )

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
        masks = np.logical_or.reduce(np.array(masks), axis=0)[..., np.newaxis]
        image = masks * render_image + (1 - masks) * sim_image
        return image

    def get_item_multi_item(self, start_ts, dataset_path, mask_id):
        class_name = "object"
        if "orange_bottle" in dataset_path:
            class_name = "orange bottle"
        elif "corn" in dataset_path:
            class_name = "corn"
        elif "eggplant" in dataset_path:
            class_name = "eggplant"
        elif "green" in dataset_path:
            class_name = "cucumber"
        elif "ball" in dataset_path:
            class_name = "ball"
        lang = f"pick the {class_name} and place it in the basket"
        if lang in self.lang_emb_cache:
            lang_emb = self.lang_emb_cache[lang]
        else:
            inputs = tokenizer([lang], padding=True, return_tensors="pt")
            outputs = lang_encoder(**inputs)
            pooled_output = outputs.pooler_output.detach().cpu()
            lang_emb = self.lang_emb_cache[lang] = pooled_output[0]
            # breakpoint()
        try:
            lmdb_path = str(
                Path(os.path.realpath(dataset_path)).parent / "lmdb"
            )  # dataset_path直接就是lmdb路径
            with lmdb.open(lmdb_path, lock=False) as lmdb_env:
                with lmdb_env.begin(write=False) as txn:
                    # 读取action数据
                    action = np.array(pickle.loads(txn.get("action".encode("utf-8"))))
                    base_action_bytes = txn.get("base_action".encode("utf-8"))
                    if base_action_bytes is not None:
                        base_action = pickle.loads(base_action_bytes)
                        base_action = preprocess_base_action(base_action)
                        action = np.concatenate([action, base_action], axis=-1)
                    else:
                        dummy_base_action = np.zeros([action.shape[0], 2])
                        action = np.concatenate([action, dummy_base_action], axis=-1)

                    # 读取状态数据
                    qpos = pickle.loads(txn.get(f"observations/qpos".encode("utf-8")))[
                        start_ts
                    ]
                    # qvel = pickle.loads(txn.get(f"observations/qvel".encode("utf-8")))[start_ts]

                    original_action_shape = action.shape
                    episode_len = original_action_shape[0]

                    # 读取图像数据
                    image_dict = dict()
                    mask_dict = dict()
                    for cam_name in self.camera_names:
                        # 尝试读取固定渲染图像
                        fix_render_key = (
                            f"observations/fix_render_images/{cam_name}".encode("utf-8")
                        )
                        render_image_bytes = txn.get(fix_render_key)

                        # 读取sim图像和mask
                        sim_key = (
                            f"observations/sim_images/{cam_name}/{start_ts}".encode(
                                "utf-8"
                            )
                        )
                        mask_key = f"observations/mask/{cam_name}/{start_ts}".encode(
                            "utf-8"
                        )
                        sim_image = cv2.imdecode(
                            pickle.loads(txn.get(sim_key)), cv2.IMREAD_COLOR
                        )
                        mask = cv2.imdecode(
                            pickle.loads(txn.get(mask_key)), cv2.IMREAD_GRAYSCALE
                        )

                        if render_image_bytes is not None:
                            # render_image = np.array(pickle.loads(render_image_bytes)).reshape(np.array(sim_image).shape)
                            render_image = np.array(pickle.loads(render_image_bytes))
                            if render_image.shape[0] >= 2:
                                render_image = render_image[0]
                            else:
                                render_image = render_image.reshape(
                                    np.array(sim_image).shape
                                )
                        else:
                            # 读取普通渲染图像
                            render_key = f"observations/render_images/{cam_name}/{start_ts}".encode(
                                "utf-8"
                            )
                            render_image = cv2.imdecode(
                                pickle.loads(txn.get(render_key)), cv2.IMREAD_COLOR
                            )

                        # 读取picking item信息
                        with open(
                            Path(os.path.realpath(dataset_path)).parent / "info.json",
                            "r",
                        ) as f:
                            picking_item_name = json.load(f).get(
                                "picking_item_name", None
                            )

                        target_mask = np.zeros_like(mask)
                        if picking_item_name in self._name_to_id[cam_name]:
                            target_mask[
                                mask == self._name_to_id[cam_name][picking_item_name]
                            ] = 255

                        mix_image = self.get_mix_images(
                            render_image,
                            sim_image,
                            mask,
                            picking_item_name,
                            cam_name,
                            mask_id,
                        )
                        image_dict[cam_name] = np.uint8(mix_image)
                        mask_dict[cam_name] = np.expand_dims(
                            np.uint8(target_mask), axis=-1
                        )

                # get all actions after and including start_ts
                if self.is_sim:
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
            all_cam_masks = []
            for cam_name in self.camera_names:
                # augmentation
                if self.transform is not None:
                    tmp_img = image_dict[cam_name]
                    tmp_img = self.transform(image=tmp_img)["image"]
                    all_cam_images.append(tmp_img)
                else:
                    all_cam_images.append(image_dict[cam_name])
                all_cam_masks.append(mask_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)
            all_cam_masks = np.stack(all_cam_masks, axis=0)

            # all_cam_images = np.concatenate([all_cam_images, all_cam_masks], axis=-1)
            # breakpoint()

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
        return image_data, qpos_data, action_data, is_pad, lang_emb

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
        self.multi_item = True
        dataset_origin_path = os.path.realpath(dataset_path)
        if not os.path.exists(Path(dataset_origin_path).parent / "info.json"):
            self.multi_item = False
        else:
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
