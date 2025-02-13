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
import random
from .datasets_utils import preprocess_base_action
from pathlib import Path
import time
import json

try:
    import av
except:
    print("av not installed")
try:
    import torch_utils.distributed as dist
except:
    from src.act_plus_plus.torch_utils import distributed as dist
from .mp4_manager import FastVideoReader, VideoToShm, ShmVideoReader

e = IPython.embed


class EpisodicDatasetMp4(torch.utils.data.Dataset):
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
        background_names={"ground", "table", "BACKGROUND"},
    ):
        super(EpisodicDatasetMp4).__init__()
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
        # 如果是DDP训练，等待所有进程初始化完成
        if torch.distributed.is_initialized():
            self.video_transfer = VideoToShm(self.original_dataset_root_path)
            if dist.get_rank() == 0:
                self.video_transfer.copy_videos()
            else:
                self.video_transfer.get_meta_data()
            torch.distributed.barrier()
        else:
            self.video_transfer = VideoToShm(self.original_dataset_root_path)
            self.video_transfer.copy_videos()

    def initialize(self):
        """初始化视频管理器，确保在DDP训练时正确共享实例"""
        if not self._initialized:
            self._initialized = True
            # self.video_manager = ShmVideoReader(shm_path=self.video_transfer.shm_path, enable_gpu=False, cache_size=20)
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

    def get_video_frame(self, video_path: str, frame_index: int) -> np.ndarray:
        """
        获取视频的指定帧（与OpenCV行为一致）

        参数:
            video_path (str): 视频文件路径
            frame_index (int): 要获取的帧索引（从0开始）

        返回:
            np.ndarray: 返回指定帧的图像数组，格式为 RGB，形状为 (height, width, 3)
        """
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]

            # 获取视频总帧数
            total_frames = stream.frames

            # 检查帧索引是否有效
            if frame_index < 0 or frame_index >= total_frames:
                raise ValueError(f"帧索引 {frame_index} 超出范围 [0, {total_frames-1}]")

            # 计算时间基准和目标时间戳
            time_base = stream.time_base
            duration_per_frame = stream.duration / total_frames
            target_pts = int(frame_index * duration_per_frame)

            # 设置视频流参数
            stream.thread_type = "AUTO"

            # 精确定位到目标帧
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)

            # 读取目标帧
            for frame in container.decode(video=0):
                if frame.pts >= target_pts:
                    return frame.to_ndarray(format="rgb24")

        except FileNotFoundError:
            import traceback

            traceback.print_exc()
            print("\n\n")
            raise FileNotFoundError(f"找不到视频文件: {video_path}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("\n\n")
            raise Exception(f"读取视频帧时发生错误: {str(e)}")
        finally:
            if "container" in locals():
                container.close()

    def get_video_frame_cv2(self, video_path: str, frame_index: int) -> np.ndarray:
        """
        使用OpenCV获取视频的指定帧

        参数:
            video_path (str): 视频文件路径
            frame_index (int): 要获取的帧索引（从0开始）

        返回:
            np.ndarray: 返回指定帧的图像数组，格式为 RGB，形状为 (height, width, 3)

        异常:
            ValueError: 当帧索引超出范围或视频无法打开时抛出
            FileNotFoundError: 当视频文件不存在时抛出
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")

        cap = cv2.VideoCapture(video_path)
        try:
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 检查帧索引是否有效
            if frame_index < 0 or frame_index >= total_frames:
                raise ValueError(f"帧索引 {frame_index} 超出范围 [0, {total_frames-1}]")

            # 设置读取位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"无法读取第 {frame_index} 帧")

            # OpenCV 默认使用BGR格式，转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb

        finally:
            cap.release()

    def get_image(self, dataset_path, key, start_ts):
        video_abs_path = (
            Path(os.path.realpath(dataset_path)).parent.absolute() / f"{key}.mp4"
        )
        start_time = time.time()
        image = self.get_video_frame_cv2(
            self.video_transfer.video_paths_to_shm_path[str(video_abs_path)], start_ts
        )
        end_time = time.time()
        # dist.print0(f'get_image time: {end_time - start_time}')
        return image

    def __len__(self):
        return sum(self.episode_len)

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

    def get_random_images(
        self, render_image, sim_image, mask, picking_item_name, camera_name
    ):
        items_to_random = [item for item in self.items if item != picking_item_name]
        items_to_mask = self.random_choice(items_to_random)
        masks = [
            mask == self._name_to_id[camera_name][item]
            for item in items_to_mask
            if item in self._name_to_id[camera_name]
        ]
        masks += [mask == id for id in self.background_ids[camera_name]]
        masks = np.logical_or.reduce(np.array(masks), axis=0)[..., None]
        image = masks * render_image + (1 - masks) * sim_image
        return image

    def get_item_multi_item(self, start_ts, dataset_path):
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
                    if cam_name in root["observations/render_images"].keys():
                        # render_image = root[f'/observations/render_images/{cam_name}'][start_ts]
                        render_image = self.get_image(
                            dataset_path,
                            f"observations/render_images/{cam_name}",
                            start_ts,
                        )
                    else:
                        render_image = root[
                            f"/observations/fix_render_images/{cam_name}"
                        ]
                    sim_image = self.get_image(
                        (dataset_path), f"observations/sim_images/{cam_name}", start_ts
                    )
                    mask = root[f"/observations/mask/{cam_name}"][start_ts]
                    with open(
                        Path(os.path.realpath(dataset_path)).parent / "info.json", "r"
                    ) as f:
                        picking_item_name = json.load(f)["picking_item_name"]
                    mix_image = self.get_random_images(
                        render_image, sim_image, mask, picking_item_name, cam_name
                    )
                    image_dict[cam_name] = mix_image

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
        if not self._initialized:
            self.initialize()
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        dataset_origin_path = os.path.realpath(dataset_path)
        multi_item = False
        with h5py.File(dataset_path, "r") as root:
            if not (Path(dataset_origin_path).parent / "observations/images").exists():
                multi_item = True
                if not hasattr(self, "_name_to_id"):
                    with open(Path(dataset_origin_path).parent / "info.json", "r") as f:
                        data = json.load(f)
                        self.items = data["item_names"]
                        idToLabels = data["mask_idToLabels"]
                    self._init_name_to_id(idToLabels)
        if multi_item:
            return self.get_item_multi_item(start_ts, dataset_path)
        else:
            return self.get_item_single_item(start_ts, dataset_path)
