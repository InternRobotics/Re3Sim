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

try:
    from datasets.episodic_dataset import EpisodicDataset
    from datasets.real_robot_dataset import RealRobotDataset
    from datasets.real_world_episodic_dataset import RealWorldEpisodicDataset
    from datasets.episodic_dataset_jpg import EpisodicDatasetJpg
    from datasets.episodic_dataset_jpg_continue import EpisodicDatasetJpgContinue
    from datasets.episodic_dataset_lmdb import EpisodicDatasetLmdb
    from datasets.episodic_dataset_lmdb_expand_last_step import (
        EpisodicDatasetLmdbExpandLastStep,
    )
    from datasets.real_world_episodic_dataset_lmdb import RealWorldEpisodicLmdbDataset

    # from datasets.mix_dataset import MixDataset
    from datasets.datasets_utils import *
    from torch_utils import distributed as dist
except Exception as e:
    import traceback

    traceback.print_exc()
    from .datasets.episodic_dataset import EpisodicDataset
    from .datasets.real_robot_dataset import RealRobotDataset
    from .datasets.real_world_episodic_dataset import RealWorldEpisodicDataset
    from .datasets.episodic_dataset_jpg import EpisodicDatasetJpg
    from .datasets.episodic_dataset_jpg_continue import EpisodicDatasetJpgContinue
    from .datasets.episodic_dataset_lmdb import EpisodicDatasetLmdb
    from .datasets.episodic_dataset_lmdb_expand_last_step import (
        EpisodicDatasetLmdbExpandLastStep,
    )
    from .datasets.real_world_episodic_dataset_lmdb import RealWorldEpisodicLmdbDataset
    from .datasets.datasets_utils import *

    # from .datasets.mix_dataset import MixDataset
    from .torch_utils import distributed as dist
try:
    import albumentations as A
except:
    pass
from torch.utils.data.distributed import DistributedSampler

e = IPython.embed


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_norm_stats(dataset_path_list, relative_control=False):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    error_occurred = False
    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                qvel = root["/observations/qvel"][()]
                if "/base_action" in root:
                    base_action = root["/base_action"][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root["/action"][()], base_action], axis=-1)
                else:
                    action = root["/action"][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            dist.print0(f"Error loading {dataset_path} in get_norm_stats")
            dist.print0(e)
            error_occurred = True
            # quit()
        if not error_occurred:
            all_qpos_data.append(torch.from_numpy(qpos))
            if relative_control:
                all_action_data.append(
                    torch.from_numpy(action)
                    - torch.from_numpy(
                        np.concatenate([qpos, np.zeros((qpos.shape[0], 2))], axis=-1)
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


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    print(dataset_dir)
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            if "features" in filename:
                continue
            if skip_mirrored_data and "mirror" in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    dist.print0(f"Found {len(hdf5_files)} hdf5 files")
    return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = (
        np.array(sample_weights) / np.sum(sample_weights)
        if sample_weights is not None
        else None
    )
    sum_dataset_len_l = np.cumsum(
        [0] + [np.sum(episode_len) for episode_len in episode_len_l]
    )
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(
                sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1]
            )
            batch.append(step_idx)
        yield batch


def get_augment_transform(augment_type, augment_prob):
    # Augment images
    val_transform = None
    if augment_type == 0:
        train_transform = None

    elif augment_type == 1:
        p = augment_prob
        train_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=p),  # Randomly adjust brightness and contrast
                A.RandomGamma(p=p),  # Randomly adjust gamma
            ]
        )

    elif augment_type == 2:
        train_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(var_limit=(10, 30), p=0.2),  # Add Gaussian noise
                A.RandomGamma(p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),  # Apply contrast limited adaptive histogram equalization
            ]
        )
    elif augment_type == 3:
        train_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(var_limit=(30, 70), p=0.4),  # Add Gaussian noise
                A.RandomGamma(p=0.4),
                A.CLAHE(clip_limit=3.0, p=0.4),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2
                ),  # Randomly adjust color
            ]
        )
    elif augment_type == 4:
        p = augment_prob
        train_transform = A.Compose(
            [
                # p ~ 0.5
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=p),
                        A.MotionBlur(p=p),
                    ]
                ),
            ]
        )
    elif augment_type == 5:
        p = augment_prob
        train_transform = A.Compose(
            [
                # p ~ 0.5
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=p),
                        A.MotionBlur(p=p),
                        A.Defocus(p=p),
                    ]
                ),
            ]
        )
    elif augment_type == 6:
        p = augment_prob
        train_transform = A.Compose(
            [
                # p ~ 0.5
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=p),
                        A.MotionBlur(p=p),
                        A.Defocus(p=p),
                        A.ColorJitter(p=p),
                    ]
                ),
            ]
        )
    elif augment_type == 10:
        p = augment_prob
        train_transform = A.Compose(
            [
                # p ~ 0.5
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=p),
                        A.MotionBlur(p=p),
                        A.Defocus(p=p),
                        A.ColorJitter(p=p),
                        A.GaussNoise(p=p),
                    ]
                ),
            ]
        )

    return train_transform, val_transform


def load_data(
    dataset_dir_l,
    name_filter,
    camera_names,
    batch_size_train,
    batch_size_val,
    chunk_size,
    skip_mirrored_data=False,
    load_pretrain=False,
    stats_dir_l=None,
    sample_weights=None,
    train_ratio=0.99,
    relative_control=False,
    dataset_cls="EpisodicDataset",
    augment_type=0,
    augment_prob=0.5,
    original_dataset_root_path=None,
    return_dataset=False,
):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    train_transform, val_transform = get_augment_transform(augment_type, augment_prob)
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [
        find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l
    ]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [
        len(dataset_path_list) for dataset_path_list in dataset_path_list_list
    ]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # Obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[: int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0) :]
    train_episode_ids_l = [train_episode_ids_0] + [
        np.arange(num_episodes) + num_episodes_cumsum[idx]
        for idx, num_episodes in enumerate(num_episodes_l[1:])
    ]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    dist.print0(
        f"\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n"
    )

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     dist.print0('Loaded pretrain dataset stats')
    if (
        dataset_cls == "EpisodicDataset"
        or dataset_cls == "EpisodicDatasetJpg"
        or dataset_cls == "EpisodicDatasetJpgContinue"
    ):
        _, all_episode_len = get_norm_stats(dataset_path_list, relative_control)
    elif dataset_cls == "RealRobotDataset":
        _, all_episode_len = RealRobotDataset.get_norm_stats(
            dataset_path_list, relative_control
        )
    elif dataset_cls == "RealWorldEpisodicDataset":
        _, all_episode_len = RealWorldEpisodicDataset.get_norm_stats(dataset_path_list)
    elif dataset_cls == "RealWorldEpisodicLmdbDataset":
        _, all_episode_len = RealWorldEpisodicLmdbDataset.get_norm_stats(
            dataset_path_list
        )
    elif dataset_cls == "EpisodicDatasetLmdb":
        _, all_episode_len = EpisodicDatasetLmdb.get_norm_stats(dataset_path_list)
    elif dataset_cls == "EpisodicDatasetLmdbExpandLastStep":
        _, all_episode_len = EpisodicDatasetLmdbExpandLastStep.get_norm_stats(
            dataset_path_list
        )
    else:
        raise ValueError(f"Unknown dataset class: {dataset_cls}")
    train_episode_len_l = [
        [all_episode_len[i] for i in train_episode_ids]
        for train_episode_ids in train_episode_ids_l
    ]
    val_episode_len_l = [
        [all_episode_len[i] for i in val_episode_ids]
        for val_episode_ids in val_episode_ids_l
    ]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]

    if (
        dataset_cls == "EpisodicDataset"
        or dataset_cls == "EpisodicDatasetJpg"
        or dataset_cls == "EpisodicDatasetJpgContinue"
    ):
        norm_stats, _ = get_norm_stats(
            flatten_list(
                [
                    find_all_hdf5(stats_dir, skip_mirrored_data)
                    for stats_dir in stats_dir_l
                ]
            ),
            relative_control,
        )
    elif dataset_cls == "RealRobotDataset":
        norm_stats, _ = RealRobotDataset.get_norm_stats(
            flatten_list(
                [
                    find_all_hdf5(stats_dir, skip_mirrored_data)
                    for stats_dir in stats_dir_l
                ]
            ),
            relative_control,
        )
    elif dataset_cls == "RealWorldEpisodicDataset":
        norm_stats, _ = RealWorldEpisodicDataset.get_norm_stats(
            flatten_list(
                [
                    find_all_hdf5(stats_dir, skip_mirrored_data)
                    for stats_dir in stats_dir_l
                ]
            )
        )
    elif dataset_cls == "RealWorldEpisodicLmdbDataset":
        norm_stats, _ = RealWorldEpisodicLmdbDataset.get_norm_stats(
            flatten_list(
                [
                    find_all_hdf5(stats_dir, skip_mirrored_data)
                    for stats_dir in stats_dir_l
                ]
            )
        )
    elif dataset_cls == "EpisodicDatasetLmdb":
        norm_stats, _ = EpisodicDatasetLmdb.get_norm_stats(
            flatten_list(
                [
                    find_all_hdf5(stats_dir, skip_mirrored_data)
                    for stats_dir in stats_dir_l
                ]
            )
        )
    elif dataset_cls == "EpisodicDatasetLmdbExpandLastStep":
        norm_stats, _ = EpisodicDatasetLmdbExpandLastStep.get_norm_stats(
            flatten_list(
                [
                    find_all_hdf5(stats_dir, skip_mirrored_data)
                    for stats_dir in stats_dir_l
                ]
            )
        )
    dist.print0(f"Norm stats from: {stats_dir_l}")

    if dataset_cls == "EpisodicDataset":
        train_dataset = EpisodicDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = EpisodicDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "EpisodicDatasetJpg":
        train_dataset = EpisodicDatasetJpg(
            original_dataset_root_path,
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = EpisodicDatasetJpg(
            original_dataset_root_path,
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "EpisodicDatasetJpgContinue":
        train_dataset = EpisodicDatasetJpgContinue(
            original_dataset_root_path,
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = EpisodicDatasetJpgContinue(
            original_dataset_root_path,
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "RealRobotDataset":
        train_dataset = RealRobotDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = RealRobotDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "RealWorldEpisodicDataset":
        train_dataset = RealWorldEpisodicDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = RealWorldEpisodicDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "RealWorldEpisodicLmdbDataset":
        train_dataset = RealWorldEpisodicLmdbDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = RealWorldEpisodicLmdbDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "EpisodicDatasetLmdb":
        train_dataset = EpisodicDatasetLmdb(
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = EpisodicDatasetLmdb(
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "EpisodicDatasetLmdbExpandLastStep":
        train_dataset = EpisodicDatasetLmdbExpandLastStep(
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = EpisodicDatasetLmdbExpandLastStep(
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    elif dataset_cls == "MixDataset":
        train_dataset = MixDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            train_episode_ids,
            train_episode_len,
            chunk_size,
            relative_control,
            train_transform,
        )
        val_dataset = MixDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            val_episode_ids,
            val_episode_len,
            chunk_size,
            relative_control,
            val_transform,
        )
    else:
        raise ValueError(f"Unknown dataset class: {dataset_cls}")
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_num_workers = 6
    val_num_workers = 1
    prefetch_factor = 4
    dist.print0(
        f"Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}"
    )
    dist.print0(f"The length of train_dataset is {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=train_num_workers,
        prefetch_factor=prefetch_factor,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=val_num_workers,
        prefetch_factor=prefetch_factor,
    )

    if return_dataset:
        return (
            train_dataloader,
            val_dataloader,
            norm_stats,
            train_dataset.is_sim,
            train_dataset,
        )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0  # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
