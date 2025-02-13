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
    from datasets.episodic_dataset_lmdb import EpisodicDatasetLmdb
    from datasets.real_world_episodic_dataset_lmdb import RealWorldEpisodicLmdbDataset
except:
    from src.act_plus_plus.torch_utils import distributed as dist
from .jpg_manager import JpgToShm

e = IPython.embed


class EpisodicDatasetLmdb(torch.utils.data.Dataset):
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
        self._sim_dataset = EpisodicDatasetLmdb(
            dataset_path_list,
            camera_names,
            norm_stats,
            episode_ids,
            episode_len,
            chunk_size,
            relative_control,
            transform,
            all_data_in_mem,
            background_names,
        )
        self._real_dataset = RealWorldEpisodicLmdbDataset(
            dataset_path_list,
            camera_names,
            norm_stats,
            episode_ids,
            episode_len,
            chunk_size,
            relative_control,
            transform,
            all_data_in_mem,
            background_names,
        )
