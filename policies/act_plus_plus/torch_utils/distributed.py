# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
from . import training_stats
import subprocess

# ----------------------------------------------------------------------------


def init():
    if "SLURM_PROCID" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
    else:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    backend = "nccl"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    sync_device = torch.device("cuda") if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)


# ----------------------------------------------------------------------------


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# ----------------------------------------------------------------------------


def get_world_size():
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )


# ----------------------------------------------------------------------------


def should_stop():
    return False


# ----------------------------------------------------------------------------


def update_progress(cur, total):
    _ = cur, total


# ----------------------------------------------------------------------------


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------
