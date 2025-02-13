import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import re
import time
from torchvision import transforms
import random
from datetime import datetime
from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN

try:
    from torch_utils import distributed as dist
    from torch_utils.lr_scheduler import get_scheduler
    from utils import load_data  # data functions
    from utils import sample_box_pose, sample_insertion_pose  # robot functions
    from utils import (
        compute_dict_mean,
        set_seed,
        detach_dict,
        calibrate_linear_vel,
        postprocess_base_action,
    )  # helper functions
except Exception as e:
    import traceback

    traceback.print_exc()
    from .torch_utils import distributed as dist
    from .torch_utils.lr_scheduler import get_scheduler
    from .utils import load_data  # data functions
    from .utils import sample_box_pose, sample_insertion_pose  # robot functions
    from .utils import (
        compute_dict_mean,
        set_seed,
        detach_dict,
        calibrate_linear_vel,
        postprocess_base_action,
    )  # helper functions
from policy import ACTPolicy, ACTPolicyDinov2
from configclass import TrainingConfig
from visualize_episodes import save_videos
import albumentations as A

from detr.models.latent_model import Latent_Model_Transformer
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

# from sim_env import BOX_POSE
import logging
import IPython
from constants import SIM_TASK_CONFIGS

logging.basicConfig(level=logging.INFO)
import yaml

e = IPython.embed

SAVE_ALL_EPISODES = None


def make_policy(policy_config):
    if (
        policy_config["backbone"] == "dino_v2"
        or policy_config["backbone"][:7] == "dinov2_"
    ):
        policy = ACTPolicyDinov2(policy_config)
    else:
        policy = ACTPolicy(policy_config)
    return policy


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            continue
        print(
            "{:80s}{:20s}{:20s}{}".format(
                name,
                "(Trainable)" if p.requires_grad else "(Fixed)",
                "(Has grad):" if p.grad is not None else "(No grad backward):",
                list(p.shape),
            )
        )


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        dist.print0("rand crop resize is used!")
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[
            ...,
            int(original_size[0] * (1 - ratio) / 2) : int(
                original_size[0] * (1 + ratio) / 2
            ),
            int(original_size[1] * (1 - ratio) / 2) : int(
                original_size[1] * (1 + ratio) / 2
            ),
        ]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def train_bc(train_dataloader, val_dataloader, config, logger_writter):
    if SAVE_ALL_EPISODES:
        dist.print0("Saving all episodes")

    # 计算总steps数
    num_epochs = config["num_epochs"]
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch

    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_config = config["policy_config"]
    validate_every = config["validate_every"] // dist.get_world_size()
    save_every = config["save_every"] // dist.get_world_size()

    set_seed(seed)

    policy = make_policy(policy_config)
    if config["load_pretrain"]:
        loading_status = policy.model.module.load_state_dict(
            torch.load(
                os.path.join(
                    "/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all",
                    "policy_step_50000_seed_0.ckpt",
                )
            )
        )
        dist.print0(f"loaded! {loading_status}")
    if config["resume_ckpt_path"] is not None:
        loading_status = policy.model.module.load_state_dict(
            torch.load(config["resume_ckpt_path"])
        )
        dist.print0(
            f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}'
        )
    policy.cuda()
    optimizer = policy.configure_optimizers()
    lr_scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=total_steps,
    )

    min_val_loss = np.inf
    best_ckpt_info = None
    global_step = 0

    for epoch in range(num_epochs):
        dist.print0(f"Epoch {epoch}/{num_epochs}")

        # 训练阶段
        policy.train()

        # 为每个epoch创建进度条
        if dist.get_rank() == 0:
            pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}")

        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]
            loss.backward()

            # print_grad_status(policy.model)

            optimizer.step()
            lr_scheduler.step()

            if dist.get_rank() == 0:
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f'{optimizer.param_groups[0]["lr"]:.6f}',
                    }
                )

                # tensorboard记录
                logger_writter.add_scalar(
                    "lr", optimizer.param_groups[0]["lr"], global_step=global_step
                )
                for k, v in forward_dict.items():
                    logger_writter.add_scalar(k, v, global_step=global_step)

            # 验证阶段 - 基于step控制
            if global_step % validate_every == 0 and global_step > 0:
                dist.print0("Validating...")
                with torch.inference_mode():
                    policy.eval()
                    validation_dicts = []

                    for val_idx, val_data in enumerate(val_dataloader):
                        forward_dict = forward_pass(val_data, policy)
                        validation_dicts.append(forward_dict)
                        if val_idx > 50:
                            break

                    validation_summary = compute_dict_mean(validation_dicts)
                    val_loss = validation_summary["loss"]

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_ckpt_info = (
                            global_step,
                            min_val_loss,
                            deepcopy(policy.model.module.state_dict()),
                        )

                    for k in list(validation_summary.keys()):
                        validation_summary[f"val_{k}"] = validation_summary.pop(k)
                    if dist.get_rank() == 0:
                        for k, v in validation_summary.items():
                            logger_writter.add_scalar(k, v, global_step=global_step)

                    # 将验证结果合并到训练进度条中
                    if dist.get_rank() == 0:
                        pbar.set_postfix(
                            {
                                "loss": f"{loss.item():.4f}",
                                "val_loss": f"{val_loss:.4f}",
                                "lr": f'{optimizer.param_groups[0]["lr"]:.6f}',
                            }
                        )

                    dist.print0(f"Step {global_step}, Val loss: {val_loss:.5f}")
                    summary_string = ""
                    for k, v in validation_summary.items():
                        if k[0] != "_":
                            summary_string += f"{k}: {v.item():.3f} "
                    dist.print0(summary_string)
                    policy.train()

            # 保存检查点 - 基于step控制
            if dist.get_rank() == 0 and global_step % save_every == 0:
                ckpt_path = os.path.join(
                    ckpt_dir, f"policy_step_{global_step}_seed_{seed}.ckpt"
                )
                torch.save(policy.model.module.state_dict(), ckpt_path)

                if not SAVE_ALL_EPISODES:
                    ckpt_pattern = re.compile(r"^policy_step_(\d+)_seed_(\d+)\.ckpt$")
                    existing_ckpts = [
                        f for f in os.listdir(ckpt_dir) if ckpt_pattern.match(f)
                    ]
                    sorted_ckpts = sorted(
                        existing_ckpts,
                        key=lambda x: int(ckpt_pattern.match(x).group(1)),
                    )

                    if len(sorted_ckpts) > 5:
                        checkpoints_to_keep = sorted_ckpts[-5:]  # 保留最新的5个
                        for ckpt in sorted_ckpts[:-5]:
                            step_num = int(ckpt_pattern.match(ckpt).group(1))
                            if step_num % 20000 == 0:  # 每20000步保留一个
                                checkpoints_to_keep.append(ckpt)
                            elif step_num % 1000 == 0 and step_num > (
                                total_steps * 0.9
                            ):
                                checkpoints_to_keep.append(ckpt)
                            else:
                                os.remove(os.path.join(ckpt_dir, ckpt))

            global_step += 1

        if dist.get_rank() == 0:
            pbar.close()

    # 保存最终模型
    if dist.get_rank() == 0:
        ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
        torch.save(policy.model.module.state_dict(), ckpt_path)

        best_step, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        dist.print0(
            f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}"
        )

        return best_ckpt_info
    else:
        return None


def repeater(data_loader):
    epoch = 0
    while True:
        # 如果data_loader有sampler属性,并且sampler有set_epoch方法
        if hasattr(data_loader, "sampler") and hasattr(
            data_loader.sampler, "set_epoch"
        ):
            data_loader.sampler.set_epoch(epoch)

        for data in data_loader:
            yield data
        dist.print0(f"Epoch {epoch} done")
        epoch += 1


cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: TrainingConfig):
    # multi-gpu
    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init()

    hydra_log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    set_seed(1)
    global SAVE_ALL_EPISODES
    SAVE_ALL_EPISODES = cfg.save_all_episodes
    args = cfg["params"]
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    date_str = date_str + f"_{random.randint(100, 999)}"
    ckpt_dir = os.path.join(ckpt_dir, date_str)
    # save config
    if dist.get_rank() == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Data saved to {ckpt_dir}")
        with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
            yaml.dump(OmegaConf.to_yaml(cfg), f)
    args["ckpt_dir"] = ckpt_dir
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    eval_every = args["eval_every"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]
    relative_control = args.get("relative_control", False)
    dist.print0(f"Using relative control: {relative_control}")

    # get task parameters
    is_sim = True
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    # num_episodes = task_config['num_episodes']
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)
    train_ratio = task_config.get("train_ratio", 0.99)
    name_filter = task_config.get("name_filter", lambda n: True)
    original_dataset_root_path = task_config.get("original_dataset_root_path", None)

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = args["backbone"]
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    args["lr"] *= dist.get_world_size()
    policy_config = {
        "lr": args["lr"],
        "num_queries": args["chunk_size"],
        "kl_weight": args["kl_weight"],
        "hidden_dim": args["hidden_dim"],
        "dim_feedforward": args["dim_feedforward"],
        "lr_backbone": lr_backbone,
        "backbone": backbone,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "camera_names": camera_names,
        "action_dim": args["action_dim"] + 2,
        "state_dim": args["state_dim"],
        "no_encoder": args["no_encoder"],
        "isaac_franka": True,
        "qpos_noise_std": args["qpos_noise_std"],
        "freeze_backbone": args["freeze_backbone"],
    }

    actuator_config = {
        "actuator_network_dir": args["actuator_network_dir"],
        "history_len": args["history_len"],
        "future_len": args["future_len"],
        "prediction_len": args["prediction_len"],
    }

    config = {
        "num_epochs": args["num_epochs"],
        "eval_every": eval_every,
        "validate_every": validate_every,
        "save_every": save_every,
        "ckpt_dir": ckpt_dir,
        "resume_ckpt_path": resume_ckpt_path,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "load_pretrain": args["load_pretrain"],
        "actuator_config": actuator_config,
        "relative_control": relative_control,
        "dataset_cls": args["dataset_cls"],
        "lr_scheduler": args["lr_scheduler"],
        "num_warmup_steps": args["num_warmup_steps"],
    }

    writer = None
    if dist.get_rank() == 0:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        config_path = os.path.join(ckpt_dir, "config.pkl")
        expr_name = ckpt_dir.split("/")[-1]
        writer = SummaryWriter(os.path.join(hydra_log_dir, "tensorboard"))
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

    augment_type = args["augment_type"]
    augment_prob = args["augment_prob"]

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        name_filter,
        camera_names,
        batch_size_train,
        batch_size_val,
        args["chunk_size"],
        args["skip_mirrored_data"],
        config["load_pretrain"],
        stats_dir_l=stats_dir,
        sample_weights=sample_weights,
        train_ratio=train_ratio,
        relative_control=relative_control,
        dataset_cls=args["dataset_cls"],
        augment_type=augment_type,
        augment_prob=augment_prob,
        original_dataset_root_path=original_dataset_root_path,
    )

    if dist.get_rank() == 0:
        # save dataset stats
        stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)

    best_ckpt_info = train_bc(
        train_dataloader, val_dataloader, config, logger_writter=writer
    )

    if dist.get_rank() == 0:
        best_step, min_val_loss, best_state_dict = best_ckpt_info
        # save best checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        dist.print0(f"Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}")
        writer.close()


if __name__ == "__main__":
    main()
