import argparse
import numpy as np
import time
from panda_py import libfranka
import cv2
import os
import yaml
import torch
from einops import rearrange
import pickle
from torchvision import transforms
import imageio

from policies.act_plus_plus.constants import FPS, PUPPET_GRIPPER_JOINT_OPEN, SIM_TASK_CONFIGS
from policies.act_plus_plus.utils import (
    load_data,
    sample_box_pose,
    sample_insertion_pose,
    compute_dict_mean,
    set_seed,
    detach_dict,
    calibrate_linear_vel,
    postprocess_base_action,
)
from policies.act_plus_plus.policy import ACTPolicy, CNNMLPPolicy
from policies.act_plus_plus.visualize_episodes import save_videos
from policies.act_plus_plus.detr.models.latent_model import Latent_Model_Transformer
from .realsense_reader import MultiRealSenseCamera
from .realrobot import RealRobot
import zarr


def zarr_to_dict(zarr_file):
    root = zarr.open(zarr_file, mode="r")
    return {
        key: zarr_to_dict(value) if isinstance(value, zarr.Group) else np.array(value)
        for key, value in root.items()
    }


def show_images(images_dict):
    for cam_name in images_dict:
        cv2.imshow(cam_name, cv2.cvtColor(images_dict[cam_name], cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


def main(args):
    # global settings
    set_seed(1)
    headless = args.get("headless", True)

    real_robot = RealRobot()
    # command line parameters
    global_params = args["global_params"]
    ckpt_dir = global_params["ckpt_dir"]
    policy_class = global_params["policy_class"]
    # get task parameters
    # num_episodes = task_config['num_episodes']
    episode_len = -1
    camera_names = global_params["camera_names"]
    ckpt_names = global_params["ckpt_names"]
    # fixed parameters
    state_dim = 14
    backbone = global_params.get("backbone", "resnet18")
    # other parameters
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 9
    policy_config = {
        "lr": global_params["lr"],
        "num_queries": global_params["chunk_size"],
        "kl_weight": global_params["kl_weight"],
        "hidden_dim": global_params["hidden_dim"],
        "dim_feedforward": global_params["dim_feedforward"],
        "backbone": backbone,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "camera_names": camera_names,
        "vq": global_params["use_vq"],
        "vq_class": global_params["vq_class"],
        "vq_dim": global_params["vq_dim"],
        "action_dim": 11,
        "state_dim": state_dim,
        "no_encoder": global_params["no_encoder"],
        "qpos_noise_std": global_params["qpos_noise_std"],
        "freeze_backbone": global_params.get("freeze_backbone", True),
    }

    config = {
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "temporal_agg": global_params["temporal_agg"],
        "camera_names": camera_names,
        "load_pretrain": global_params["load_pretrain"],
        "relative_control": global_params.get("relative_control", False),
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    policy_class = config["policy_class"]
    policy_config: dict = config["policy_config"]
    camera_names = config["camera_names"]
    dt = 1 / args["fps"]
    print(dt)
    max_timesteps = (
        config["episode_len"] if config["episode_len"] != -1 else int(100 * args["fps"])
    )
    temporal_agg = config["temporal_agg"]
    real_camera_names = global_params["real_camera_names"]
    # load policy and stats
    ckpt_name = ckpt_names[0]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    # print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    try:
        with open(os.path.join(ckpt_dir, f"dataset_stats.pkl"), "rb") as f:
            stats = pickle.load(f)
    except Exception as e:
        print(f"Error: {e}")
        stats_path = os.path.join(ckpt_dir, f"dataset_stats.zarr")
        stats = zarr_to_dict(stats_path)

    print("Loading stats done")
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    if policy_class == "Diffusion":
        post_process = (
            lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
            + stats["action_min"]
        )
    else:
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    query_frequency = global_params["query_frequency"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, policy_config["action_dim"]]
        ).cuda()

    with torch.inference_mode():
        try:
            for t in range(max_timesteps + 2):
                time0 = time.time()
                qpos_numpy = real_robot.get_robot_state()
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                if t % query_frequency == 0:
                    images = real_robot.cameras.undistorted_rgbd()
                    images_dict = {
                        real_camera_name: image
                        for real_camera_name, image in zip(real_camera_names, images[0])
                    }
                    if not headless:
                        show_images(images_dict)
                    curr_image = get_image(
                        images_dict,
                        camera_names,
                        rand_crop_resize=(config["policy_class"] == "Diffusion"),
                    )

                print(qpos.dtype)
                print(qpos.device)
                print(qpos)
                print(next(policy.parameters()).device)
                print(curr_image.device)
                if t == 0:
                    # warm up
                    for _ in range(10000000):
                        time1 = time.time()
                        policy(qpos, curr_image)
                        print(f"Time for one forward pass: {time.time() - time1}")
                    print("network warm up done")
                    time1 = time.time()

                ### query policy
                if t % query_frequency == 0:
                    time1 = time.time()
                    all_actions = policy(qpos, curr_image)
                    print(f"Time for one forward pass: {time.time() - time1}")
                if temporal_agg:
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(
                        dim=0, keepdim=True
                    )
                else:
                    raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]
                time2 = time.time()
                if config["relative_control"]:
                    target_qpos = target_qpos + qpos.detach().cpu().numpy()[0]

                ### step the environment
                if sum(target_qpos[-2:]) > 0.05:
                    target_qpos[-2:] = 0.04
                else:
                    target_qpos[-2:] = 0
                time3 = time.time()
                print(f"Target qpos: {target_qpos}")
                real_robot.apply_action(target_qpos)
                time4 = time.time()
                time.sleep(max(0, dt - (time.time() - time0)))
                print(
                    f"Avg fps {1 / (time.time() - time0)}, time for get qpos: {time2 - time1}, time for apply action: {time4 - time3}"
                )

        finally:
            real_robot.end()
            print("end")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(images_dict, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images_dict[cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print("rand crop resize is used!")
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


if __name__ == "__main__":
    config_path = "/isaac-sim/src/frankapy/src/configs/control_by_model1010_1.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
