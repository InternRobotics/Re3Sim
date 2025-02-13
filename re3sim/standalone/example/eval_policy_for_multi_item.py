import torch
import numpy as np
import imageio
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import zarr
from real2sim2real.utils.zarr_utils import zarr_to_dict
import time
from torchvision import transforms

from real2sim2real.act_plus_plus.constants import FPS
from real2sim2real.act_plus_plus.constants import (
    PUPPET_GRIPPER_JOINT_OPEN,
    SIM_TASK_CONFIGS,
)
from real2sim2real.act_plus_plus.utils import load_data  # data functions
from real2sim2real.act_plus_plus.utils import (
    sample_box_pose,
    sample_insertion_pose,
)  # robot functions
from real2sim2real.act_plus_plus.utils import (
    compute_dict_mean,
    set_seed,
    detach_dict,
    calibrate_linear_vel,
    postprocess_base_action,
)  # helper functions
from real2sim2real.act_plus_plus.policy import ACTPolicy, ACTPolicyDinov2
from real2sim2real.act_plus_plus.visualize_episodes import save_videos

from real2sim2real.act_plus_plus.detr.models.latent_model import (
    Latent_Model_Transformer,
)
import IPython

e = IPython.embed

import sys
from tqdm import tqdm
import imageio
from real2sim2real.envs.env import BaseEnv
from real2sim2real.envs.config import SimulatorConfig
from scipy.spatial.transform import Rotation as R


def get_action_type_from_config(config: dict) -> str:
    if config.get("dataset_cls", None) == "RealRobotDataset":
        print("Using ee control")
        return "ee"
    else:
        print("Using qpos control")
        return "qpos"


control_op_stat = 0


def on_press(key):
    global control_op_stat
    if str(key) == "'r'":
        control_op_stat = 2
    elif str(key) == "'c'":
        control_op_stat = 1


def main(sim_config: SimulatorConfig):
    # set_seed(1)
    global_params = sim_config.config.global_params
    # command line parameters
    ckpt_dir = global_params["ckpt_dir"]
    with open(os.path.join(ckpt_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    ckpt_names = global_params["ckpt_names"]

    # override config
    if "temporal_agg" in global_params:
        config["temporal_agg"] = global_params["temporal_agg"]
    if "relative_control" in global_params:
        config["relative_control"] = global_params["relative_control"]
    if "dataset_cls" in global_params:
        config["dataset_cls"] = global_params["dataset_cls"]
    if "num_queries" in global_params:
        config["eval_num_queries"] = global_params["num_queries"]
    else:
        config["eval_num_queries"] = config["policy_config"]["num_queries"]
    config["action_type"] = get_action_type_from_config(config)
    config["ckpt_dir"] = ckpt_dir
    results = []
    if config["dataset_cls"] == "EpisodeDataset":
        assert (
            config["temporal_agg"] == True
        ), "temporal_agg must be True for EpisodeDataset"
    if not sim_config.config.headless:
        from pynput import keyboard
        from pynput.keyboard import Key, Listener

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(
            sim_config, config, ckpt_name, save_episode=True, num_rollouts=50
        )
        # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
        results.append([ckpt_name, success_rate, avg_return])


def eval_bc(
    sim_config: SimulatorConfig, config, ckpt_name, save_episode=True, num_rollouts=50
):
    # set_seed(8)
    global control_op_stat
    ckpt_dir = config["ckpt_dir"]
    policy_config: dict = config["policy_config"]
    camera_names = config["camera_names"]
    print(f"camera_names: {camera_names}")
    dt = (
        eval(sim_config.config.simulator.physics_dt)
        if isinstance(sim_config.config.simulator.physics_dt, str)
        else sim_config.config.simulator.physics_dt
    )
    max_timesteps = (
        config["episode_len"] if config["episode_len"] != -1 else int(40 / dt)
    )
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    log_video = sim_config.config.global_params.get("log_video", False)
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_config, distributed=False)
    loading_status = policy.model.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    try:
        pickle_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
        with open(pickle_path, "rb") as f:
            stats = pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle stats: {e}\n, try loading zarr: ")
        stats_path = os.path.join(ckpt_dir, f"dataset_stats.zarr")
        stats = zarr_to_dict(stats_path)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    # query_frequency = policy_config['num_queries']
    query_frequency = config["eval_num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]
    print(f"query_frequency: {query_frequency}; log video: {log_video}")

    # load environment
    print(f"sim_config.config.headless: {sim_config.config.headless}")
    env = BaseEnv(sim_config, headless=sim_config.config.headless, to_log=False)
    env_max_reward = 10

    episode_returns = []
    highest_rewards = []
    env.reset()
    task_name = list(env.runner.current_tasks.keys())[0]
    env.runner.current_tasks[task_name].individual_reset()
    for i in range(10):
        env.runner._world.step(render=True)
    obs = env.get_observations()
    rollout_id = 0
    picked_item_num_list = []
    picked_item_num = 0
    while True:
        if rollout_id >= num_rollouts:
            break
        assert len(env.runner.current_tasks.items()) == 1, "Only one task is supported"
        task_name = list(env.runner.current_tasks.keys())[0]
        task = env.runner.current_tasks[task_name]
        total_items = task.get_left_item_num()
        print(f"There are {total_items} items to pick:")
        print(
            f"Recording rollout {rollout_id} in {ckpt_name} at a frequency of {int(1 / dt)} Hz"
        )
        if temporal_agg:
            all_time_actions = torch.zeros(
                [
                    max_timesteps,
                    max_timesteps + num_queries,
                    policy_config["action_dim"],
                ]
            ).cuda()

        rewards = []
        max_item_num = task.get_left_item_num()
        print(f"max_item_num: {max_item_num}")
        if log_video:
            video_writers = {camera_name: [] for camera_name in camera_names}
            os.makedirs(
                os.path.join(
                    ckpt_dir,
                    os.path.splitext(ckpt_name)[0] + f"_num_queries_{query_frequency}",
                ),
                exist_ok=True,
            )
            print(
                "\n\n\n make dir: ",
                os.path.join(
                    ckpt_dir,
                    os.path.splitext(ckpt_name)[0] + f"_num_queries_{query_frequency}",
                ),
                "\n\n\n",
            )
        else:
            print("\n\n\n no log video \n\n\n")
        finish = False
        with torch.inference_mode():
            time0 = time.time()
            task._task_idx = rollout_id
            t = 0
            while True:
                if t >= max_timesteps:
                    break
                task_obs = obs[task_name]
                true_obs = task_obs["observations"]
                images_dict = true_obs["images"]
                qpos_numpy = np.array(true_obs["robot"]["qpos"])
                if config["action_type"] == "ee":
                    ee_matrix = np.array(true_obs["robot"]["ee_pose"])
                    obs_ee_translation = ee_matrix[:3, 3]
                    obs_ee_euler = R.from_matrix(ee_matrix[:3, :3]).as_euler("xyz")
                    qpos_numpy = np.concatenate(
                        [qpos_numpy, obs_ee_translation, obs_ee_euler], axis=-1
                    )
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # frame_server.publish_frame(images_dict["right_camera"])
                if log_video:
                    for camera in camera_names:
                        frame = images_dict[camera]
                        video_writers[camera].append(frame)
                if t % query_frequency == 0:
                    curr_image = get_image(images_dict, camera_names)
                # print('get image: ', time.time() - time2)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos, curr_image)
                    print("network warm up done")
                    time1 = time.time()

                ### query policy
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
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
                if config["action_type"] == "ee":
                    gripper_action = action[:2]
                    ee_translation = action[2:5]
                    ee_euler = action[5:8]
                    if config["relative_control"]:
                        obs_ee_euler = qpos_numpy[-3:]
                        obs_ee_translation = qpos_numpy[-6:-3]
                        ee_euler = ee_euler + obs_ee_euler
                        ee_translation = ee_translation + obs_ee_translation
                    # print("gripper_action: ", gripper_action)
                    if sum(gripper_action) > 0.05:
                        gripper_width = 0.08
                    else:
                        gripper_width = 0.0

                    ee_matrix = R.from_euler("xyz", ee_euler).as_matrix()
                    transform = np.eye(4)
                    # TODO: rotation has bug
                    transform[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    transform[:3, :3] = ee_matrix
                    transform[:3, 3] = ee_translation
                    env_action = [{"type": "ee", "robot": (gripper_width, transform)}]
                else:
                    target_qpos = action[:-2]
                    ### step the environment
                    if sum(target_qpos[-2:]) > 0.05:
                        target_qpos[-2:] = 0.04
                    else:
                        target_qpos[-2:] = 0
                    env_action = [{"robot": target_qpos}]

                obs = env.step(env_action)
                # print('step env: ', time.time() - time5)

                rewards.append(obs[task_name]["reward"])

                # print("Action: ", target_qpos)
                if (
                    obs[task_name]["reward"] >= 6
                    and sum(obs[task_name]["observations"]["robot"]["qpos"][-2:])
                    >= 0.07
                ):
                    print("Succeed in catching item!")
                    if task.get_left_item_num() == 0:
                        print("Succeed in all items!")
                        finish = True
                    else:
                        print("Picking next item...")
                        task.pick_next_item()
                        picked_item_num += 1
                        t = -1
                        for i in range(20):
                            env.runner._world.step(render=True)
                            obs = env.get_observations()

                if (
                    obs[task_name]["done"]
                    or t >= max_timesteps - 1
                    or control_op_stat in [2, 1]
                    or finish
                ):
                    finish = True
                    if control_op_stat == 2:
                        print("Reset the environment")
                        control_op_stat = 0
                    elif control_op_stat == 1:
                        print("Continue to next rollout")
                        control_op_stat = 0
                        rollout_id += 1
                    else:
                        rollout_id += 1
                    env.runner.reset_data_logger(task_name)
                    picked_item_num_list.append(picked_item_num)
                    picked_item_num = 0
                    task.individual_reset()
                    for i in range(10):
                        env.runner._world.step(render=True)
                    obs = env.get_observations()
                    break
                t += 1

            print(f"Avg fps {max_timesteps / (time.time() - time0)}")
        if log_video:
            for camera_name in camera_names:
                video_path = os.path.join(
                    ckpt_dir,
                    os.path.splitext(ckpt_name)[0] + f"_num_queries_{query_frequency}",
                    f"rollout_{rollout_id}_{camera_name}.mp4",
                )
                imageio.mimsave(video_path, video_writers[camera_name], fps=(1 / dt))
                del video_writers[camera_name]
        if t >= max_timesteps - 1 or control_op_stat in [2, 1] or finish:
            print(f"Now rollout id is {rollout_id}")
            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards != None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            print(
                f"Rollout {rollout_id}\n{episode_return}, {episode_highest_reward}, {env_max_reward}, Success: {episode_highest_reward==env_max_reward}"
            )
            if episode_highest_reward == 2:
                print("Succeed in reaching the cube.")
            elif episode_highest_reward == 6:
                print("Succeed in catching the cube")
            success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
            avg_return = np.mean(episode_returns)
            reaching_cube_rate = np.mean(np.array(highest_rewards) >= 2)
            catching_cube_rate = np.mean(np.array(highest_rewards) >= 6)
            avg_return = np.mean(np.array(highest_rewards))
            summary_str = f"\nReaching the cube rate: {reaching_cube_rate}\n Catching the cube rate: {catching_cube_rate}\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
            for r in range(env_max_reward + 1):
                more_or_equal_r = (np.array(highest_rewards) >= r).sum()
                more_or_equal_r_rate = more_or_equal_r / num_rollouts
                summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"
            summary_str += (
                f"Picked item num: {picked_item_num}, total item num: {max_item_num}\n"
            )
            success_rate_list = []
            for i in range(max_item_num + 1):
                summary_str += f"The success rate of picking item {i} is {picked_item_num_list.count(i)/(rollout_id)*100}%\n"
                success_rate_list.append(picked_item_num_list.count(i) / (rollout_id))

            summary_str += "\n"
            for i in range(max_item_num + 1):
                summary_str += f"The success rate of picking at least {i} items is {sum(success_rate_list[i:])*100}%\n"

            summary_str += f"picked_item_num_list: {str(picked_item_num_list)}\n"
            print(summary_str)

            # save success rate to txt
            result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
            with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
                f.write(summary_str)
                f.write(repr(episode_returns))
                f.write("\n\n")
                f.write(repr(highest_rewards))
    return success_rate, avg_return


def make_policy(policy_config, distributed=True):
    if policy_config["backbone"] == "dino_v2":
        policy = ACTPolicyDinov2(policy_config, distributed)
    else:
        policy = ACTPolicy(policy_config, distributed)
    return policy


def get_image(images_dict, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images_dict[cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        # print('rand crop resize is used!')
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
    sim_config = SimulatorConfig("/tmp/config.yaml")
    main(sim_config)
