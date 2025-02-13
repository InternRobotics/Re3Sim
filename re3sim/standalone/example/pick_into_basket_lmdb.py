import os
from real2sim2real.envs.config import SimulatorConfig
from real2sim2real.envs.env import BaseEnv

file_path = "configs/example/pick_into_basket/collect_data_render_1_16_one_item.yaml"
sim_config = SimulatorConfig(file_path)

headless = sim_config.config.headless
webrtc = False

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.loggers.data_logger import DataLogger
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    pose_from_tf_matrix,
    get_relative_transform,
)
from real2sim2real.controllers.pick_and_place_mplib import PickAndPlaceController
from real2sim2real.controllers import get_action_for_multi_item_continuous as get_action
from real2sim2real.utils.logger import log
from real2sim2real.utils.utils import unset_seed
from pathlib import Path
import h5py
import json
import traceback
from typing import Dict, List
import imageio
import multiprocessing as mp
from queue import Empty
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shutil
from real2sim2real.logger.lmdb_logger import LmdbLogger as Logger

unset_seed()
i = 0
obs = env.reset()
task_name_lst = []
task_lst = []
obs_lst = []
log_once = False
for task_name, task in env.runner.current_tasks.items():
    task_name_lst.append(task_name)
    task_lst.append(task)
    obs_lst.append(obs[task_name])

loggers = {
    task_name: Logger(
        log_root_path=env.config.data_log_root_path,
        save_depth=env.config.save_depth,
        save_depth_as_png=env.config.save_depth_as_png,
    )
    for task_name in task_name_lst
}
controllers = {}
all_video_processes = []
for task_name, task in zip(task_name_lst, task_lst):
    task
    task.individual_reset()
    for i in range(30):
        env.runner._world.step(render=True)
        obs = env.get_observations()
    task._save_depth = env.config.save_depth
    loggers[task_name].debug(
        f"Joint closed positions: {task.robot.gripper.joint_closed_positions}"
    )
    loggers[task_name].debug(
        f"Joint opened positions: {task.robot.gripper.joint_opened_positions}"
    )
    joint_vel_limits = task.config.params.get("joint_vel_limits", None)
    if joint_vel_limits is not None:
        joint_vel_limits = np.array(joint_vel_limits)
    controller = PickAndPlaceController(
        name=task_name + "controller",
        task=task,
        urdf=str(os.path.join(task._asset_root, "urdfs/panda/panda.urdf")),
        srdf=str(os.path.join(task._asset_root, "urdfs/panda/panda.srdf")),
        dt=env.get_dt(),
        post_position_rela=np.array([0, 0, 0.25]),
        bg_collision_mesh_offset=sim_config.config.global_params.get(
            "bg_collision_mesh_offset", 0.0
        ),
        extra_controller_collisions=task.config.params.get(
            "extra_controller_collisions", []
        ),
        joint_vel_limits=joint_vel_limits,
    )
    controllers[task_name] = controller
    if task._render:
        loggers[task_name].info(
            f"Joint closed positions: {task.robot.gripper.joint_closed_positions}"
        )
        loggers[task_name].info(
            f"Joint opened positions: {task.robot.gripper.joint_opened_positions}"
        )
while env.simulation_app.is_running():
    actions = []
    for task_name, task in zip(task_name_lst, task_lst):
        logger = loggers[task_name]
        action, controller_done, info = get_action(
            controllers[task_name], obs[task_name]
        )
        if isinstance(action, Exception):
            print(controller_done)
            raise action
        actions.append(action)

        # log rendering data
        if task._render:
            render_obs = obs[task_name]["info"]["render"]
            logger.add_scalar_data("observations/qpos", render_obs["robot"]["qpos"])
            logger.add_scalar_data("observations/qvel", render_obs["robot"]["qvel"])
            logger.add_scalar_data("action", action["robot"])
            logger.add_scalar_data(
                "observations/ee_pose", render_obs["robot"]["ee_pose"]
            )
            if not log_once:
                logger.add_scalar_data(
                    "observations/robot_base_pose",
                    render_obs["random_info"]["robot_base_pose"],
                )
                mask_idToLabels = {}
                for camera_name in render_obs["fix_render_images"]:
                    logger.add_scalar_data(
                        f"observations/fix_render_images/{camera_name}",
                        render_obs["fix_render_images"][camera_name],
                    )
                for camera_name in render_obs["mask_idToLabels"]:
                    mask_idToLabels[camera_name] = render_obs["mask_idToLabels"][
                        camera_name
                    ]
                logger.add_json_data("mask_idToLabels", mask_idToLabels)
                logger.add_scalar_data(
                    "picking_item", render_obs["random_info"]["picking_item_idx"]
                )
                logger.add_json_data(
                    "picking_item_name", render_obs["random_info"]["picking_item_name"]
                )
                logger.add_json_data("item_names", [item.name for item in task.items])
                logger.add_json_data("random_results", task._random_results)
                log_once = True
            for camera_name in task.camera_names:
                logger.add_image_data_jpg(
                    f"observations/sim_images/{camera_name}",
                    render_obs["sim_images"][camera_name],
                )
                if camera_name in render_obs["render_images"]:
                    logger.add_image_data_jpg(
                        f"observations/render_images/{camera_name}",
                        render_obs["render_images"][camera_name],
                    )
                logger.add_image_data_png(
                    f"observations/mask/{camera_name}", render_obs["mask"][camera_name]
                )
                if "depths" in render_obs and camera_name in render_obs["depths"]:
                    logger.add_depth_data(
                        f"observations/depths/{camera_name}",
                        render_obs["depths"][camera_name],
                    )
        if obs[task_name]["done"] or controller_done:
            end_episode = True
            if (
                obs[task_name]["reward"] >= 10
                and info
                and (info["status"] == "item falling" or info["status"] == "done")
            ):
                if task.get_left_item_num() == 0:
                    log.info(f"task {task_name} success!")
                    log_path = env.runner.log_data(task_name)
                    print("Saving log data to: ", log_path)
                    # log rendering data
                    if task._render:
                        still_saving = loggers[task_name].save()
                        if not still_saving:
                            print("Disk space is not enough, exit")
                            env.simulation_app.close()
                            exit(0)
                else:
                    task.pick_next_item()
                    end_episode = False
            if end_episode:
                task.individual_reset()
                log_once = False
                for i in range(30):
                    env.runner._world.step(render=True)
                    obs = env.get_observations()
                i = 0
                controllers[task_name].set_base_pose(
                    task.robot_base_pose, remove_collision_mesh=True
                )
                env.runner.reset_data_logger(task_name)
                if task._render:
                    loggers[task_name].clear()

    obs = env.step(actions)

    i += 1
print("Done")
env.simulation_app.close()
