import argparse
import h5py
import numpy as np
import time

from pathlib import Path
import os
import random
import cv2
import re
from scipy.spatial.transform import Rotation as R
from src.frankapy.src.realrobot import PandaRealRobot
from multiprocessing import Process, Queue, Lock
import pyspacemouse
from pynput import keyboard
from pynput.keyboard import Key, Listener


def read_spacemouse(queue, lock):
    while True:
        state = pyspacemouse.read()
        with lock:
            if queue.full():
                queue.get()
            queue.put(state)


def clip_delta_min(delta, threshold):
    delta = np.where(abs(delta) < threshold, 0, delta)
    return delta


def create_formated_skill_dict(
    joints, end_effector_positions, time_since_skill_started
):
    skill_dict = dict(skill_description="GuideMode", skill_state_dict=dict())
    skill_dict["skill_state_dict"]["q"] = np.array(joints)
    skill_dict["skill_state_dict"]["O_T_EE"] = np.array(end_effector_positions)
    skill_dict["skill_state_dict"]["time_since_skill_started"] = np.array(
        time_since_skill_started
    )

    # The key (0 here) usually represents the absolute time when the skill was started but
    formatted_dict = {0: skill_dict}
    return formatted_dict


def get_log_folder(log_root: str):
    log_folder = Path(log_root) / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def get_json_log_path(log_folder: Path):
    log_folder = Path(log_folder)
    files = os.listdir(log_folder)
    pattern = r"log-(\d{6})-\d{4}"
    existing_numbers = [
        int(re.match(pattern, file).group(1))
        for file in files
        if re.match(pattern, file)
    ]
    if not existing_numbers:
        next_number = 1
    else:
        existing_numbers.sort()
        next_number = existing_numbers[-1] + 1
    random_id = random.randint(1000, 9999)
    dir_path = log_folder / f"log-{next_number:06d}-{random_id}"
    os.makedirs(dir_path, exist_ok=True)
    new_filename = f"traj.json"
    return dir_path / new_filename


def get_hdf5_log_path(log_folder: Path):
    log_folder = Path(log_folder)
    files = os.listdir(log_folder)
    pattern = r"log-(\d{6})-\d{4}"
    existing_numbers = [
        int(re.match(pattern, file).group(1))
        for file in files
        if re.match(pattern, file)
    ]
    if not existing_numbers:
        next_number = 1
    else:
        existing_numbers.sort()
        next_number = existing_numbers[-1] + 1
    random_id = random.randint(1000, 9999)
    dir_path = log_folder / f"log-{next_number:06d}-{random_id}"
    os.makedirs(dir_path, exist_ok=True)
    new_filename = f"traj.hdf5"
    return dir_path / new_filename


def save_images(images, images_dir, headless=True):
    tmp_colors, tmp_depths = images
    now_time = int(time.time() * 1000)
    assert len(tmp_colors) == len(tmp_depths)
    for i in range(len(tmp_colors)):
        os.makedirs(images_dir / f"color_{i}", exist_ok=True)
        os.makedirs(images_dir / f"depth_{i}", exist_ok=True)
        cv2.imwrite(
            str(images_dir / f"color_{i}" / f"{now_time}.png"),
            cv2.cvtColor(tmp_colors[i], cv2.COLOR_RGB2BGR),
        )
        np.save(str(images_dir / f"depth_{i}" / f"{now_time}.npy"), tmp_depths[i])

    if not headless:
        for i in range(len(tmp_colors)):
            # turn depth into colorful
            depth = tmp_depths[i]
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow(f"Color {i}", cv2.cvtColor(tmp_colors[i], cv2.COLOR_RGB2BGR))
            cv2.imshow(f"Depth {i}", depth)
    return now_time


def save_dict_to_hd5f(group, dict_data):
    for key, value in dict_data.items():
        if isinstance(value, dict):
            save_dict_to_hd5f(group.create_group(key), value)
        else:
            group.create_dataset(key, data=value)


if __name__ == "__main__":
    translation_scale = 0.005
    rotation_scale = 0.5
    gripper_open = True
    data_dir = "/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/frankapy/logs"
    data_dir = get_log_folder(data_dir)
    real_robot = PandaRealRobot()
    # real_robot.close_gripper()
    i = 0
    obs = real_robot.get_obs()
    ee_pose = obs["panda_hand_pose"]
    success = pyspacemouse.open()
    queue = Queue(maxsize=1)
    lock = Lock()
    process = Process(target=read_spacemouse, args=(queue, lock))
    process.start()

    # init saver
    end_effector_position = []
    joints = []
    gripper_width = []
    h5_file = get_hdf5_log_path(data_dir)
    multi_camera = real_robot.cameras
    time_since_skill_started = []
    timestames = []
    start_time = time.time()
    last_time = time.time()
    try:
        while True:

            start = time.time()
            if not queue.empty():
                with lock:
                    state = queue.get()
            else:
                continue
            translation = ee_pose[:3, 3]
            rotation = R.from_matrix(ee_pose[:3, :3])
            print("state", state)
            delta_translation = np.array(
                [
                    state.x * translation_scale,
                    state.y * translation_scale,
                    state.z * translation_scale,
                ]
            )
            delta_translation = clip_delta_min(
                delta_translation, 0.1 * translation_scale
            )
            translation += delta_translation
            if state.buttons[1]:
                rotation_delta = R.from_euler(
                    "yxz",
                    [
                        state.roll * rotation_scale,
                        -state.pitch * rotation_scale,
                        -state.yaw * rotation_scale,
                    ],
                    degrees=True,
                )
                rotation = rotation_delta * rotation
            if state.buttons[0]:
                gripper_open = not gripper_open
            ee_pose = np.eye(4)
            ee_pose[:3, 3] = translation
            ee_pose[:3, :3] = rotation.as_matrix()
            action = (0.08 if gripper_open else 0.0, ee_pose)
            obs = real_robot.step(action, type="ee")
            i += 1
            time.sleep(max(0, 1 / 15 - (time.time() - start)))
            print(f"fps for one step: {1 / (time.time() - start)}")

            # collect data
            end_effector_position.append(ee_pose)
            joints.append(real_robot.get_robot_state())
            print(real_robot.get_robot_state())
            time_since_skill_started.append(time.time() - start_time)
            # time_stamp = save_images(real_robot.cameras.undistorted_rgbd(), h5_file.parent, headless=False)
            print(type(real_robot.cameras.undistorted_rgbd()))
            timestames.append(time.time())

            # with h5py.File(h5_file, 'w') as f:
            #     f["end_effector_position"] = np.array(end_effector_position)
            #     f["joints"] = np.array(joints)
            #     f["time_since_skill_started"] = np.array(time_since_skill_started)
            #     f["timestames"] = np.array(timestames)
            #     f["start_time"] = start_time
            #     f["last_time"] = last_time
            #     intr_colors = multi_camera.get_intrinsic_color()
            #     intr_depths = multi_camera.get_intrinsic_depth()
            #     for i, (intr_color, intr_depth) in enumerate(zip(intr_colors, intr_depths)):
            #         group = f.create_group(f"intrinsic_color_{i}")
            #         save_dict_to_hd5f(group, intr_color)
            #         group = f.create_group(f"intrinsic_depth_{i}")
            #         save_dict_to_hd5f(group, intr_depth)

    finally:
        real_robot.end()
