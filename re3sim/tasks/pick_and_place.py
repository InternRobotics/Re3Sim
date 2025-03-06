from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple
import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.prims import GeometryPrim, XFormPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.franka.franka import Franka
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.transformations import (
    tf_matrices_from_poses,
    pose_from_tf_matrix,
    get_relative_transform,
    tf_matrix_from_pose,
)
from omni.isaac.core.utils.prims import (
    is_prim_path_valid,
    create_prim,
    get_prim_at_path,
)
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
import random
import os
import sys
import tempfile

sys.path.append("/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0")
from ..utils.utils import *
from ..utils.items import Item, create_camera
from ..utils.prim import get_points_at_path, franka_picked_up
import roboticstoolbox as rtb

from .task import BaseTask
from ..envs.config import TaskUserConfig
from ..utils.logger import log
from ..utils.frame_tools import FrameTools
from ..utils.motion_control import PinocchioMotionControl
from ..randomize import Randomize
from ..background import Background
from pathlib import Path


def load_transform_matrix(transform, default=None):
    if transform is None:
        if default is None:
            raise ValueError("Invalid transform type")
        transform = default
    elif isinstance(transform, list):
        transform = np.array(transform)
    elif isinstance(transform, str):
        transform = np.load(transform)
    else:
        raise NotImplementedError("Invalid transform type")
    return transform


def tmp_func(matrix):
    matrix[:3, :3] = np.eye(3)
    matrix[:2, :2] *= -1
    return matrix


@BaseTask.register("PickAndPlace")
class PickAndPlace(BaseTask):
    def __init__(self, config: TaskUserConfig, scene: Scene):
        super().__init__(config, scene)
        # global config
        self.config = config
        config_params = config.params
        self._asset_root = config_params.get("asset_root", ".")
        self._root_prim_path = config.root_path
        self._save_depth = config_params.get("save_depth", False)

        # franka
        self._franka_cam_prim_path = (
            config.root_path
            + config.robots_root_path
            + config_params.get("franka_cam_prim_path", "/X1")
        )
        self._franka_prim_path = self._franka_cam_prim_path + "/Franka"
        self._franka_robot_name = (
            f"{config_params.get('franka_robot_name', 'franka')}_{config.env_id}"
        )

        # target
        self._target_position = np.array(
            config_params.get("target_position", [-0.17, 0.13, 0.6])
        )
        self._target_orientation = np.array(
            config_params.get("target_orientation", euler_to_quaternion(0, 0, 0))
        )
        self._item_target_region = np.array(
            config_params.get("item_target_region", [[-0.23, 0.1], [-0.5, -0.3]])
        )
        # settings
        self._priviliged_info = config_params.get("priviliged_info", False)
        self._render = config_params.get("render", False)
        self._physics_dt = config_params.get("physics_dt", 1 / 60)

        # randomize
        self._random_params: Dict = config_params.get("random", {})
        self._random_method: List[Randomize] = [
            Randomize.randomize_dict[random_func_name](random_params)
            for random_func_name, random_params in self._random_params.items()
        ]

        # frames
        self._frames = FrameTools("isaacsim")
        marker_to_isaacsim = config_params.get("marker_to_isaacsim", None)
        tmp = np.eye(4)
        tmp[2, 3] = 0.8
        marker_to_isaacsim = load_transform_matrix(marker_to_isaacsim, tmp)
        marker_to_robot_base = config_params.get("marker_to_robot_base", None)
        marker_to_robot_base = load_transform_matrix(marker_to_robot_base)

        # background
        self.background = Background(
            self._asset_root, config.root_path, marker_to_isaacsim, config.background
        )

        # temp
        # marker_to_robot_base = tmp_func(marker_to_robot_base)
        robot_base_to_marker = np.linalg.inv(marker_to_robot_base)

        self._frames.add_frame_transform("marker", marker_to_isaacsim)
        self._frames.add_frame_transform_relative_to(
            "robot_base", "marker", robot_base_to_marker
        )
        # self._frames.save("src/tests/frame.json")

        # init
        self._initialized = False
        self._initialized_semantic_index = False
        self._global_step_index = 0
        self._physics_dt = (
            eval(self._physics_dt)
            if isinstance(self._physics_dt, str)
            else self._physics_dt
        )
        self._totol_global_step_index = int(
            config_params.get("task_max_duration", 90) / self._physics_dt
        )
        self._item_start_height = self._frames.get_frame_translation("marker")[2]
        self._picking_item = 0
        self._item_picked_times = {}
        self._picked_item = set()

        self._tmp_dir = tempfile.mkdtemp()
        print(f"Temporary directory created at: {self._tmp_dir}")

    def load(self):
        # franka_asset_path = os.path.join(self._asset_root, "usd/franka_with_cameras.usd")
        self.robots = {}
        self.objects = {}
        if is_prim_path_valid(self._root_prim_path):
            xform_prim = XFormPrim(prim_path=self._root_prim_path)
            xform_prim.set_local_pose(translation=self.config.offset)
            log.debug("Found prim at", self._root_prim_path)
        else:
            create_prim(
                prim_path=self._root_prim_path,
                prim_type="Xform",
                translation=self.config.offset,
            )
            log.debug(f"Created prim at {self._root_prim_path}")
        self.background.load()
        self.robot = self.set_robot()
        self._franka_view = ArticulationView(self._franka_prim_path)

        self.items: List[Item] = []
        self._item_name_to_index = {}
        for item_config in self.config.items:
            item = Item(
                frame_tools=self._frames,
                root_prim_path=self._root_prim_path,
                **item_config,
            )
            if item.object is not None:
                self.scene.add(item.object)
            self.items.append(item)
            self._item_name_to_index[item.name] = len(self.items) - 1
            self._item_picked_times[item.name] = 0
        self._mile_stone = {
            item.name: {
                "at_the_cube": False,
                "catch_the_cube": False,
                "cube_at_the_target": False,
            }
            for item in self.items
        }
        self._task_objects[self.robot.name] = self.robot
        self._picking_item = 0

    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """

        franka_translations = self._frames.get_frame_translation("robot_base")
        franka_orientations = self._frames.get_frame_rotation("robot_base", "quat")
        xform = XFormPrim(prim_path=self._franka_cam_prim_path)
        xform.set_local_pose(
            translation=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0]
        )
        try:
            robot = Franka(
                prim_path=self._franka_prim_path,
                name=self._franka_robot_name,
                usd_path=self.config.params.get("franka_usd_path", None),
            )
        except:
            prim = XFormPrim(prim_path=self._franka_prim_path)
            robot = Franka(
                prim_path=self._franka_prim_path,
                name=self._franka_robot_name,
                usd_path=self.config.params.get("franka_usd_path", None),
            )
        robot.set_local_pose(
            translation=franka_translations, orientation=franka_orientations
        )
        robot.set_solver_position_iteration_count(32)
        robot.set_solver_velocity_iteration_count(32)
        self._cspace_controller = RMPFlowController(
            name=self._franka_robot_name + "_cspace_controller",
            robot_articulation=robot,
        )
        self.ee_controller = PinocchioMotionControl(
            urdf_path=os.path.join(self._asset_root, "urdfs/panda/panda.urdf"),
            wrist_name="panda_hand",
            arm_init_qpos=np.array([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04]),
        )
        return robot

    def initialize(self):
        self._initialized = True
        self.robot.initialize()
        self.robot._articulation_view.initialize()
        self._franka_view.initialize()
        self._franka_view.set_max_joint_velocities([1.0] * 9)
        # add semantic info
        if self._priviliged_info and not self._render:
            return
        set_semantic_label(get_prim_at_path("/World/groundPlane"), "ground")
        set_semantic_label(get_prim_at_path(self.background.prim_path), "table")
        for item in self.items:
            set_semantic_label(item.object.prim, item.name)
        self._cameras = []
        self._cameras_name = []
        self._fix_cameras = {}
        self._camera_params = []
        for camera_config in self.config.cameras:
            if camera_config.get("wrist_camera", False):
                camera_prim_path = (
                    self._franka_prim_path
                    + "/panda_hand"
                    + camera_config["relative_prim"]
                )
            else:
                camera_prim_path = (
                    self._franka_prim_path + camera_config["relative_prim"]
                )
            camera = create_camera(
                camera_config["name"],
                camera_prim_path,
                camera_config["position"],
                camera_config["orientation"],
                camera_config["camera_params"],
            )
            self._camera_params.append(camera_config["camera_params"])
            self._cameras.append(camera)
            self._cameras_name.append(camera_config["data_name"])
            if camera_config.get("fixed", False):
                self._fix_cameras[camera_config["data_name"]] = None
            # camera.initialize()
            camera.add_motion_vectors_to_frame()
            camera.add_semantic_segmentation_to_frame()
            camera.add_distance_to_image_plane_to_frame()

    def initialize_semantic_index(self, segment_data) -> Tuple[int, int]:
        info = segment_data["info"]
        name_to_index = {}
        for key, value in info["idToLabels"].items():
            name_to_index[value["class"]] = int(key)
        # assert table_index_tmp is not None and ground_indexes_tmp is not None
        return name_to_index

    def individual_reset(self):
        self.robot.gripper.open()
        self._global_step_index = 0

        self.robot.set_joint_positions([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04])
        self._mile_stone = {
            item.name: {
                "at_the_cube": False,
                "catch_the_cube": False,
                "cube_at_the_target": False,
            }
            for item in self.items
        }
        self._item_start_height = self._frames.get_frame_translation("marker")[2]
        del self.ee_controller
        self.ee_controller = PinocchioMotionControl(
            urdf_path=os.path.join(self._asset_root, "urdfs/panda/panda.urdf"),
            wrist_name="panda_hand",
            arm_init_qpos=np.array([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04]),
        )
        for item in self.items:
            item.reset(align_surface=True, table_pcd=self.bg_pcd, percent=100)

        status = False
        max_attempts = 1000  
        attempt = 0
        self._picked_item = set()
        random_results = {}
        while not status and attempt < max_attempts:
            for random_method in self._random_method:
                random_results[random_method.name] = random_method.randomize(task=self)
            self._picking_item, status = self.get_picking_item()
            attempt += 1

        if not status:
            print("Warning: Could not find a suitable picking item after max attempts")
            self._picking_item, status = random.choice(range(len(self.items)))

        self._random_results = random_results
        self._picked_item.add(self.items[self._picking_item].name)
        print(f"Picking item: {self.items[self._picking_item].name}")

    def get_picking_item(self):
        min_x = np.inf
        min_x_item_name = None
        for item in self.items:
            if item.name in self._picked_item:
                continue
            pcd = get_points_at_path(
                item.object.prim_path, relative_frame_prim_path=self._franka_prim_path
            )
            item_x_min = np.min(pcd[:, 0])
            if item_x_min < min_x:
                min_x = item_x_min
                min_x_item_name = item.name

        idx = self._item_name_to_index[min_x_item_name]
        item = self.items[idx]

        max_picked_items = max(self._item_picked_times.values())
        min_picked_items = min(self._item_picked_times.values())
        if max_picked_items - min_picked_items <= 3:
            return idx, True

        max_times_item_names = [
            name
            for name, times in self._item_picked_times.items()
            if times == max_picked_items
        ]
        if item.name in max_times_item_names:
            return 0, False

        return idx, True

    def is_done(self) -> bool:
        """[summary]"""
        joint_state = self.robot.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities
        cube_position, _ = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(self.items[self._picking_item].object.prim_path),
                get_prim_at_path(self._root_prim_path),
            )
        )
        if (
            np.any(np.isnan(qpos))
            or np.any(np.isnan(qvel))
            or np.any(np.isnan(cube_position))
        ):
            return True
        # timeout
        if self._global_step_index > self._totol_global_step_index:
            return True
        return False

    def get_left_item_num(self, return_index=False):
        i = 0
        index = 0
        for item in self.items:
            milestone = self._mile_stone[item.name]
            if not milestone["catch_the_cube"]:
                i += 1
                index = i
        if return_index:
            return i, index
        else:
            return i

    def apply_action(self, action: Dict[str, np.ndarray]) -> None:
        from omni.isaac.core.utils.types import ArticulationAction

        action_type = action.get(f"type_{self.config.env_id}", "qpos")
        self._global_step_index += 1
        franka_action = action.get(f"robot_{self.config.env_id}")
        if action_type == "qpos":
            franka_action = np.array(franka_action)
            if franka_action.shape[-1] == 7:
                gripper_joints = self.robot.gripper.get_joint_positions()
                franka_action = np.concatenate([franka_action, gripper_joints])
            elif franka_action.shape[-1] == 9:
                pass
            else:
                raise ValueError("Invalid action shape")
            if type(franka_action) != np.ndarray:
                print("!")
            self._franka_view.set_joint_position_targets(franka_action)
            # self.robot.gripper.apply_action(ArticulationAction(joint_positions=franka_action[-2:], ))
        elif action_type == "ee":

            gripper_width, transform = franka_action
            pos, rot_mat = transform[:3, 3], transform[:3, :3]
            sol = self.ee_controller.control(pos, rot_mat)[:7]
            qpos = np.concatenate([sol, [gripper_width / 2], [gripper_width / 2]])
            # position, orientation = pose_from_tf_matrix(transform)
            # qpos = self._cspace_controller.forward(target_end_effector_position=position, target_end_effector_orientation=orientation)
            # qpos = np.concatenate([qpos.joint_positions.reshape(-1), [gripper_width / 2], [gripper_width / 2]])
            self._franka_view.set_joint_position_targets(qpos)

    def get_observations_non_privileged(self) -> Dict[str, Any]:
        """All pose related are in robot base frame

        Returns:
            Dict[str, Any]: _description_
        """
        joint_state = self.robot.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities
        ee_matrix = get_relative_transform(
            get_prim_at_path(self._franka_prim_path + "/panda_hand"),
            get_prim_at_path(self._franka_prim_path),
        )
        robot_base_pose = get_relative_transform(
            get_prim_at_path(self._franka_prim_path),
            get_prim_at_path(self._root_prim_path),
        )
        obs = {}
        obs["robot"] = {
            "qpos": qpos,
            "qvel": qvel,
            "ee_pose": ee_matrix,
            "robot_base_pose": robot_base_pose,
        }

        images = {}
        obs["root_info"] = {}
        for i, (camera_name, camera) in enumerate(
            zip(self._cameras_name, self._cameras)
        ):
            tmp = camera.get_rgba()
            if len(tmp.shape) == 1:
                log.warn(f"Camera {camera_name} has no image")
                images[camera_name] = np.zeros((*camera.get_resolution(), 3))
                continue
            sim_only_image = tmp[:, :, :3]
            semantic_seg = camera._custom_annotators["semantic_segmentation"]
            segment_data = semantic_seg.get_data()
            if not self._initialized_semantic_index:
                self._name_to_indexes = {
                    camera_name: {} for camera_name in self._cameras_name
                }
                self._name_to_indexes[camera_name] = self.initialize_semantic_index(
                    segment_data
                )
            try:
                table_mask = np.where(
                    segment_data["data"] == self._name_to_indexes[camera_name]["table"],
                    1,
                    0,
                )
            except:
                table_mask = np.zeros_like(segment_data["data"])
            ground_mask = np.where(
                segment_data["data"]
                == self._name_to_indexes[camera_name].get("ground", 0),
                1,
                0,
            )
            mix_mask = np.logical_or(table_mask, ground_mask)
            resolution = camera.get_resolution()
            fx, fy = compute_fx_fy(camera, resolution[1], resolution[0])
            cam_pose = get_relative_transform(
                get_prim_at_path(camera.prim_path),
                get_prim_at_path(self._root_prim_path),
            )
            if camera_name in self._fix_cameras:
                if self._fix_cameras[camera_name] is None:
                    gaussian_rendered_background = self.background.render(
                        cam_pose=cam_pose,
                        width=resolution[0],
                        height=resolution[1],
                        fx=fx,
                        fy=fy,
                        camera_pose_frame="isaacsim",
                    )
                    self._fix_cameras[camera_name] = gaussian_rendered_background
                else:
                    gaussian_rendered_background = self._fix_cameras[camera_name]
            else:
                gaussian_rendered_background = self.background.render(
                    cam_pose=cam_pose,
                    width=resolution[0],
                    height=resolution[1],
                    fx=fx,
                    fy=fy,
                    camera_pose_frame="isaacsim",
                )
            mixed_image = np.where(
                mix_mask[..., None], gaussian_rendered_background, sim_only_image
            )
            images[camera_name] = mixed_image
            obs["root_info"][camera_name] = {
                "table_mask": table_mask,
                "ground_mask": ground_mask,
                "mix_mask": mix_mask,
                "name_to_indexes": self._name_to_indexes[camera_name],
                "semantic_seg_data": segment_data["data"],
                "intrinsics": get_intrinsic_matrix(camera),
                "camera_world_pose": pose_from_tf_matrix(
                    get_relative_transform(
                        get_prim_at_path(camera.prim_path),
                        get_prim_at_path(self._root_prim_path),
                    )
                ),
            }
        obs["images"] = images
        # depth
        depths = {}
        for i, (camera_name, camera) in enumerate(
            zip(self._cameras_name, self._cameras)
        ):
            depth = camera.get_depth()
            if len(depth.shape) != 2:
                log.warn(f"Camera {camera_name} has no depth image")
                depths[camera_name] = np.zeros(camera.get_resolution())
                continue
            depths[camera_name] = depth
        obs["depths"] = depths
        return obs

    def get_observations_for_collecting_data(self) -> Dict[str, Any]:
        """All pose related are in robot base frame

        Returns:
            Dict[str, Any]: _description_
        """

        joint_state = self.robot.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities
        ee_matrix = get_relative_transform(
            get_prim_at_path(self._franka_prim_path + "/panda_hand"),
            get_prim_at_path(self._franka_prim_path),
        )
        robot_base_pose = get_relative_transform(
            get_prim_at_path(self._franka_prim_path),
            get_prim_at_path(self._root_prim_path),
        )
        obs = {}
        obs["robot"] = {"qpos": qpos, "qvel": qvel, "ee_pose": ee_matrix}
        obs["mask"] = {}
        render_images = {}
        fix_render_images = {}
        obs["root_info"] = {}
        sim_images = {}
        obs["mask_idToLabels"] = {}
        for i, (camera_name, camera) in enumerate(
            zip(self._cameras_name, self._cameras)
        ):
            tmp = camera.get_rgba()
            if len(tmp.shape) == 1:
                log.warn(f"Camera {camera_name} has no image")
                sim_images[camera_name] = np.zeros((*camera.get_resolution(), 3))
                continue
            sim_only_image = tmp[:, :, :3]
            sim_images[camera_name] = sim_only_image
            semantic_seg = camera._custom_annotators["semantic_segmentation"]
            segment_data = semantic_seg.get_data()
            if self._global_step_index < 10:
                obs["mask_idToLabels"][camera_name] = segment_data["info"]["idToLabels"]
            obs["mask"][camera_name] = segment_data["data"]

            # render gs
            resolution = camera.get_resolution()
            fx, fy = compute_fx_fy(camera, resolution[1], resolution[0])
            cam_pose = get_relative_transform(
                get_prim_at_path(camera.prim_path),
                get_prim_at_path(self._root_prim_path),
            )
            if camera_name in self._fix_cameras:
                gaussian_rendered_background = self.background.render(
                    cam_pose=cam_pose,
                    width=resolution[0],
                    height=resolution[1],
                    fx=fx,
                    fy=fy,
                    camera_pose_frame="isaacsim",
                )
                fix_render_images[camera_name] = gaussian_rendered_background
            else:
                gaussian_rendered_background = self.background.render(
                    cam_pose=cam_pose,
                    width=resolution[0],
                    height=resolution[1],
                    fx=fx,
                    fy=fy,
                    camera_pose_frame="isaacsim",
                )
                render_images[camera_name] = gaussian_rendered_background
        obs["render_images"] = render_images
        obs["fix_render_images"] = fix_render_images
        obs["sim_images"] = sim_images
        # depth
        if self._save_depth:
            depths = {}
            for i, (camera_name, camera) in enumerate(
                zip(self._cameras_name, self._cameras)
            ):
                depth = camera.get_depth()
                if len(depth.shape) != 2:
                    log.warn(f"Camera {camera_name} has no depth image")
                    depths[camera_name] = np.zeros(camera.get_resolution())
                    continue
                depths[camera_name] = depth
            obs["depths"] = depths

        # log random data
        obs["random_info"] = {
            "robot_base_pose": robot_base_pose,
            "picking_item_idx": self._picking_item,
            "picking_item_name": self.items[self._picking_item].name,
        }

        return obs

    def get_observations(self) -> Dict[str, Any]:
        res = {}
        render_info = None
        if self._priviliged_info:
            res["observations"] = self.get_obserations_privileged()
        else:
            render_info = self.get_observations_non_privileged()
            res["observations"] = render_info
        res["reward"] = self.get_reward()
        # res["done"] = self.is_done() or res["reward"] == 10
        res["done"] = self.is_done()
        res["info"] = {}
        if self._render:
            if render_info is None:
                render_info = self.get_observations_for_collecting_data()
            res["info"]["render"] = render_info
        res["info"]["picking_item"] = self._picking_item
        res["info"]["picking_item_idx"] = self._picking_item
        res["info"]["picking_item_name"] = self.items[self._picking_item].name
        return res

    def get_obserations_privileged(self) -> Dict[str, Any]:
        """
        takes about 1 ms for 1 env
        """
        joint_state = self.robot.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities
        ee_matrix = get_relative_transform(
            get_prim_at_path(self._franka_prim_path + "/panda_hand"),
            get_prim_at_path(self._franka_prim_path),
        )
        robot_base_pose = get_relative_transform(
            get_prim_at_path(self._franka_prim_path),
            get_prim_at_path(self._root_prim_path),
        )
        obs = {}
        obs["robot"] = {
            "qpos": qpos,
            "qvel": qvel,
            "ee_pose": ee_matrix,
            "robot_base_pose": robot_base_pose,
        }

        cube_position, cube_orientation = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(self.items[self._picking_item].object.prim_path),
                get_prim_at_path(self._root_prim_path),
            )
        )
        cube_position = np.mean(
            get_points_at_path(
                self.items[self._picking_item].object.prim_path,
                relative_frame_prim_path=self._root_prim_path,
            ),
            axis=0,
        )
        gripper_position, gripper_orientation = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(self._franka_prim_path + "/panda_hand"),
                get_prim_at_path(self._root_prim_path),
            )
        )
        target_position, target_orientation = (
            self._target_position,
            self._target_orientation,
        )

        obs["cube"] = {
            "position": np.array(cube_position),
            "orientation": np.array(cube_orientation),
        }
        obs["target"] = {
            "position": np.array(target_position),
            "orientation": np.array(target_orientation),
        }
        obs["robot"]["gripper"] = {
            "position": np.array(gripper_position),
            "orientation": np.array(gripper_orientation),
        }
        obs["other"] = {
            "cameras_pose": {},
            "camera_intrinsics": {},
            "image_width": {},
            "image_height": {},
        }
        if self._render:
            for camera_name, camera in zip(self._cameras_name, self._cameras):
                camera_pose = [
                    i.tolist()
                    for i in pose_from_tf_matrix(
                        get_relative_transform(
                            get_prim_at_path(camera.prim_path),
                            get_prim_at_path(self._root_prim_path),
                        )
                    )
                ]
                obs["other"]["cameras_pose"][camera_name] = camera_pose
                resolution = camera.get_resolution()
                obs["other"]["camera_intrinsics"][camera_name] = list(
                    compute_fx_fy(camera, *resolution)
                )
                (
                    obs["other"]["image_width"][camera_name],
                    obs["other"]["image_height"][camera_name],
                ) = resolution

        return obs

    def get_reward(self):
        return self.check_reward(self._picking_item)

    def check_reward(self, picking_item):
        if self._mile_stone[self.items[picking_item].name]["cube_at_the_target"]:
            return 10
        cube_position, _ = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(self.items[picking_item].object.prim_path),
                get_prim_at_path(self._root_prim_path),
            )
        )
        left_finger_position, _ = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(self._franka_prim_path + "/panda_leftfinger"),
                get_prim_at_path(self._root_prim_path),
            )
        )
        right_finger_position, _ = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(self._franka_prim_path + "/panda_rightfinger"),
                get_prim_at_path(self._root_prim_path),
            )
        )
        mid_finger_position = (left_finger_position + right_finger_position) / 2
        self._item_start_height = max(
            min(self._item_start_height, cube_position[2]),
            self._frames.get_frame_translation("marker")[2],
        )
        self.update_milestone(mid_finger_position, cube_position)
        if self._mile_stone[self.items[picking_item].name]["cube_at_the_target"]:
            return 10
        if self._mile_stone[self.items[picking_item].name]["catch_the_cube"]:
            return 6
        if self._mile_stone[self.items[picking_item].name]["at_the_cube"]:
            return 2
        return 0

    def pick_next_item(self):
        self._global_step_index = 0
        self._picked_item.add(self.items[self._picking_item].name)
        self._picking_item, status = self.get_picking_item()

    def update_milestone(self, mid_finger_position, cube_position):
        if not self._mile_stone[self.items[self._picking_item].name]["at_the_cube"]:
            if self._target_reached(
                mid_finger_position, cube_position, threshold=0.1, to_print=False
            ):
                self._mile_stone[self.items[self._picking_item].name][
                    "at_the_cube"
                ] = True
            return
        if not self._mile_stone[self.items[self._picking_item].name]["catch_the_cube"]:
            if cube_position[2] > self._item_start_height + 0.05 and franka_picked_up(
                self._franka_prim_path,
                self.items[self._picking_item].prim_path,
                self.bg_pcd,
                table_pcd_frame_prim_path=self._root_prim_path,
            ):
                self._mile_stone[self.items[self._picking_item].name][
                    "catch_the_cube"
                ] = True
            return
        if not self._mile_stone[self.items[self._picking_item].name][
            "cube_at_the_target"
        ]:
            panda_hand_position = pose_from_tf_matrix(
                get_relative_transform(
                    get_prim_at_path(self._franka_prim_path + "/panda_hand"),
                    get_prim_at_path(self._root_prim_path),
                )
            )[0]
            cube_position = np.mean(
                get_points_at_path(
                    self.items[self._picking_item].object.prim_path,
                    relative_frame_prim_path=self._root_prim_path,
                ),
                axis=0,
            )
            if (
                cube_position[0] > self._item_target_region[0][0]
                and cube_position[0] < self._item_target_region[0][1]
                and cube_position[1] > self._item_target_region[1][0]
                and cube_position[1] < self._item_target_region[1][1]
            ) or (
                self._target_reached(
                    panda_hand_position,
                    self._target_position
                    + self.items[self._picking_item].params["place_height_offset"],
                    to_print=False,
                )
                and sum(self.robot.gripper.get_joint_positions()) > 0.04
            ):
                self._mile_stone[self.items[self._picking_item].name][
                    "cube_at_the_target"
                ] = True
                self._item_picked_times[self.items[self._picking_item].name] += 1
            return

    def _target_reached(
        self, end_effector_position, target_position, threshold=0.02, to_print=True
    ) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if np.linalg.norm(
            np.array(end_effector_position) - np.array(target_position)
        ) < (threshold / get_stage_units()):
            return True
        else:
            if to_print:
                print(
                    "Distance: ",
                    np.linalg.norm(
                        np.array(end_effector_position) - np.array(target_position)
                    ),
                )
            return False

    @property
    def gripper(self):
        return self.robot.gripper

    @property
    def data_log_root_path(self):
        return self._data_log_root_path

    @property
    def to_render(self):
        return self._render

    @property
    def camera_names(self):
        return self._cameras_name

    @property
    def frames(self):
        return self._frames

    @property
    def robot_base_pose(self):
        return pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(self._franka_prim_path),
                get_prim_at_path(self._root_prim_path),
            )
        )

    @property
    def tmp_dir(self):
        return self._tmp_dir

    @property
    def camera_params(self):
        return {
            camera_name: camera_param
            for camera_name, camera_param in zip(
                self._cameras_name, self._camera_params
            )
        }

    @property
    def get_current_item(self) -> Item:
        return self.items[self._picking_item]

    @property
    def bg_pcd(self):
        if hasattr(self, "_bg_pcd"):
            return self._bg_pcd
        else:
            self._bg_pcd = get_points_at_path(
                self.background.prim_path, relative_frame_prim_path=self._root_prim_path
            )
            return self._bg_pcd
