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
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask as OmniBaseTask
from omni.isaac.franka.franka import Franka
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.transformations import tf_matrices_from_poses, pose_from_tf_matrix, get_relative_transform
from omni.isaac.core.utils.prims import is_prim_path_valid, create_prim, get_prim_at_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from scipy.spatial.transform import Rotation as R
import carb
import random
from pxr import UsdGeom
from collections import deque
from pxr.Usd import Prim
import omni
import os
import sys
import tempfile

sys.path.append("/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0")
from ..utils.utils import *
from ..utils.gaussian_renderer import GaussianRenderer
from ..utils.items import Item, create_camera
from ..utils.prim import get_points_at_path, franka_picked_up
from ..randomize.item_pose_random import random_place_items
import roboticstoolbox as rtb



from ..tasks.task import BaseTask
from ..envs.config import TaskUserConfig
from ..utils.logger import log
from ..utils.frame_tools import FrameTools
from ..utils.motion_control import PinocchioMotionControl
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

@BaseTask.register("EvalPickAndPlace")
class EvalPickAndPlace(BaseTask):
    def __init__(self, config: TaskUserConfig, scene: Scene):
        super().__init__(config, scene)
        # global config
        self.config = config
        config_params = config.params
        self._asset_root = config_params.get("asset_root", ".")
        self._root_prim_path = config.root_path
        
        # franka
        self._franka_cam_prim_path = config.root_path + config.robots_root_path + config_params.get("franka_cam_prim_path", "/X1")
        self._franka_prim_path = self._franka_cam_prim_path + "/Franka"
        self._franka_robot_name = f"{config_params.get('franka_robot_name', 'franka')}_{config.env_id}"
        
        # background
        self._bg_usd_path = os.path.join(self._asset_root, config_params.get("bg_usd_path", "usd/processed/green_table/green_table.usd"))
        self._bg_prim_path = config.root_path + "/background" + config_params.get("table_prim_path", "/Table")
        usd_relative_path = config_params.get("bg_usd_path", "usd/processed/green_table/green_table.usd")
        usd_relative_path = usd_relative_path.split("/")
        mesh_relative_path = os.path.join("mesh", "/".join(usd_relative_path[1:]))
        mesh_relative_path = mesh_relative_path.replace(".usd", ".obj")
        self._bg_mesh_path = os.path.join(self._asset_root, mesh_relative_path)
        self._bg_translation = np.array(config_params.get("bg_translation", [0, 0, 0]))
        self._bg_scale = np.array([1.0] * 3) * config_params.get("bg_scale", 1.0)
        if Path(self._bg_mesh_path).parent.exists():
            bg_transform_matrix = np.load(Path(self._bg_mesh_path).parent / "mesh_to_marker.npy")
        elif (Path(self._bg_usd_path).parent / "mesh_to_marker.npy").exists():
            bg_transform_matrix = np.load(Path(self._bg_usd_path).parent / "mesh_to_marker.npy")
        else:
            raise FileNotFoundError(f"No mesh_to_marker.npy found in {Path(self._bg_usd_path).parent} or {Path(self._bg_mesh_path).parent}")

        # target
        self._target_position = np.array(config_params.get("target_position", [-0.17, 0.13, 0.6]))
        self._target_orientation = np.array(config_params.get("target_orientation", euler_to_quaternion(0, 0, 0)))
        self._item_target_region = np.array(config_params.get("item_target_region", [[-0.23, 0.1], [-0.5, -0.3]]))
        # settings
        self._priviliged_info = config_params.get("priviliged_info", False)
        self._render = config_params.get("render", False)
        self._physics_dt = config_params.get("physics_dt", 1 / 60)
        
        # randomize
        # self._random_params: Dict = config_params.get("random", {})

        # frames
        self._frames = FrameTools("isaacsim")
        marker_to_isaacsim = config_params.get("marker_to_isaacsim", None)
        tmp = np.eye(4)
        tmp[2, 3] = 0.8
        marker_to_isaacsim = load_transform_matrix(marker_to_isaacsim, tmp)
        marker_to_robot_base = config_params.get("marker_to_robot_base", None)
        marker_to_robot_base = load_transform_matrix(marker_to_robot_base)
        #temp
        # marker_to_robot_base = tmp_func(marker_to_robot_base)
        robot_base_to_marker = np.linalg.inv(marker_to_robot_base)

        self._frames.add_frame_transform("marker", marker_to_isaacsim)
        self._frames.add_frame_transform_relative_to("robot_base", "marker", robot_base_to_marker)
        self._frames.add_frame_transform_relative_to("mesh_bg", "marker", bg_transform_matrix)
        self._frames.apply_scale_to("mesh_bg", self._bg_scale)
        self._frames.apply_scale_to('marker', self._bg_scale)
        self._frames.apply_translation_to("mesh_bg", self._bg_translation)
        self._frames.apply_translation_to("marker", self._bg_translation)
        gs_path = config_params.get("gs_path")
        gs_path = os.path.join(self._asset_root, gs_path)
        gs_transform_matrix = np.load(Path(gs_path) / "gs_to_marker.npy")
        self._frames.add_frame_transform_relative_to("gs", "marker", gs_transform_matrix)
        # self._frames.save("src/tests/frame.json")

        # init
        self._initialized = False
        self._initialized_semantic_index = False
        self._global_step_index = 0
        self._physics_dt = eval(self._physics_dt) if isinstance(self._physics_dt, str) else self._physics_dt
        self._totol_global_step_index = int(config_params.get("task_max_duration", 90) / self._physics_dt)
        self._item_start_height = self._frames.get_frame_translation("marker")[2]
        self._picking_item = None
        self._item_picked_times = {}
        self._task_idx = 0
        
        # delay
        self._delay = 0
        self._delay_queue = deque(maxlen=1)
        self._renderer = GaussianRenderer(gaussian_model_path=os.path.join(gs_path, "point_cloud/iteration_30000/point_cloud.ply"))
        self._tmp_dir = tempfile.mkdtemp()
        self._item_random_places = np.load(config_params.get("random_test_cases_path"))
        self._picked_item = set()
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
        create_prim(
            prim_path=self._bg_prim_path,
            usd_path=self._bg_usd_path,
            scale=self.config.scene_scale,
        )
        # create_prim(
        #     prim_path=self._franka_cam_prim_path,
        #     usd_path=franka_asset_path,
        #     scale=self.config.scene_scale,
        # )
        bg_prim = XFormPrim(self._bg_prim_path)
        # bg_prim.set_local_pose(translation=self._frames.get_frame_translation("mesh_bg") + self._bg_translation, orientation=self._frames.get_frame_rotation("mesh_bg", "quat"))
        bg_prim.set_local_pose(translation=self._frames.get_frame_translation("mesh_bg"), orientation=self._frames.get_frame_rotation("mesh_bg", "quat"))
        bg_prim.set_local_scale(scale= np.array([1.0] * 3) * self._frames.get_frame_scale("mesh_bg"))
        self._robot = self.set_robot()
        self._franka_view = ArticulationView(self._franka_prim_path)

        self._items: List[Item] = []
        self._item_name_to_index = {}
        for item_config in self.config.items:
            item = Item(frame_tools=self._frames, root_prim_path=self._root_prim_path, **item_config)
            if item.object is not None:
                self.scene.add(item.object)
            self._items.append(item)
            self._item_name_to_index[item.name] = len(self._items) - 1
            self._item_picked_times[item.name] = 0
        self._task_objects[self._robot.name] = self._robot 
        self._picking_item = None
        
        
        # # save the points cloud of background
        pcd = get_points_at_path(self._bg_prim_path, relative_frame_prim_path=self._root_prim_path)
        np.savez(os.path.join(self._tmp_dir, "pcd_bg.npz"), pcd=pcd)
        
    def set_robot(self)  -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        
        franka_translations = self._frames.get_frame_translation("robot_base")
        franka_orientations = self._frames.get_frame_rotation("robot_base", "quat")
        xform = XFormPrim(prim_path=self._franka_cam_prim_path)
        xform.set_local_pose(translation=[0., 0., 0.], orientation=[1., 0., 0., 0.])
        robot = Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name, usd_path=self.config.params.get("franka_usd_path", None))
        robot.set_local_pose(translation=franka_translations, orientation=franka_orientations)
        robot.set_solver_position_iteration_count(32)
        robot.set_solver_velocity_iteration_count(32)
        self._cspace_controller = RMPFlowController(name=self._franka_robot_name + "_cspace_controller", robot_articulation=robot)
        self._rtb_robot = rtb.models.Panda()
        self.ee_controller = PinocchioMotionControl(
            urdf_path=os.path.join(self._asset_root, "urdfs/panda/panda.urdf"),
            wrist_name="panda_hand",
            arm_init_qpos=np.array([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04]),
        )
        return robot
    
    def initialize(self):
        self._initialized = True
        self._robot.initialize()
        self._robot._articulation_view.initialize()
        self._franka_view.initialize()
        self._franka_view.set_max_joint_velocities([1.0] * 9)
        # add semantic info 
        if self._priviliged_info and not self._render:
            return
        set_semantic_label(get_prim_at_path("/World/groundPlane"), "ground")
        set_semantic_label(get_prim_at_path(self._bg_prim_path), "table")
        for item in self._items:
            set_semantic_label(item.object.prim, item.name)
        self._cameras = []
        self._cameras_name = []
        self._fix_cameras = {}
        self._camera_params = []
        for camera_config in self.config.cameras:
            if camera_config.get("wrist_camera", False):
                camera_prim_path = self._franka_prim_path + "/panda_hand" + camera_config["relative_prim"]
            else:
                camera_prim_path = self._franka_prim_path + camera_config["relative_prim"]
            camera = create_camera(camera_config["name"], camera_prim_path, camera_config["position"], camera_config["orientation"], camera_config["camera_params"])
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
            name_to_index[value['class']] = int(key)
        # assert table_index_tmp is not None and ground_indexes_tmp is not None
        return name_to_index
            
    def individual_reset(self):
        self._robot.gripper.open()
        self._global_step_index = 0
        
        self._robot.set_joint_positions([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04])
        self._rtb_robot.q = np.array([0, -0.84, 0, -2.59, 0, 1.75, 0.77])
        self._mile_stone = {item.name: {
            "at_the_cube": False,
            "catch_the_cube": False,
            "cube_at_the_target": False,
        } for item in self._items}
        self._item_start_height = self._frames.get_frame_translation("marker")[2]
        del self.ee_controller
        self.ee_controller = PinocchioMotionControl(
            urdf_path=os.path.join(self._asset_root, "urdfs/panda/panda.urdf"),
            wrist_name="panda_hand",
            arm_init_qpos=np.array([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04]),
        )
        for item in self._items:
            item.reset(align_surface=True, table_pcd=self.table_pcd, percent=100)
        
        franka_prim = XFormPrim(prim_path=self._franka_cam_prim_path)
        franka_prim.set_local_pose(translation=np.array([0.0, 0.0, 0.0]))
        self._random_items()
        self._picking_item = None
        self._picked_item = set()

    def reset_robot(self):
        self._robot.set_joint_positions([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04])
        self._rtb_robot.q = np.array([0, -0.84, 0, -2.59, 0, 1.75, 0.77])
        self.ee_controller = PinocchioMotionControl(
            urdf_path=os.path.join(self._asset_root, "urdfs/panda/panda.urdf"),
            wrist_name="panda_hand",
            arm_init_qpos=np.array([0, -0.84, 0, -2.59, 0, 1.75, 0.77, 0.04, 0.04]),
        )
        franka_prim = XFormPrim(prim_path=self._franka_cam_prim_path)
        franka_prim.set_local_pose(translation=np.array([0.0, 0.0, 0.0]))

    def pick_next_item(self):
        # self.reset_robot()
        self._global_step_index = 0
        self._picked_item.add(self._picking_item)
        self._picking_item = None

    def _random_robot_base_position(self, **params):
        franka_prim = XFormPrim(prim_path=self._franka_cam_prim_path)
        print("Robot base location:", franka_prim.get_local_pose()) # TODO: need to check whether there is init pose
        position = np.random.uniform(params['min'], params["max"])
        franka_prim.set_local_pose(translation=position)
        return position.tolist()

    def _random_items(self):
        positions = np.array(self._item_random_places[self._task_idx, :, :3])
        orientations = np.array(self._item_random_places[self._task_idx, :, 3:])
        for item, position, orientation in zip(self._items, positions[:len(self._items)], orientations[:len(self._items)]):
            item.xform.set_local_pose(translation=position, orientation=orientation)
        return positions.tolist(), orientations.tolist()
    
    def is_done(self) -> bool:
        """[summary]"""
        joint_state = self._robot.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities
        # cube_position, _ = pose_from_tf_matrix(get_relative_transform(get_prim_at_path(self._items[self._picking_item].object.prim_path), get_prim_at_path(self._root_prim_path)))
        if np.any(np.isnan(qpos)) or np.any(np.isnan(qvel)):
            return True
        # timeout
        if self._global_step_index > self._totol_global_step_index:
            return True
        return False
    
    def get_left_item_num(self, return_index=False):
        i = 0
        index = 0
        for item in self._items:
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
                gripper_joints = self._robot.gripper.get_joint_positions()
                franka_action = np.concatenate([franka_action, gripper_joints])
            elif franka_action.shape[-1] == 9:
                pass
            else:
                raise ValueError("Invalid action shape")
            if type(franka_action) != np.ndarray:
                print("!")
            self._franka_view.set_joint_position_targets(franka_action)
            # self._robot.gripper.apply_action(ArticulationAction(joint_positions=franka_action[-2:], ))
        elif action_type == "ee":
            
            gripper_width, transform = franka_action
            pos, rot_mat = transform[:3, 3], transform[:3, :3]
            sol = self.ee_controller.control(pos, rot_mat)[:7]
            qpos = np.concatenate([sol, [gripper_width / 2], [gripper_width / 2]])
            self._rtb_robot.q = sol
            # position, orientation = pose_from_tf_matrix(transform)
            # qpos = self._cspace_controller.forward(target_end_effector_position=position, target_end_effector_orientation=orientation)
            # qpos = np.concatenate([qpos.joint_positions.reshape(-1), [gripper_width / 2], [gripper_width / 2]])
            self._franka_view.set_joint_position_targets(qpos)
         
    def get_observations_non_privileged(self) -> Dict[str, Any]:
        """All pose related are in robot base frame

        Returns:
            Dict[str, Any]: _description_
        """
        joint_state = self._robot.get_joints_state()
        qpos, qvel = joint_state.positions, joint_state.velocities
        ee_matrix = get_relative_transform(get_prim_at_path(self._franka_prim_path + "/panda_hand"), get_prim_at_path(self._franka_prim_path))
        robot_base_pose = get_relative_transform(get_prim_at_path(self._franka_prim_path), get_prim_at_path(self._root_prim_path))  
        obs = {}
        obs["robot"] = {"qpos": qpos, "qvel": qvel, "ee_pose": ee_matrix, "robot_base_pose": robot_base_pose}

        images = {}
        obs["root_info"] = {}
        for i, (camera_name, camera) in enumerate(zip(self._cameras_name, self._cameras)):    
            tmp = camera.get_rgba()
            if len(tmp.shape) == 1:
                log.warn(f"Camera {camera_name} has no image")
                images[camera_name] = np.zeros((*camera.get_resolution(), 3))
                continue       
            sim_only_image = tmp[:, :, :3]
            semantic_seg = camera._custom_annotators["semantic_segmentation"]
            segment_data = semantic_seg.get_data()
            if not self._initialized_semantic_index:
                self._name_to_indexes = {camera_name: {} for camera_name in self._cameras_name}
                self._name_to_indexes[camera_name] = self.initialize_semantic_index(segment_data)
            try:
                table_mask = np.where(segment_data["data"] == self._name_to_indexes[camera_name]["table"], 1, 0)
            except:
                table_mask = np.zeros_like(segment_data["data"])
            ground_mask = np.where(segment_data["data"] == self._name_to_indexes[camera_name].get("ground", 0), 1, 0)
            mix_mask = np.logical_or(table_mask, ground_mask)
            resolution = camera.get_resolution()
            fx, fy = compute_fx_fy(camera, resolution[1], resolution[0])
            cam_pose = get_relative_transform(get_prim_at_path(camera.prim_path), get_prim_at_path(self._root_prim_path))
            cam_pose = self._frames.get_frame_transform_relative_to('isaacsim', "gs") @ cam_pose
            rotation_matrix = cam_pose[:3, :3] / np.abs((np.linalg.det(cam_pose[:3, :3])) ** (1 / 3))
            translation = cam_pose[:3, 3]
            x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
            cam_pose = [translation, [w, x, y, z]]
            if camera_name in self._fix_cameras:
                if self._fix_cameras[camera_name] is None:
                    gaussian_rendered_background = self._renderer.render(
                        cam_pose=cam_pose,
                        width=resolution[0],
                        height=resolution[1],
                        fx=fx,
                        fy=fy
                    )
                    self._fix_cameras[camera_name] = gaussian_rendered_background
                else:
                    gaussian_rendered_background = self._fix_cameras[camera_name]
            else:
                gaussian_rendered_background = self._renderer.render(
                        cam_pose=cam_pose,
                        width=resolution[0],
                        height=resolution[1],
                        fx=fx,
                        fy=fy
                    )
            mixed_image = np.where(mix_mask[..., None], gaussian_rendered_background, sim_only_image)
            images[camera_name] = mixed_image
            obs["root_info"][camera_name] = {"table_mask": table_mask, "ground_mask": ground_mask, "mix_mask": mix_mask, "name_to_indexes": self._name_to_indexes[camera_name], "semantic_seg_data" :segment_data["data"], "intrinsics": get_intrinsic_matrix(camera), "camera_world_pose": pose_from_tf_matrix(get_relative_transform(get_prim_at_path(camera.prim_path), get_prim_at_path(self._root_prim_path)))}
        obs["images"] = images
        # return obs
        # depth 
        depths = {}
        for i, (camera_name, camera) in enumerate(zip(self._cameras_name, self._cameras)):
            depth = camera.get_depth()
            if len(depth.shape) != 2:
                log.warn(f"Camera {camera_name} has no depth image")
                depths[camera_name] = np.zeros(camera.get_resolution())
                continue
            depths[camera_name] = depth
        obs["depths"] = depths
        return obs         
   

    def get_observations(self) -> Dict[str, Any]:
        res = {}
        res["observations"] = self.get_observations_non_privileged()
        res["reward"] = self.get_reward()
        res["done"] = self.is_done()
        res["info"] = {}
        res["info"]["picking_item"] = self._picking_item
        return res
        
    def get_reward(self):
        if self._picking_item is None:
            max_reward = 0
            item_index = 0
            for item in self._items:
                if item.name not in self._picked_item:
                    reward = self.check_reward(item.name)
                    if reward > max_reward:
                        max_reward = reward
                        item_index = self._items.index(item)
            if max_reward >=6:
                self._picking_item = self._items[item_index].name
            return max_reward
        else:
            return self.check_reward(self._picking_item)
        
    def check_reward(self, picking_item):
        if self._mile_stone[picking_item]['cube_at_the_target']:
            return 10
        cube_position, _ = pose_from_tf_matrix(get_relative_transform(get_prim_at_path(self._items[self._item_name_to_index[picking_item]].object.prim_path), get_prim_at_path(self._root_prim_path)))
        left_finger_position, _ = pose_from_tf_matrix(get_relative_transform(get_prim_at_path(self._franka_prim_path + "/panda_leftfinger"), get_prim_at_path(self._root_prim_path)))
        right_finger_position, _ = pose_from_tf_matrix(get_relative_transform(get_prim_at_path(self._franka_prim_path + "/panda_rightfinger"), get_prim_at_path(self._root_prim_path)))
        mid_finger_position = (left_finger_position + right_finger_position) / 2
        self._item_start_height = max(min(self._item_start_height, cube_position[2]), self._frames.get_frame_translation("marker")[2])
        self.update_milestone(mid_finger_position, cube_position, picking_item)
        if self._mile_stone[picking_item]['cube_at_the_target']:
            return 10
        if self._mile_stone[picking_item]['catch_the_cube']:
            return 6
        if self._mile_stone[picking_item]['at_the_cube']:
            return 2
        return 0

    def update_milestone(self, mid_finger_position, cube_position, picking_item):
        if not self._mile_stone[picking_item]["at_the_cube"]:
            if self._target_reached(mid_finger_position, cube_position, threshold=0.1, to_print=False):
                self._mile_stone[picking_item]["at_the_cube"] = True
            return
        if not self._mile_stone[picking_item]["catch_the_cube"]:
            if cube_position[2] > self._item_start_height + 0.05 and franka_picked_up(self._franka_prim_path, self._items[self._item_name_to_index[picking_item]].prim_path, self.table_pcd, threshold=0.1, table_pcd_frame_prim_path=self._root_prim_path):
                self._mile_stone[picking_item]["catch_the_cube"] = True
            return
        if not self._mile_stone[picking_item]["cube_at_the_target"]:
            panda_hand_position = pose_from_tf_matrix(get_relative_transform(get_prim_at_path(self._franka_prim_path + "/panda_hand"), get_prim_at_path(self._root_prim_path)))[0]
            cube_position = np.mean(get_points_at_path(self._items[self._item_name_to_index[picking_item]].object.prim_path, relative_frame_prim_path=self._root_prim_path), axis=0)
            if (cube_position[0] > self._item_target_region[0][0] and cube_position[0] < self._item_target_region[0][1] and cube_position[1] > self._item_target_region[1][0] and cube_position[1] < self._item_target_region[1][1]) or (self._target_reached(panda_hand_position, self._target_position + self._items[self._item_name_to_index[picking_item]].params["place_height_offset"], to_print=False) and sum(self._robot.gripper.get_joint_positions()) > 0.04):
                self._mile_stone[picking_item]["cube_at_the_target"] = True
                self._item_picked_times[picking_item] += 1
            return
    def _target_reached(self, end_effector_position, target_position, threshold=0.02, to_print=True) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if np.linalg.norm(np.array(end_effector_position) - np.array(target_position)) < (threshold / get_stage_units()):
            return True
        else:
            if to_print:
                print("Distance: ", np.linalg.norm(np.array(end_effector_position) - np.array(target_position)))
            return False
        
    @property
    def gripper(self):
        return self._robot.gripper
    
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
        return pose_from_tf_matrix(get_relative_transform(get_prim_at_path(self._franka_prim_path), get_prim_at_path(self._root_prim_path)))
    
    @property
    def tmp_dir(self):
        return self._tmp_dir

    @property
    def camera_params(self):
        return {camera_name: camera_param for camera_name, camera_param in zip(self._cameras_name, self._camera_params)}

    @property
    def get_current_item(self) -> Item:
        return self._items[self._item_name_to_index[self._picking_item]] if self._picking_item is not None else None
    
    @property
    def table_pcd(self):
        if hasattr(self, "_table_pcd"):
            return self._table_pcd
        else:
            self._table_pcd = get_points_at_path(self._bg_prim_path, relative_frame_prim_path=self._root_prim_path)
            return self._table_pcd


