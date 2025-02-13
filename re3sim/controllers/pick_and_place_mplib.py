import typing

import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.utils.prims import get_prim_at_path
from ..utils.prim import recursive_parse
import mplib
from typing import Tuple, List, Union
from ..utils.utils import get_anygrasp_pose
from ..utils.prim import get_points_at_path, franka_picked_up
from ..tasks import BaseTask
from scipy.spatial.transform import Rotation as R
from ..utils.pcd_utils import point_cloud_to_mesh
from omni.isaac.core.utils.transformations import (
    tf_matrix_from_pose,
    pose_from_tf_matrix,
    get_relative_transform,
)
from omni.isaac.sensor import Camera
from mplib.collision_detection.fcl import (
    Triangle,
    BVHModel,
    load_mesh_as_Convex,
    CollisionObject,
)
import threading
import trimesh
import os
import roboticstoolbox as rtb


def joint_positions_to_ee_pose_translation_euler(
    joint_positions: np.ndarray, rtb_robot: rtb.models.Panda
):
    """
    Args:
        joint_positions (np.ndarray): joint positions of the robot
        rtb_robot (rtb.models.Panda): the robot model

    Returns:
        np.ndarray: the translation and euler angles of the end effector (x, y, z, rx, ry, rz)
    """
    return np.concatenate(
        [
            rtb_robot.fkine(joint_positions[:7]).A[:3, 3],
            R.from_matrix(rtb_robot.fkine(joint_positions[:7]).A[:3, :3]).as_euler(
                "xyz", degrees=True
            ),
        ]
    )


class PickAndPlaceController(BaseController):
    def __init__(
        self,
        name: str,
        task: BaseTask,
        urdf="/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/franka/panda/panda.urdf",
        srdf="/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/franka/panda/panda.srdf",
        pre_pick_position_rela: np.ndarray = np.array((0, 0, 0.04)),
        post_position_rela: np.ndarray = np.array((0, 0, 0.1)),
        dt: float = None,
        bg_collision_mesh_offset: float = 0.0,  # for mplib only
        joint_vel_limits=None,
        extra_controller_collisions: List[str] = [],
        **kwargs,
    ) -> None:
        BaseController.__init__(self, name=name)
        self.task = task
        self.items = task.items
        self._gripper_joint_opened_positions = task.robot.gripper.joint_opened_positions
        self._gripper_joint_closed_positions = task.robot.gripper.joint_closed_positions
        self._last_plan_world_time = 0.0
        self._dt = dt
        self._change_phase_time = 0.0
        self._anygrasp_url = "http://10.6.8.89:5001/process"
        if hasattr(task, "camera_params") and "wrist_camera" in task.camera_params:
            self._hand_camera_prim_path = (
                task.robot.prim_path
                + "/panda_hand"
                + "/wrist_camera"
                + "/camera/camera"
            )
        else:
            self._hand_camera_prim_path = None
        base_pose = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(task.robot.prim_path),
                get_prim_at_path(task._root_prim_path),
            )
        )
        self._root_prim_path = task._root_prim_path
        self._franka_prim_path = task.robot.prim_path
        self._bg_collision_mesh_offset = bg_collision_mesh_offset
        self._extra_controller_collisions = extra_controller_collisions
        self._joint_vel_limits = joint_vel_limits
        # collision mesh
        self.pcd = task.bg_pcd
        self.rtb_robot = rtb.models.Panda()
        # planner
        self.planner = mplib.Planner(
            urdf=urdf,
            srdf=srdf,
            move_group="panda_hand",
            joint_vel_limits=self._joint_vel_limits,
        )

        self.set_base_pose(base_pose, remove_collision_mesh=False)
        size = np.array([0.01, 0.01, 0.01]).reshape(3, 1)
        link_prim_path = self._franka_prim_path + "/panda_hand"
        if self._hand_camera_prim_path:
            pose = pose_from_tf_matrix(
                get_relative_transform(
                    get_prim_at_path(link_prim_path),
                    get_prim_at_path(self._hand_camera_prim_path),
                )
            )
            self.planner.update_attached_box(
                size=size, pose=mplib.Pose(p=pose[0], q=pose[1])
            )

        # init
        self.plan_results = None
        self.cmd_idx = 0
        self._post_position_rela = post_position_rela
        self._pre_pick_position_rela = pre_pick_position_rela
        self.milestone = {
            "reach_pre_pick_position": False,
            "at_the_cube": False,
            "cube_picked": False,
            "reach_post_position": False,
            "at_the_target": False,
            "cube_placed": False,
        }
        self._gripper_open_now = True
        self._item_idx = 0
        self._reset_needed = False

    def set_base_pose(
        self,
        base_pose: Tuple[np.ndarray, np.ndarray] = None,
        remove_collision_mesh: bool = True,
    ):
        if base_pose is None:
            base_pose = pose_from_tf_matrix(
                get_relative_transform(
                    get_prim_at_path(self.task.robot.prim_path),
                    get_prim_at_path(self.task._root_prim_path),
                )
            )
        self.planner.set_base_pose(mplib.Pose(p=base_pose[0], q=base_pose[1]))

        points = np.array(self.pcd)

        cube_size = 0.2
        base_center = base_pose[0]
        filtered_points = []
        for point in points:
            if (
                abs(point[0] - base_center[0]) > cube_size
                or abs(point[1] - base_center[1]) > cube_size
                or abs(point[2] - base_center[2]) > cube_size
            ):
                filtered_points.append(point)
        points = np.array(filtered_points) - np.array(
            [0, 0, self._bg_collision_mesh_offset]
        )
        for collision_mesh_path in self._extra_controller_collisions:
            collision_mesh_points = np.load(collision_mesh_path)
            points = np.concatenate([points, collision_mesh_points])

        if remove_collision_mesh:
            self.planner.remove_point_cloud()
        self.planner.update_point_cloud(points, resolution=0.02)

    def forward(
        self,
        picking_position: np.ndarray,
        picking_orientation: np.ndarray,
        placing_position: np.ndarray,
        placing_orientation: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: np.ndarray | None = None,
        **extra_params,
    ) -> np.ndarray:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            picking_orientation (np.ndarray): The object's orientation to be picked in local frame.
            placing_position (np.ndarray): The object's position to be placed in local frame.
            placing_orientation (np.ndarray): The object's orientation to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): Offset of the end effector target. Defaults to None.
            **extra_params: Additional parameters like rgb, depth images and camera info.

        Returns:
            Tuple[np.ndarray, dict]: A tuple containing:
                - np.ndarray: Joint positions to be executed by the ArticulationController
                - dict: Additional info about the controller status and metrics
        """
        current_time = extra_params.get("current_time", None)
        if self._reset_needed:
            self._reset_needed = False
            return -1, {"status": "item falling"}
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        target_joint_positions = current_joint_positions
        if self.is_done():
            return current_joint_positions, {"status": "done"}
        if not self.plan_results:
            rgb, depth, info_for_anygrasp = (
                extra_params.get("rgb", None),
                extra_params.get("depth", None),
                extra_params.get("info", None),
            )
            pose, use_motion_planner = self.get_target_pose(
                picking_position,
                picking_orientation,
                placing_position,
                placing_orientation,
                end_effector_offset,
                rgb,
                depth,
                info_for_anygrasp,
            )
            if use_motion_planner:
                if isinstance(pose, list):
                    threads = []
                    for i, p in enumerate(pose):
                        status_code, result = self.plan_motion(
                            p, current_joint_positions
                        )
                        if status_code == 0 and i != 5:
                            self.last_target_position = p.p
                            break
                else:
                    for i in range(20):
                        status_code, result = self.plan_motion(
                            mplib.Pose(p=pose.p + np.array([0, 0, 0.01]) * i, q=pose.q),
                            current_joint_positions,
                        )
                        if status_code == 0:
                            self.last_target_position = (
                                pose.p + np.array([0, 0, 0.01]) * i
                            )
                            print(f"plan success after {i} times")
                            break
                if status_code != 0:
                    return -1, {"status": "IK failed", "result": result}
                else:
                    self.plan_results = result
                    self.cmd_idx = 0
            self._last_plan_world_time = current_time
        if self.plan_results:
            ee_position, ee_orientation = pose_from_tf_matrix(
                get_relative_transform(
                    get_prim_at_path(self._franka_prim_path + "/panda_hand"),
                    get_prim_at_path(self._root_prim_path),
                )
            )
            target_joint_positions = self.excute_plan(
                current_joint_positions,
                ee_position,
                picking_position,
                placing_position,
                current_time,
            )
        abs_ee_action = joint_positions_to_ee_pose_translation_euler(
            target_joint_positions, self.rtb_robot
        )
        delta_ee_action = abs_ee_action - joint_positions_to_ee_pose_translation_euler(
            current_joint_positions, self.rtb_robot
        )
        desired_velocity = delta_ee_action[:3] / self._dt
        desired_angular_velocity = np.deg2rad(delta_ee_action[3:]) / self._dt
        gripper_action = np.sum(target_joint_positions[-2:]) * 0.8
        gripper_velocity = (
            gripper_action - np.sum(current_joint_positions[-2:])
        ) / self._dt
        info = {
            "abs_ee_action": abs_ee_action,
            "delta_ee_action": delta_ee_action,
            "desired_velocity": desired_velocity,
            "desired_angular_velocity": desired_angular_velocity,
            "gripper_action": gripper_action,
            "gripper_velocity": gripper_velocity,
        }
        replan = (
            np.linalg.norm(info["desired_velocity"]) < 0.005
            and np.linalg.norm(info["gripper_velocity"]) < 1e-4
        )
        info["replan"] = replan
        if replan and hasattr(self, "_last_plan_world_time"):
            self._last_plan_world_time -= self._dt
        info["status"] = "ok" if not self.is_done() else "done"
        return target_joint_positions, info

    def excute_plan(
        self,
        current_joint_positions,
        current_gripper_position,
        cube_position,
        placing_position,
        current_time,
    ):
        target_joint_positions = current_joint_positions
        if isinstance(self.plan_results, str):
            if self.plan_results == "Open gripper":
                self._gripper_open_now = True
                gripper_action = np.sum(self._gripper_joint_opened_positions) * 0.8
                gripper_velocity = (
                    gripper_action - np.sum(current_joint_positions[-2:])
                ) / self._dt
                # if gripper_velocity < 1e-4:
                if self.cmd_idx <= 3:
                    gripper_position = self._gripper_joint_opened_positions
                    joint_points = np.concatenate(
                        [current_joint_positions[:-2], gripper_position]
                    )
                    target_joint_positions = joint_points
                    self.cmd_idx += 1
                else:
                    self.plan_results = None
                    self.cmd_idx = 0
                    self.update_milestone(
                        current_gripper_position, cube_position, placing_position
                    )
            elif self.plan_results == "Close gripper":
                self._gripper_open_now = False
                gripper_action = np.sum(self._gripper_joint_closed_positions) * 0.8
                gripper_velocity = (
                    gripper_action - np.sum(current_joint_positions[-2:])
                ) / self._dt
                # if gripper_velocity < 1e-4:
                if self.cmd_idx <= 3:
                    gripper_position = self._gripper_joint_closed_positions
                    joint_points = np.concatenate(
                        [current_joint_positions[:-2], gripper_position]
                    )
                    target_joint_positions = joint_points
                    self.cmd_idx += 1
                else:
                    self.plan_results = None
                    self.cmd_idx = 0
                    self.update_milestone(
                        current_gripper_position, cube_position, placing_position
                    )
            else:
                raise NotImplementedError("Invalid str plan results")
        else:
            target_joint_positions = self.excute_plan_from_motion_planner(
                current_joint_positions, current_time
            )
            self.update_milestone(
                current_gripper_position, cube_position, placing_position
            )
        return target_joint_positions

    def excute_plan_from_motion_planner(self, current_joint_positions, current_time):
        if self.plan_results["status"] == "Success" and self.cmd_idx < len(
            self.plan_results["position"]
        ):
            joint_points = self.plan_results["position"][self.cmd_idx]
            joint_velocities = self.plan_results["velocity"][self.cmd_idx]
            gripper_velocity = np.zeros(2)
            gripper_position = self.get_gripper_position()
            joint_points = np.concatenate([joint_points, gripper_position])
            joint_velocities = np.concatenate([joint_velocities, gripper_velocity])
            target_joint_positions = joint_points
            while (
                self.cmd_idx < len(self.plan_results["time"])
                and self.plan_results["time"][self.cmd_idx] + self._last_plan_world_time
                < current_time
            ):
                self.cmd_idx += 1
        if (
            self.cmd_idx >= len(self.plan_results["position"])
            and len(self.plan_results["time"]) > 0
            and current_time
            - self._last_plan_world_time
            - self.plan_results["time"][-1]
            < self._change_phase_time
        ):
            # self.cmd_idx = 0
            self.cmd_idx += 1
            joint_points = np.concatenate(
                [current_joint_positions[:-2], self.get_gripper_position()]
            )
            target_joint_positions = joint_points
        elif (
            len(self.plan_results["time"]) > 0
            and current_time
            - self._last_plan_world_time
            - self.plan_results["time"][-1]
            >= self._change_phase_time
        ):
            self.cmd_idx = 0
            self.plan_results = None
            joint_points = np.concatenate(
                [current_joint_positions[:-2], self.get_gripper_position()]
            )
            target_joint_positions = joint_points
        elif len(self.plan_results["position"]) == 0:
            self.plan_results = None
            self.cmd_idx = 0
            joint_points = np.concatenate(
                [current_joint_positions[:-2], self.get_gripper_position()]
            )
            target_joint_positions = joint_points
        return target_joint_positions

    def update_milestone(self, gripper_position, cube_position, placing_position):
        if not self.is_done():
            if not self.milestone["reach_pre_pick_position"]:
                if self.target_reached(
                    gripper_position, self.pre_pick_position, to_print=False
                ):
                    self.milestone["reach_pre_pick_position"] = True
                return
            if (
                not self.target_reached(
                    gripper_position, cube_position, threshold=0.15, to_print=False
                )
                and self.milestone["at_the_cube"]
                and not self.milestone["at_the_target"]
            ):
                self.plan_results = None
                self._gripper_open_now = True
                self.milestone["reach_pre_pick_position"] = False
                self.milestone["at_the_cube"] = False
                self.milestone["cube_picked"] = False
                self.milestone["reach_post_position"] = False
                self.milestone["at_the_target"] = False
                self.detach_item(self._item_idx)
                self._reset_needed = True
            else:
                if not self.milestone["at_the_cube"]:
                    if not self.plan_results and franka_picked_up(
                        self._franka_prim_path, self.items[self._item_idx].prim_path
                    ):
                        self.milestone["at_the_cube"] = True
                    return
                elif not self.milestone["cube_picked"]:
                    self.milestone["cube_picked"] = True
                    self.attach_item(self._item_idx)
                    return
                elif not self.milestone["reach_post_position"]:
                    if hasattr(self, "last_target_position"):
                        if not self.plan_results and self.target_reached(
                            gripper_position, self.last_target_position
                        ):
                            self.milestone["reach_post_position"] = True
                    return
                elif not self.milestone["at_the_target"]:
                    if not self.plan_results and self.target_reached(
                        gripper_position, self.last_target_position
                    ):
                        self.milestone["at_the_target"] = True
                    return
                elif not self.milestone["cube_placed"]:
                    if self._gripper_open_now:
                        self.milestone["cube_placed"] = True
                    return

    def compute_picking_pose(
        self,
        item_position: np.ndarray,
        item_orientation: np.ndarray,
        avoidance_axises: List[np.ndarray | List[float]],
        offset: np.ndarray = np.array([0, 0, 0.115]),
        anygrasp=False,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        info: dict = None,
    ):
        """

        Args:
            item_position (np.ndarray): in meters
            item_orientation (np.ndarray): in quaternion [w, x, y, z]
            avoidance_axises (List[np.ndarray  |  List[float]]): in vector format
            offset (np.ndarray, optional): in meters. Defaults to np.array([0, 0, 0.115]). avoidance axis 0 is x axis, avoidance axis 1 is y axis
            anygrasp (bool, optional): whether to use anygrasp to compute the pose. Defaults to False.
        Returns:
            mplib.Pose: the target pose of the robot for the current event
        """
        from ..utils.pcd_utils import render_point_cloud_to_image
        import pickle

        if anygrasp:
            online = True
            if online:
                int_matrix = info["intrinsics"]
                fx, fy = float(int_matrix[0, 0]), float(int_matrix[1, 1])
                cx, cy = float(int_matrix[0, 2]), float(int_matrix[1, 2])
                camera_world_pose = info["camera_world_pose"]
                grasp_list = get_anygrasp_pose(
                    rgb, depth, (fx, fy, cx, cy), camera_pose=camera_world_pose
                )
                grasp_pose = grasp_list[0]
                return mplib.Pose(
                    p=grasp_pose["translation"], q=grasp_pose["orientation"]
                )
            else:
                with open(
                    "/isaac-sim/src/tests/test-anygrasp/res_test_anygrasp.pkl", "rb"
                ) as f:
                    res = pickle.load(f)
                points = res["points"]
                grasp_lists = res["grasp_lists"]
                item_transform = get_relative_transform(
                    get_prim_at_path(self.items[self._item_idx].object.prim_path),
                    get_prim_at_path(self._root_prim_path),
                )
                tmp_pose_list = [g[0] for g in grasp_lists]
                pose_list = []
                for pose in tmp_pose_list:
                    pose_transformation = tf_matrix_from_pose(
                        translation=pose["translation"], orientation=pose["orientation"]
                    )
                    pose_transformation = item_transform @ pose_transformation
                    pos, quat = pose_from_tf_matrix(pose_transformation)
                    pose_list.append(mplib.Pose(p=pos, q=quat))
                return pose_list
        else:
            assert len(avoidance_axises) <= 2, "Avoidance axises should be less than 2"
            if len(avoidance_axises) == 0:
                pose = mplib.Pose(
                    p=[
                        item_position[0] + offset[0],
                        item_position[1] - offset[1],
                        item_position[2] - offset[2],
                    ],
                    q=[0, 0, 1, 0],
                )
                return pose
            elif len(avoidance_axises) == 1:
                axis = avoidance_axises[0]
                axis_in_isaacsim = (
                    R.from_quat(item_orientation, scalar_first=True).as_matrix() @ axis
                )
                axis_in_isaacsim = axis_in_isaacsim / np.linalg.norm(axis_in_isaacsim)
                normal_axis = np.cross(axis_in_isaacsim, [0, 0, 1])
                normal_axis = normal_axis / np.linalg.norm(normal_axis)
                picking_axis = np.cross(normal_axis, axis_in_isaacsim)
                picking_axis = picking_axis / np.linalg.norm(picking_axis)
                ee_rotation = np.concatenate(
                    [
                        axis_in_isaacsim.reshape(-1, 1),
                        normal_axis.reshape(-1, 1),
                        -picking_axis.reshape(-1, 1),
                    ],
                    axis=1,
                )
                ee_position = item_position + ee_rotation @ offset
                return mplib.Pose(
                    p=ee_position,
                    q=R.from_matrix(ee_rotation).as_quat(scalar_first=True),
                )
            elif len(avoidance_axises) == 2:
                axis1 = avoidance_axises[0]
                axis2 = avoidance_axises[1]
                axis1_in_isaacsim = (
                    R.from_quat(item_orientation, scalar_first=True).as_matrix() @ axis1
                )
                axis1_in_isaacsim = axis1_in_isaacsim / np.linalg.norm(
                    axis1_in_isaacsim
                )
                axis2_in_isaacsim = (
                    R.from_quat(item_orientation, scalar_first=True).as_matrix() @ axis2
                )
                axis2_in_isaacsim = axis2_in_isaacsim / np.linalg.norm(
                    axis2_in_isaacsim
                )
                normal_axis = np.cross(axis1_in_isaacsim, axis2_in_isaacsim)
                normal_axis = normal_axis / np.linalg.norm(normal_axis)
                ee_rotation = np.concatenate(
                    [
                        axis1_in_isaacsim.reshape(-1, 1),
                        axis2_in_isaacsim.reshape(-1, 1),
                        normal_axis.reshape(-1, 1),
                    ],
                    axis=1,
                )
                ee_position = item_position + ee_rotation @ offset
                return mplib.Pose(
                    p=ee_position,
                    q=R.from_matrix(ee_rotation).as_quat(scalar_first=True),
                )

    def get_target_pose(
        self,
        picking_position: np.ndarray,
        picking_orientation: np.ndarray,
        placing_position: np.ndarray,
        placing_orientation: np.ndarray,
        end_effector_offset: np.ndarray | None = None,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        info: dict = None,
    ):
        """
        Returns: Tuple[mplib.Pose, bool, dict]
        mplib.Pose:
            the target pose of the robot for the current event
        bool:
            whether to use motion planner
        dict:
            other info
        """
        self.cmd_idx = 0
        if not self.milestone["reach_pre_pick_position"]:
            pose = self.compute_picking_pose(
                picking_position,
                picking_orientation,
                avoidance_axises=np.array(
                    self.items[self._item_idx].params["avoidance_axises"]
                ),
                offset=np.array(
                    self.items[self._item_idx].params["pick_height_offset"]
                ),
            )
            self.pre_pick_position = (
                np.array([pose.p[0], pose.p[1], pose.p[2]])
                + self._pre_pick_position_rela
            )
            pose = mplib.Pose(p=self.pre_pick_position, q=pose.q)
            return pose, True
        if not self.milestone["at_the_cube"]:
            pose = self.compute_picking_pose(
                picking_position,
                picking_orientation,
                avoidance_axises=np.array(
                    self.items[self._item_idx].params["avoidance_axises"]
                ),
                offset=np.array(
                    self.items[self._item_idx].params["pick_height_offset"]
                ),
                anygrasp=False,
                rgb=rgb,
                depth=depth,
                info=info,
            )
            return pose, True
        if not self.milestone["cube_picked"]:
            self.plan_results = "Close gripper"
            return None, False
        if not self.milestone["reach_post_position"]:
            pose = self.compute_picking_pose(
                picking_position,
                picking_orientation,
                avoidance_axises=np.array(
                    self.items[self._item_idx].params["avoidance_axises"]
                ),
                offset=np.array(
                    self.items[self._item_idx].params["pick_height_offset"]
                ),
            )
            post_pick_position = (
                np.array([pose.p[0], pose.p[1], pose.p[2]]) + self._post_position_rela
            )
            pose = mplib.Pose(p=post_pick_position, q=placing_orientation)
            return pose, True
        if not self.milestone["at_the_target"]:
            place_height_offset = self.items[self._item_idx].params[
                "place_height_offset"
            ]
            pose = mplib.Pose(
                p=[
                    placing_position[0] + place_height_offset[0],
                    placing_position[1] + place_height_offset[1],
                    placing_position[2] + place_height_offset[2],
                ],
                q=placing_orientation,
            )
            return pose, True
        if not self.milestone["cube_placed"]:
            self.plan_results = "Open gripper"
            return None, False

    def plan_motion(self, target_pose: mplib.Pose, current_joint_positions: np.ndarray):
        try:
            result = self.planner.plan_pose(
                target_pose, current_joint_positions, time_step=self._dt
            )
        except Exception as e:
            result = {"status": "Failed: " + str(e)}
        if result["status"] != "Success":
            print(result["status"])
            return -1, None
        else:
            return 0, result

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """

        self.plan_results = None
        self.cmd_idx = 0
        self._gripper_open_now = True
        self.milestone = {
            "at_the_cube": False,
            "reach_pre_pick_position": False,
            "cube_picked": False,
            "reach_post_position": False,
            "at_the_target": False,
            "cube_placed": False,
        }
        self._last_plan_world_time = 0.0
        BaseController.reset(self)

    def get_gripper_position(self):
        if self._gripper_open_now:
            return self._gripper_joint_opened_positions
        else:
            return self._gripper_joint_closed_positions

    def get_cube_point_cloud(self):
        prim = get_prim_at_path(self._cube_prim_path)
        (
            new_points,
            faceuv_total,
            normals_total,
            faceVertexCounts_total,
            faceVertexIndices_total,
            mesh_total,
        ) = recursive_parse(prim)

        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        return self.milestone["cube_placed"]

    def success(self, cube_position, target_position) -> bool:
        return self.target_reached(cube_position, target_position)

    def target_reached(
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

    def attach_item(self, item_idx):
        pcd = get_points_at_path(self.items[item_idx].prim_path)
        mesh = point_cloud_to_mesh(pcd)
        link_prim_path = self._franka_prim_path + "/panda_hand"
        pose = pose_from_tf_matrix(
            get_relative_transform(
                get_prim_at_path(link_prim_path),
                get_prim_at_path(self.items[item_idx].prim_path),
            )
        )
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertices_fcl = [np.reshape(v, (3, 1)) for v in vertices]  
        faces_fcl = [Triangle(*face) for face in faces] 
        bvh_model = BVHModel()
        bvh_model.begin_model(len(faces_fcl), len(vertices_fcl))
        bvh_model.add_sub_model(vertices_fcl, faces_fcl)
        bvh_model.end_model()
        pose = mplib.Pose(p=pose[0], q=pose[1])
        self.planner.update_attached_object(bvh_model, pose, name=f"item_{item_idx}")

    def detach_item(self, item_idx):
        self.planner.detach_object(f"item_{item_idx}", also_remove=True)
