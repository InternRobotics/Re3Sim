from panda_py import libfranka
import panda_py
from panda_py import controllers
import time
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R
import traceback

from src.frankapy.src.realsense_reader import MultiRealSenseCamera
from src.utils.motion_control import PinocchioMotionControl
from multiprocessing import Value, Pipe, Queue, Process, Manager
import threading


def grasp(gripper: libfranka.Gripper, dict_data: dict):
    while True:
        gripper_width = dict_data["gripper_width"]
        end = dict_data.get("end", False)
        if end:
            break
        print(f"Gripper width in thread: {gripper_width}")
        if gripper_width > 0.04:
            gripper.move(width=gripper_width, speed=0.2)
        else:
            gripper.grasp(width=gripper_width, speed=0.2, force=10, epsilon_outer=0.04)


class PandaRealRobot:
    def __init__(self, hostname="172.16.0.2", fps=30, init_pose=None):
        self.gripper = libfranka.Gripper(hostname)
        self.panda = panda_py.Panda(hostname)
        self.controller = controllers.JointPosition()
        self.panda.enable_logging(int(1e2))
        if init_pose is not None:
            self.init_pose = init_pose
        else:
            self.init_pose = [
                0,
                -0.84,
                0,
                -2.59,
                0,
                1.75,
                0.77,
            ]
        self.manager = Manager()
        self.dict_data = self.manager.dict({"gripper_width": 0.08})
        self.thread = threading.Thread(
            target=grasp, args=(self.gripper, self.dict_data)
        )
        self.thread.start()
        self.init_robot()
        self._rtb_robot = rtb.models.Panda()
        self.ee_controller = PinocchioMotionControl(
            urdf_path="/isaac-sim/src/assets/urdfs/panda/panda.urdf",
            wrist_name="panda_hand",
            arm_init_qpos=np.array(self.init_pose + [0.04, 0.04]),
        )
        self._rtb_robot.q = self.init_pose
        self.cameras = MultiRealSenseCamera(fps=60, image_width=640, image_height=480)
        self.panda.start_controller(self.controller)
        # other
        self.gripper_width = 0.08
        self.current_step = 0
        self.horizon = 50  # TODO
        self._buffer = {}
        self.ctx = self.panda.create_context(frequency=fps)
        self._last_qpos = None

        print("Finished initializing robot.")

    def init_robot(self):
        self.panda.move_to_joint_position(self.init_pose)
        self.dict_data["gripper_width"] = 0.08

    def log_pose(self):
        while True:
            try:
                tmp = np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)
                # log = self.panda.get_log().copy()
                # tmp = np.ascontiguousarray(log['O_T_EE'][-1]).reshape(4, 4).astype(np.float32)
                # tmp[3, :] = [0.0, 0.0, 0.0, 1.0]
                # # print(tmp)
                # time.sleep(0.001)
            except:
                print("FAILED")
                time.sleep(0.05)
                pass

    @property
    def tcp_pose(self):
        return np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)

    def get_gripper_width(self):
        return self.gripper.read_once().width

    def get_joint_position(self):
        return self.panda.get_log()["q"][-1]

    def get_ee_pose(self):
        self.panda.get_pose()

    def get_robot_state(self):
        """
        Get the real robot state.
        """

        # log = self.panda.get_log().copy()

        gripper_qpos = self.get_gripper_width()

        self._last_qpos = self.get_joint_position()
        robot_qpos = np.concatenate(
            [self._last_qpos, [gripper_qpos / 2.0], [gripper_qpos / 2.0]]
        )

        # obs = np.concatenate(
        #     [self.panda.get_position(), self.panda.get_orientation(), robot_qpos],
        #     dtype=np.float32,  # 15
        # )

        obs = robot_qpos
        assert obs.shape == (9,), f"incorrect obs shape, {obs.shape}"

        return obs

    def get_obs(self, visualize=False):
        """
        Get the real robot observation.
        """
        start_time = time.time()
        # pcd = self.get_pcd(visualize=visualize)
        start_time = time.time()
        state = self.get_robot_state()
        # obs = {"state": self.get_robot_state(), "pointcloud": self.get_pcd(visualize=visualize)}
        obs = {
            "state": state,
            # "pointcloud": pcd
        }

        return obs

    def _clip_action(self, action, delta):
        action[:7] = np.clip(
            action[:7], self._last_qpos - delta, self._last_qpos + delta
        )
        return action

    def apply_action(self, action, type="joint"):
        try:
            if type == "joint":
                assert action.shape == (9,), f"incorrect action shape, {action.shape}"
                gripper_width = sum(action[7:])
                if gripper_width > 0.04:
                    gripper_width = 0.08
                else:
                    gripper_width = 0.0
                action = self._clip_action(action, delta=0.03)
                time0 = time.time()
                if self.ctx.ok():
                    self.controller.set_control(action[:7])
                    self.dict_data["gripper_width"] = gripper_width
                time1 = time.time()
            elif type == "ee":
                gripper_width, transform = action
                pos, rot_mat = transform[:3, 3], transform[:3, :3]
                sol = self.ee_controller.control(pos, rot_mat)[:7]
                self._rtb_robot.q = sol
                if self.ctx.ok():
                    self.controller.set_control(sol)
                    if abs(gripper_width - self.gripper_width) > 0.01:
                        self.dict_data["gripper_width"] = gripper_width
        except Exception as e:
            print(e)

    def step(self, action, visualize=False):
        self.apply_action(action)
        return self.get_obs(visualize=visualize)

    def end(self):
        self.panda.get_robot().stop()
        self.dict_data["end"] = True
        self.thread.join()


# from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
# from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
# from franka_interface_msgs.msg import SensorDataGroup
# from frankapy import FrankaArm, SensorDataMessageType
# from frankapy import FrankaConstants as FC
# import rospy


class FrankapyRealRobot:
    def __init__(self, hostname="172.16.0.2", fps=30, init_pose=None):
        self.robot = FrankaArm()
        if init_pose is not None:
            self.init_pose = init_pose
        else:
            self.init_pose = [
                0,
                -0.84,
                0,
                -2.59,
                0,
                1.75,
                0.77,
            ]
        self.init_robot()
        self._rtb_robot = rtb.models.Panda()
        self._rtb_robot.q = self.init_pose
        print("Finished initializing robot.")
        # other
        self.gripper_width = 0.08
        self.current_step = 0
        self.pub = pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )
        self.rate = rospy.Rate(fps)
        self.gripper_closed = False
        self.cameras = MultiRealSenseCamera(fps=60, image_width=640, image_height=480)
        self.init_time = time.time()
        self.idx = 0

    def init_robot(self):
        self.robot.reset_joints()
        self.robot.open_gripper()
        self.robot.goto_joints(self.init_pose, duration=3)
        self.robot.goto_joints(
            self.init_pose, duration=600, dynamic=True, buffer_time=10
        )

    def get_robot_state(self):
        """
        Get the real robot state.
        """

        # log = self.panda.get_log().copy()

        gripper_width = self.robot.get_gripper_width()

        robot_qpos = np.concatenate(
            [self.robot.get_joints(), [gripper_width / 2.0], [gripper_width / 2.0]]
        )

        # obs = np.concatenate(
        #     [self.panda.get_position(), self.panda.get_orientation(), robot_qpos],
        #     dtype=np.float32,  # 15
        # )

        obs = robot_qpos
        assert obs.shape == (9,), f"incorrect obs shape, {obs.shape}"

        return obs

    def get_obs(self, visualize=False):
        """
        Get the real robot observation.
        """
        start_time = time.time()
        # pcd = self.get_pcd(visualize=visualize)
        start_time = time.time()
        state = self.get_robot_state()
        # obs = {"state": self.get_robot_state(), "pointcloud": self.get_pcd(visualize=visualize)}
        obs = {
            "state": state,
            # "pointcloud": pcd
        }

        return obs

    def _clip_action(self, action, delta):
        action[:7] = np.clip(
            action[:7], self._last_qpos - delta, self._last_qpos + delta
        )
        return action

    def apply_action(self, action, type="joint"):
        try:
            if type == "joint":
                assert action.shape == (9,), f"incorrect action shape, {action.shape}"
                # action = self._clip_action(action, delta=0.03)
                qpos = action[:7]
                gripper_width = sum(action[7:])
                time0 = time.time()
                traj_gen_proto_msg = JointPositionSensorMessage(
                    id=self.idx, timestamp=time.time() - self.init_time, joints=qpos
                )
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION
                    )
                )
                self.pub.publish(ros_msg)
                if gripper_width > 0.04 and self.gripper_closed:
                    self.robot.open_gripper()
                    self.gripper_closed = False
                elif gripper_width < 0.04 and not self.gripper_closed:
                    self.robot.close_gripper()
                    self.gripper_closed = True
            elif type == "ee":
                gripper_width, transform = action
                tep = SE3(transform)
                sol = self._rtb_robot.ik_LM(
                    tep, end="panda_hand", q0=self._rtb_robot.q
                )[0]
                self._rtb_robot.q = sol
                traj_gen_proto_msg = JointPositionSensorMessage(
                    id=self.idx, timestamp=time.time() - self.init_time, joints=sol
                )
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION
                    )
                )
                self.pub.publish(ros_msg)
                if gripper_width > 0.04 and self.gripper_closed:
                    self.robot.open_gripper()
                    self.gripper_closed = False
                elif gripper_width < 0.04 and not self.gripper_closed:
                    self.robot.close_gripper()
                    self.gripper_closed = True
        except Exception as e:
            traceback.print_exc()
            raise e
        self.idx += 1

    def step(self, action, visualize=False):
        self.apply_action(action)
        return self.get_obs(visualize=visualize)

    def end(self):
        # self.robot.get_robot().stop()
        term_proto_msg = ShouldTerminateSensorMessage(
            timestamp=time.time() - self.init_time, should_terminate=True
        )
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE
            )
        )
        self.pub.publish(ros_msg)

    def close_gripper(self):
        self.robot.close_gripper()
        self.gripper_closed = True

    def open_gripper(self):
        self.robot.open_gripper()
        self.gripper_closed = False


class RealRobot:
    def __init__(self, hostname="172.16.0.2", fps=30, init_pose=None):
        self.panda = panda_py.Panda(hostname)
        self.frankapy_robot = FrankaArm()
        self.controller = controllers.JointPosition()
        self.panda.enable_logging(int(1e2))
        if init_pose is not None:
            self.init_pose = init_pose
        else:
            self.init_pose = [
                0,
                -0.84,
                0,
                -2.59,
                0,
                1.75,
                0.77,
            ]
        self.init_robot()
        self._rtb_robot = rtb.models.Panda()
        self._rtb_robot.q = self.init_pose
        self.cameras = MultiRealSenseCamera(fps=60, image_width=640, image_height=480)
        self.panda.start_controller(self.controller)
        # other
        self.gripper_closed = False
        self.current_step = 0
        self.horizon = 50  # TODO
        self._buffer = {}
        self.ctx = self.panda.create_context(frequency=fps)
        self._last_qpos = None

        print("Finished initializing robot.")

    def init_robot(self):
        self.frankapy_robot.open_gripper()
        self.panda.move_to_joint_position(self.init_pose)

    def log_pose(self):
        pass

    @property
    def tcp_pose(self):
        return np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)

    def get_robot_state(self):
        """
        Get the real robot state.
        """

        # log = self.panda.get_log().copy()

        # gripper_state = self.gripper.read_once()
        # gripper_qpos = gripper_state.width
        gripper_qpos = self.frankapy_robot.get_gripper_width()
        print(f"Gripper:", gripper_qpos)
        # gripper_qpos = 0.08

        self._last_qpos = self.panda.get_log()["q"][-1]
        robot_qpos = np.concatenate(
            [self._last_qpos, [gripper_qpos / 2.0], [gripper_qpos / 2.0]]
        )

        # obs = np.concatenate(
        #     [self.panda.get_position(), self.panda.get_orientation(), robot_qpos],
        #     dtype=np.float32,  # 15
        # )

        obs = robot_qpos
        assert obs.shape == (9,), f"incorrect obs shape, {obs.shape}"

        return obs

    def get_obs(self, visualize=False):
        """
        Get the real robot observation.
        """
        start_time = time.time()
        # pcd = self.get_pcd(visualize=visualize)
        start_time = time.time()
        state = self.get_robot_state()
        # obs = {"state": self.get_robot_state(), "pointcloud": self.get_pcd(visualize=visualize)}
        obs = {
            "state": state,
            # "pointcloud": pcd
        }

        return obs

    def _clip_action(self, action, delta):
        action[:7] = np.clip(
            action[:7], self._last_qpos - delta, self._last_qpos + delta
        )
        return action

    def apply_action(self, action, type="joint"):
        try:
            if type == "joint":
                assert action.shape == (9,), f"incorrect action shape, {action.shape}"
                gripper_width = sum(action[7:])
                action = self._clip_action(action, delta=0.03)
                time0 = time.time()
                if self.ctx.ok():
                    self.controller.set_control(action[:7])
                    if gripper_width > 0.04 and self.gripper_closed:
                        self.frankapy_robot.open_gripper()
                        self.gripper_closed = False
                    elif gripper_width < 0.04 and not self.gripper_closed:
                        self.frankapy_robot.close_gripper()
                        self.gripper_closed = True
                time1 = time.time()
            elif type == "ee":
                gripper_width, transform = action
                tep = SE3(transform)
                sol = self._rtb_robot.ik_LM(
                    tep, end="panda_hand", q0=self._rtb_robot.q
                )[0]
                self._rtb_robot.q = sol
                if self.ctx.ok():
                    self.controller.set_control(sol)
                    if gripper_width > 0.04 and self.gripper_closed:
                        self.frankapy_robot.open_gripper()
                        self.gripper_closed = False
                    elif gripper_width < 0.04 and not self.gripper_closed:
                        self.frankapy_robot.close_gripper()
                        self.gripper_closed = True
        except Exception as e:
            print(e)

    def step(self, action, visualize=False):
        self.apply_action(action)
        return self.get_obs(visualize=visualize)

    def end(self):
        self.panda.get_robot().stop()
