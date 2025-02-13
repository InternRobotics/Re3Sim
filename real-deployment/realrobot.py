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


class PandaRealRobot:
    def __init__(
        self,
        hostname="172.16.0.2",
        fps=30,
        init_pose=None,
        image_fps=60,
        image_width=640,
        image_height=480,
    ):
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
        self.init_robot()
        self._rtb_robot = rtb.models.Panda()
        self.ee_controller = PinocchioMotionControl(
            urdf_path="/isaac-sim/src/assets/urdfs/panda/panda.urdf",
            wrist_name="panda_hand",
            arm_init_qpos=np.array(self.init_pose + [0.04, 0.04]),
        )
        self._rtb_robot.q = self.init_pose
        self.cameras = MultiRealSenseCamera(
            fps=image_fps, image_width=image_width, image_height=image_height
        )
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
        self.gripper.homing()
        # self.gripper.move(width=0.08, speed=0.1)

    def log_pose(self):
        while True:
            try:
                tmp = np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)
                # log = self.panda.get_log().copy()
                # tmp = np.ascontiguousarray(log['O_T_EE'][-1]).reshape(4, 4).astype(np.float32)
                # tmp[3, :] = [0.0, 0.0, 0.0, 1.0]
                # # print(tmp)
                self.tcp_pose_queue.put(tmp)
                # time.sleep(0.001)
            except:
                print("FAILED")
                time.sleep(0.05)
                pass

    @property
    def tcp_pose(self):
        return np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)

    def get_robot_state(self):
        """
        Get the real robot state.
        """

        # log = self.panda.get_log().copy()

        gripper_state = self.gripper.read_once()
        gripper_qpos = gripper_state.width

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
            "tcp_pose": self.tcp_pose,
            "panda_hand_pose": self._rtb_robot.fkine(
                self._rtb_robot.q, end="panda_hand"
            ).A,
            # "pointcloud": pcd
        }

        return obs

    def _clip_action(self, action, delta):
        action[:7] = np.clip(
            action[:7], self._last_qpos - delta, self._last_qpos + delta
        )
        return action

    def apply_action(self, action, type="joint"):
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
                if abs(gripper_width - self.gripper_width) > 0.01:
                    status = self.gripper.grasp(
                        width=gripper_width, speed=0.1, force=40, epsilon_outer=0.08
                    )
                    print(
                        f"Gripper grasp status: {status}, gripper width: {gripper_width}"
                    )
                    self.gripper_width = gripper_width
            time1 = time.time()
        elif type == "ee":
            gripper_width, transform = action
            pos, rot_mat = transform[:3, 3], transform[:3, :3]
            sol = self.ee_controller.control(pos, rot_mat)[:7]
            self._rtb_robot.q = sol
            if self.ctx.ok():
                self.controller.set_control(sol)
                if abs(gripper_width - self.gripper_width) > 0.01:
                    status = self.gripper.grasp(
                        width=gripper_width, speed=0.1, force=40, epsilon_outer=0.08
                    )
                    self.gripper_width = gripper_width

    def step(self, action, visualize=False, type="joint"):
        self.apply_action(action, type=type)
        return self.get_obs(visualize=visualize)

    def end(self):
        self.panda.get_robot().stop()


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
        while True:
            try:
                tmp = np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)
                # log = self.panda.get_log().copy()
                # tmp = np.ascontiguousarray(log['O_T_EE'][-1]).reshape(4, 4).astype(np.float32)
                # tmp[3, :] = [0.0, 0.0, 0.0, 1.0]
                # # print(tmp)
                self.tcp_pose_queue.put(tmp)
                # time.sleep(0.001)
            except:
                print("FAILED")
                time.sleep(0.05)
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
