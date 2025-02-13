import numpy as np
from pynput.keyboard import Key, Listener
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
import cv2

import time
import rospy


def main():
    fa = FrankaArm()
    fa.reset_joints()
    fa.goto_pose(
        FC.HOME_POSE,
        duration=10000,
        dynamic=True,
        buffer_time=10000,
        cartesian_impedances=[1200.0, 1200.0, 1200.0, 50.0, 50.0, 50.0],
    )
    T_ee_world = fa.get_pose()
    print(
        "Translation: {} | Rotation: {}".format(
            T_ee_world.translation, T_ee_world.quaternion
        )
    )

    fa.open_gripper()
    dt = 0.01
    init_time = rospy.Time.now().to_time()
    rospy.loginfo("Initializing Sensor Publisher. Starting")
    pub = rospy.Publisher(
        FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10
    )
    rate = rospy.Rate(1 / dt)
    i = 0
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Control Window", img)
    while True:
        start_time = rospy.Time.now().to_time() - init_time
        # T_ee_world = fa.get_pose()
        op = cv2.waitKey(1) & 0xFF
        if op == ord("w"):
            print("w")
            T_ee_world.translation += [0.002, 0.0, 0.0]
        elif op == ord("s"):
            T_ee_world.translation -= [0.002, 0.0, 0.0]
        elif op == ord("a"):
            T_ee_world.translation += [0, 0.002, 0]
        elif op == ord("d"):
            T_ee_world.translation -= [0, 0.002, 0]
        elif op == ord("e"):
            T_ee_world.translation += [0, 0, 0.002]
        elif op == ord("q"):
            T_ee_world.translation -= [0, 0, 0.002]
        elif op == ord("c"):
            fa.close_gripper()
        elif op == ord("o"):
            fa.open_gripper()
        elif op == ord("/"):
            break
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i,
            timestamp=start_time,
            position=T_ee_world.translation,
            quaternion=T_ee_world.quaternion,
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION
            ),
        )
        rospy.loginfo(
            "Publishing: ID {}, position: {}".format(
                traj_gen_proto_msg.id, T_ee_world.translation
            )
        )
        pub.publish(ros_msg)
        i += 1
        time.sleep(max(0, dt - (rospy.Time.now().to_time() - init_time - start_time)))

    fa.reset_joints()


if __name__ == "__main__":
    main()
