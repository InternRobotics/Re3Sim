from frankapy import FrankaArm
from src.realsense_reader import MultiRealSenseCamera
import cv2
import numpy as np
import time


def collect_image_qpos_pair(
    franka: FrankaArm, camera: MultiRealSenseCamera, data_dir: str, index: int
):
    franka.reset_joints()
    franka.open_gripper()
    franka.run_guide_mode(100000, block=False)
    start_time = time.time()
    last_time = None
    i = 0
    while last_time is None or (last_time - start_time) < 100000:
        image = camera.undistorted_rgbd()[0][index]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        op = cv2.waitKey(1) & 0xFF
        cv2.imshow("image", image)
        if op == ord("s"):
            joints = franka.get_joints()
            np.save(data_dir + "/image_{}.npy".format(i), image)
            cv2.imwrite(data_dir + "/image_{}.png".format(i), image)
            np.save(data_dir + "/joints_{}.npy".format(i), joints)
            i += 1
            pass
        elif op == ord("q"):
            break


if __name__ == "__main__":
    fa = FrankaArm()
    camera = MultiRealSenseCamera()
    index = 1
    data_dir = "/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/frankapy/tests"
    collect_image_qpos_pair(fa, camera, data_dir, index)
