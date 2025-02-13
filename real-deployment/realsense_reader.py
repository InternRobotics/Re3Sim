import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from pathlib import Path
import random
import re


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


# this code is tested for multi-camera setup
# pipeline = rs.pipeline()
# rsconfig = rs.config()
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = rsconfig.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()


def get_serial_numbers():
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print(
                "Found device: ",
                d.get_info(rs.camera_info.name),
                " ",
                d.get_info(rs.camera_info.serial_number),
            )
    else:
        print("No Intel Device connected")


class MultiRealSenseCamera:
    def __init__(self, image_width=640, image_height=480, fps=30):
        super().__init__()
        # set initial pipelines and configs
        self.serial_numbers, self.device_idxs = self.get_serial_numbers()
        self.total_cam_num = len(self.serial_numbers)
        self.pipelines = [None] * self.total_cam_num
        self.configs = [None] * self.total_cam_num

        # set resolutions and fps
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps

        # set pipelines and configs
        for i, serial_number in zip(range(0, self.total_cam_num), self.serial_numbers):
            self.pipelines[i] = rs.pipeline()
            self.configs[i] = rs.config()
            self.configs[i].enable_device(serial_number)
            self.configs[i].enable_stream(
                rs.stream.depth,
                self.image_width,
                self.image_height,
                rs.format.z16,
                self.fps,
            )
            self.configs[i].enable_stream(
                rs.stream.color,
                self.image_width,
                self.image_height,
                rs.format.rgb8,
                self.fps,
            )

        # Start streaming
        self.sensors = [None] * self.total_cam_num
        self.cfgs = [None] * self.total_cam_num
        self.depth_scales = [None] * self.total_cam_num
        # set master & slave
        master_or_slave = 1
        for i in range(0, self.total_cam_num):
            depth_sensor = self.ctx.devices[self.device_idxs[i]].first_depth_sensor()
            color_sensor = self.ctx.devices[self.device_idxs[i]].first_color_sensor()
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
            if i == 0:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode, master_or_slave)
                master_or_slave = 2
            else:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode, master_or_slave)

            self.cfgs[i] = self.pipelines[i].start(self.configs[i])
            depth_scale = (
                self.cfgs[i].get_device().first_depth_sensor().get_depth_scale()
            )
            self.depth_scales[i] = depth_scale
            # sensor = self.pipelines[i].get_active_profile().get_device().query_sensors()[1]
            # sensor.set_option(rs.option.exposure, 330)

    def undistorted_rgbd(self):
        depth_frame = [None] * self.total_cam_num
        color_frame = [None] * self.total_cam_num
        depth_image = [None] * self.total_cam_num
        color_image = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            frame = self.pipelines[i].wait_for_frames()
            align_frame = rs.align(rs.stream.color).process(frame)
            depth_frame[i] = align_frame.get_depth_frame()
            color_frame[i] = align_frame.get_color_frame()
            depth_image[i] = (
                np.asanyarray(depth_frame[i].get_data()) * self.depth_scales[i]
            )
            color_image[i] = np.asanyarray(color_frame[i].get_data())
        return color_image, depth_image

    def undistorted_rgb(self):
        color_frame = [None] * self.total_cam_num
        color_image = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            frame = self.pipelines[i].wait_for_frames()
            align_frame = rs.align(rs.stream.color).process(frame)
            color_frame[i] = align_frame.get_color_frame()
            color_image[i] = np.asanyarray(color_frame[i].get_data())
        return color_image

    def get_serial_numbers(self):
        serial_numbers = []
        device_idxs = []
        self.ctx = rs.context()
        if len(self.ctx.devices) > 0:
            for j, d in enumerate(self.ctx.devices):
                name = d.get_info(rs.camera_info.name)
                serial_number = d.get_info(rs.camera_info.serial_number)
                print(f"Found device: {name} {serial_number}")
                serial_numbers.append(serial_number)
                device_idxs.append(j)
        else:
            print("No Intel Device connected")
        return serial_numbers, device_idxs

    def get_intrinsic_color(self):
        intrinsic = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            profile = self.cfgs[i].get_stream(rs.stream.color).as_video_stream_profile()
            intr = profile.get_intrinsics()
            intrinsic[i] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
            }
        return intrinsic

    def get_intrinsic_depth(self):
        intrinsic = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            profile = self.cfgs[i].get_stream(rs.stream.depth).as_video_stream_profile()
            intr = profile.get_intrinsics()
            intrinsic[i] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
            }
        return intrinsic


def main1():
    # to get serial numbers
    # get_serial_numbers()
    multi_camera = MultiRealSenseCamera(fps=60)
    step_id = 0
    data_path_root = "images"

    log_folder = get_log_folder(data_path_root)
    os.makedirs(log_folder / "image", exist_ok=True)
    os.makedirs(log_folder / "depth", exist_ok=True)
    print("Data saved in ", log_folder)

    camera_idx = -1
    while True:
        step_id += 1
        t1 = time.time()
        color_image, depth_image = multi_camera.undistorted_rgbd()
        # color_image = multi_camera.undistorted_rgb()
        t2 = time.time()
        # if t2 - t1 > 0.02:
        # print("t2 - t1", t2 - t1)
        data_list = []
        if camera_idx == -1:
            for i in range(0, multi_camera.total_cam_num):
                # data_list.append([color_image[0], color_image[1]])
                cv2.imshow(f"Color Image {i}", color_image[i])
                cv2.imshow(
                    f"Depth Image {i}", depth_image[i] / depth_image[i].max() * 255
                )
                # cv2.imwrite("color.png", color_image[1])
                # print("depth_image.shape :", depth_image[i].shape)
                # print("depth_image.max :", depth_image[i].max())
                # print("depth_image.min :", depth_image[i].min())
                # break
        else:
            cv2.imshow(f"Color Image {camera_idx}", color_image[camera_idx])
            cv2.imshow(
                f"Depth Image {camera_idx}",
                depth_image[camera_idx] / depth_image[camera_idx].max() * 255,
            )

        if step_id < 500:
            continue
        if camera_idx == -1:
            camera_idx = int(input("Enter camera index: "))
            print(f"Saving images from camera {camera_idx}")

            # close useless windows
            for i in range(0, multi_camera.total_cam_num):
                if i == camera_idx:
                    continue
                cv2.destroyWindow(f"Color Image {i}")
                cv2.destroyWindow(f"Depth Image {i}")

        op = cv2.waitKey(1) & 0xFF
        if op == ord("q"):
            break
        elif op == ord("s"):
            print(f"Saved image (id {step_id}) at {log_folder}")
            cv2.imwrite(log_folder / f"image/{step_id}.png", color_image[camera_idx])
            cv2.imwrite(log_folder / f"depth/{step_id}.png", depth_image[camera_idx])
            np.save(log_folder / f"depth/{step_id}.npy", depth_image[camera_idx])


def main2():
    multi_camera = MultiRealSenseCamera(fps=60)
    color_image, depth_image = multi_camera.undistorted_rgbd()
    print("color_image.shape :", color_image[0].shape)
    print("depth_image.shape :", depth_image[0].shape)
    print("depth_image.max :", depth_image[0].max())
    print("depth_image.min :", depth_image[0].min())
    cv2.imshow("Color Image 0", color_image[0])
    cv2.imshow("Color Image 1", color_image[1])
    print("intrinsic :", multi_camera.get_intrinsic_color())
    cv2.waitKey(0)
    # print intrinsic


if __name__ == "__main__":
    main1()
