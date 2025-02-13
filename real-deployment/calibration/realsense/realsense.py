import pyrealsense2 as rs
import numpy as np
import open3d as o3d


class Camera:
    serial_counter = 0

    def __init__(self, serial_number, rgb_resolution, depth_resolution):
        self.serial_number = serial_number
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.serial_id = Camera.serial_counter
        Camera.serial_counter += 1
        self.pipeline = rs.pipeline()
        self.config = self.configure_pipeline()
        self.align = rs.align(rs.stream.color)

    def configure_pipeline(self):
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, *self.rgb_resolution, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, *self.depth_resolution, rs.format.z16, 30)
        return config

    def shoot(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print(f"No frames received from camera {self.serial_number}")
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def get_intrinsics_matrix(self):
        profile = self.pipeline.get_active_profile()
        rgb_intrinsics_raw = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        # To a matrix
        rgb_intrinsics = np.array(
            [
                [rgb_intrinsics_raw.fx, 0, rgb_intrinsics_raw.ppx],
                [0, rgb_intrinsics_raw.fy, rgb_intrinsics_raw.ppy],
                [0, 0, 1],
            ]
        )

        depth_intrinsics_raw = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        # To a matrix
        depth_intrinsics = np.array(
            [
                [depth_intrinsics_raw.fx, 0, depth_intrinsics_raw.ppx],
                [0, depth_intrinsics_raw.fy, depth_intrinsics_raw.ppy],
                [0, 0, 1],
            ]
        )

        return (
            rgb_intrinsics,
            rgb_intrinsics_raw.coeffs,
            depth_intrinsics,
            depth_intrinsics_raw.coeffs,
        )

    def get_intrinsics_raw(self):
        profile = self.pipeline.get_active_profile()
        rgb_intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        depth_intrinsics = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return (
            rgb_intrinsics,
            rgb_intrinsics.coeffs,
            depth_intrinsics,
            depth_intrinsics.coeffs,
        )

    def get_depth_scale(self):
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def get_pointcloud(self, depth_trunc):
        color_image, depth_image = self.shoot()
        # get intrinsic parameters
        rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = (
            self.get_intrinsics_matrix()
        )
        depth_scale = self.get_depth_scale()

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1 / depth_scale,
            depth_trunc=depth_trunc,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width=color_image.shape[1],
                height=color_image.shape[0],
                fx=rgb_intrinsics[0, 0],
                fy=rgb_intrinsics[1, 1],
                cx=rgb_intrinsics[0, 2],
                cy=rgb_intrinsics[1, 2],
            ),
        )

        return pcd

    def start(self):
        self.pipeline.start(self.config)

    def stop(self):
        self.pipeline.stop()


def get_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    device_serials = [
        device.get_info(rs.camera_info.serial_number) for device in devices
    ]
    device_serials.sort()
    return device_serials
