import dearpygui.dearpygui as dpg
import numpy as np
from multiprocessing import Process, Lock, Array, Manager
import ctypes
from typing import Tuple, Dict, Any
import time
from real2sim2real.envs.env import BaseEnv
from scipy.spatial.transform import Rotation as R
import imageio
import os


class SharedState:
    def __init__(self):
        # Camera pose as 6 floats (3 position + 3 rotation)
        self.pose_lock = Lock()
        self.dict = Manager().dict()
        self.dict["pose"] = np.zeros(6, dtype=np.float64)
        self.dict["image"] = np.ones((480, 640, 3), dtype=np.float32)
        # Fixed image size 640x480x3
        self.image_lock = Lock()

        # Image update flag
        self.image_updated = Array("B", [0])


class CameraController:
    def __init__(self):
        self.last_mouse_pos = (0, 0)
        self.is_rotating = False
        self.is_panning = False

        # Camera parameters
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.position = np.array([0.0, 0.0, -5.0])

        # Control parameters
        self.rotate_speed = 0.01
        self.pan_speed = 0.01
        self.zoom_speed = 0.1

    def update_from_shared(self, shared_pose):
        with shared_pose.pose_lock:
            pose_data = shared_pose.dict["pose"][:]
            self.position = np.array(pose_data[:3])
            self.rotation = np.array(pose_data[3:])

    def write_to_shared(self, shared_pose):
        with shared_pose.pose_lock:
            shared_pose.dict["pose"][:3] = self.position
            shared_pose.dict["pose"][3:] = self.rotation


def gui_process(shared_state: SharedState):
    dpg.create_context()

    # Initialize camera controller
    camera_controller = CameraController()

    # Add save image callback
    def _save_image(sender, app_data):
        try:
            with shared_state.image_lock:
                img_data = np.array(shared_state.dict["image"])
                img_data = (img_data * 255).astype(np.uint8)
                save_path = dpg.get_value("save_path_input")
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f"image_{int(time.time())}.png")
                imageio.imwrite(save_path, img_data)
                print(f"Image saved to: {save_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def _mouse_handler(sender, app_data):
        if not dpg.is_item_hovered("image_display"):
            return

        mouse_pos = dpg.get_mouse_pos()

        if dpg.is_mouse_button_down(0):  # Left button rotate
            if not camera_controller.is_rotating:
                camera_controller.last_mouse_pos = mouse_pos
                camera_controller.is_rotating = True
            else:
                delta = (
                    mouse_pos[0] - camera_controller.last_mouse_pos[0],
                    mouse_pos[1] - camera_controller.last_mouse_pos[1],
                )
                camera_controller.rotation[1] += (
                    delta[0] * camera_controller.rotate_speed
                )
                camera_controller.rotation[0] += (
                    delta[1] * camera_controller.rotate_speed
                )
                camera_controller.last_mouse_pos = mouse_pos

        elif dpg.is_mouse_button_down(1):  # Right button pan
            if not camera_controller.is_panning:
                camera_controller.last_mouse_pos = mouse_pos
                camera_controller.is_panning = True
            else:
                delta = (
                    mouse_pos[0] - camera_controller.last_mouse_pos[0],
                    mouse_pos[1] - camera_controller.last_mouse_pos[1],
                )
                camera_controller.position[0] += delta[0] * camera_controller.pan_speed
                camera_controller.position[1] -= delta[1] * camera_controller.pan_speed
                camera_controller.last_mouse_pos = mouse_pos

        if not dpg.is_mouse_button_down(0):
            camera_controller.is_rotating = False
        if not dpg.is_mouse_button_down(1):
            camera_controller.is_panning = False

        # Update shared pose
        camera_controller.write_to_shared(shared_state)

    def _mouse_wheel_handler(sender, app_data):
        if dpg.is_item_hovered("image_display"):
            camera_controller.position[2] += app_data * camera_controller.zoom_speed
            camera_controller.write_to_shared(shared_state)

    # Create window and image display
    with dpg.window(label="Camera View", tag="main_window"):
        dpg.add_text("Camera View")

        # Add save path input and save button
        default_path = os.path.join(os.getcwd(), "gui_images")
        dpg.add_input_text(
            label="path", default_value=default_path, tag="save_path_input", width=400
        )
        dpg.add_button(label="save image", callback=_save_image)

        # Create texture
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=640,
                height=480,
                default_value=np.ones((480, 640, 3), dtype=np.float32),
                format=dpg.mvFormat_Float_rgb,
                tag="image_texture",
            )

        dpg.add_image("image_texture", tag="image_display")

    # Register callbacks
    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(callback=_mouse_handler)
        dpg.add_mouse_wheel_handler(callback=_mouse_wheel_handler)

    # Create viewport
    dpg.create_viewport(title="Camera Viewer", width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Main loop
    try:
        while dpg.is_dearpygui_running():
            # Check image update
            with shared_state.image_lock:
                if shared_state.image_updated[0]:
                    try:
                        img_data = np.array(shared_state.dict["image"])
                        img_data = img_data.reshape((480, 640, 3))
                        dpg.set_value("image_texture", img_data)
                        shared_state.image_updated[0] = 0

                        # tmp save images for render vis
                        tmp_dir = "tmp_for_render"
                        save_path = os.path.join(
                            tmp_dir, f"{int(time.time() * 1000)}_timestep.png"
                        )
                        imageio.imwrite(save_path, (img_data * 255).astype(np.uint8))
                    except Exception as img_error:
                        import traceback

                        print(f"Image update error: {img_error}")
                        traceback.print_exc()

            # Render frame
            try:
                dpg.render_dearpygui_frame()
            except Exception as render_error:
                import traceback

                print(f"Render error: {render_error}")
                traceback.print_exc()
                break

    except Exception as e:
        import traceback

        print(f"GUI process error: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up GUI resources...")
        dpg.destroy_context()
        print("GUI process terminated")


class GuiForEnv:
    def __init__(
        self,
        env: BaseEnv,
        shared_state: SharedState,
        camera_params: Dict[Any, Any] = None,
    ):
        self.env = env
        from real2sim2real.tasks.pick_and_place import PickAndPlace

        self.task: PickAndPlace = list(env.runner.current_tasks.values())[0]
        self.shared_state = shared_state
        self.camera_params = camera_params
        from real2sim2real.utils.items import create_camera
        from real2sim2real.utils.utils import compute_fx_fy
        from omni.isaac.core.utils.transformations import (
            pose_from_tf_matrix,
            get_relative_transform,
            tf_matrix_from_pose,
        )
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.prims import XFormPrim

        if self.camera_params is None:
            self.camera_params = {
                "name": "_camera_for_gui",
                "position": [0.96971275, 0.12246315, 0.79907076],
                "orientation": [-0.15189554, 0.68471294, 0.69626954, -0.15265866],
                "camera_params": [
                    606.18847656,
                    605.83990479,
                    313.10461426,
                    247.62963867,
                    640,
                    480,
                ],
                "prim_path": "/_camera_for_gui",
            }

        cam_pos, cam_orientation = (
            self.camera_params["position"],
            self.camera_params["orientation"],
        )
        cam_tf = tf_matrix_from_pose(cam_pos, cam_orientation)
        cam_to_world_tf = (
            get_relative_transform(
                get_prim_at_path(self.task.robot.prim_path),
                get_prim_at_path(self.task._root_prim_path),
            )
            @ cam_tf
        )
        cam_to_world_pos, cam_to_world_ori = pose_from_tf_matrix(cam_to_world_tf)

        self.camera = create_camera(
            self.camera_params["name"],
            self.camera_params["prim_path"],
            cam_to_world_pos,
            cam_to_world_ori,
            self.camera_params["camera_params"],
        )
        self.camera_xform = XFormPrim(self.camera_params["prim_path"])
        self.camera.add_motion_vectors_to_frame()
        self.camera.add_semantic_segmentation_to_frame()
        self.camera.add_distance_to_image_plane_to_frame()
        self.semantic_label_to_index = {}
        init_camera_pose = get_relative_transform(
            get_prim_at_path(self.camera_params["prim_path"]),
            get_prim_at_path(self.task._root_prim_path),
        )
        init_camera_pos = init_camera_pose[:3, 3]
        init_camera_rot = init_camera_pose[:3, :3]
        init_camera_euler = R.from_matrix(init_camera_rot).as_euler("xyz")
        self.init_camera_pose = np.concatenate([init_camera_pos, init_camera_euler])
        with self.shared_state.pose_lock:
            self.shared_state.dict["pose"] = self.init_camera_pose

    def initialize_semantic_index(self, segment_data) -> Tuple[int, int]:
        info = segment_data["info"]
        name_to_index = {}
        for key, value in info["idToLabels"].items():
            name_to_index[value["class"]] = int(key)
        return name_to_index

    def update_camera_pose(self, shared_state: SharedState):
        with shared_state.pose_lock:
            cam_pose = shared_state.dict["pose"]
        cam_pos = cam_pose[:3]
        cam_euler = cam_pose[3:]
        cam_world_pose = np.eye(4)
        cam_world_pose[:3, 3] = cam_pos
        cam_world_pose[:3, :3] = R.from_euler("xyz", cam_euler).as_matrix()
        self.camera_xform.set_local_pose(cam_world_pose)

    def render(self):
        sim_image = self.camera.get_rgba()
        if len(sim_image.shape) == 1:
            width, height = self.camera.get_resolution()
            return np.zeros((height, width, 3))

        sim_image = sim_image[:, :, :3]
        semantic_seg = self.camera._custom_annotators["semantic_segmentation"]
        segment_data = semantic_seg.get_data()
        if not self.semantic_label_to_index:
            self.semantic_label_to_index = self.initialize_semantic_index(segment_data)
        try:
            table_mask = np.where(
                segment_data["data"] == self.semantic_label_to_index["table"], 1, 0
            )
        except:
            table_mask = np.zeros_like(segment_data["data"])
        ground_mask = np.where(
            segment_data["data"] == self.semantic_label_to_index.get("ground", 0), 1, 0
        )
        mix_mask = np.logical_or(table_mask, ground_mask)
        resolution = self.camera.get_resolution()
        from real2sim2real.utils.utils import compute_fx_fy
        from omni.isaac.core.utils.transformations import get_relative_transform
        from omni.isaac.core.utils.prims import get_prim_at_path

        fx, fy = compute_fx_fy(self.camera, resolution[1], resolution[0])
        cam_pose = get_relative_transform(
            get_prim_at_path(self.camera.prim_path),
            get_prim_at_path(self.task._root_prim_path),
        )
        gaussian_rendered_background = self.task.background.render(
            cam_pose=cam_pose,
            width=resolution[0],
            height=resolution[1],
            fx=fx,
            fy=fy,
            camera_pose_frame="isaacsim",
        )
        mixed_image = np.where(
            mix_mask[..., None], gaussian_rendered_background, sim_image
        )
        with self.shared_state.image_lock:
            self.shared_state.dict["image"] = mixed_image.astype(np.float32) / 255.0
            self.shared_state.image_updated[0] = 1


if __name__ == "__main__":
    from real2sim2real.envs.config import SimulatorConfig
    from real2sim2real.envs.env import BaseEnv

    # Create shared state
    shared_state = SharedState()

    # Start GUI process
    gui_proc = Process(target=gui_process, args=(shared_state,))
    gui_proc.start()

    file_path = "configs/clean/1203_pick_multi_continuous_new_align.yaml"
    sim_config = SimulatorConfig(file_path)

    headless = False
    webrtc = False

    env = BaseEnv(sim_config, headless=False, webrtc=webrtc)

    import numpy as np
    from omni.isaac.core.utils.rotations import (
        euler_angles_to_quat,
        quat_to_euler_angles,
    )

    i = 0
    env.reset()
    while env.simulation_app.is_running():
        i += 1
        if i == 10:
            gui = GuiForEnv(env, shared_state)
        elif i >= 10:
            gui.render()

        env._runner._world.step(render=True)

    env.simulation_app.close()
    # Start your render process here
    # render_proc = Process(target=your_render_process, args=(shared_state,))
    # render_proc.start()
