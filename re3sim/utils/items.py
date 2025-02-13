from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.prims import XFormPrim, RigidPrimView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.sensor import Camera, CameraView
import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from ..utils.frame_tools import FrameTools
from ..utils.prim import get_points_at_path
from ..utils.pcd_utils import bbox_from_pcd
import os


def create_dynamic_cube(name, cube_params, prim_path):
    static_friction = cube_params.get("static_friction", 4.0)
    dynamic_friction = cube_params.get("dynamic_friction", 4.0)
    restitution = cube_params.get("restitution", 0.0)
    mass = cube_params.get("mass", 0.01)
    physics_material_path = find_unique_string_name(
        initial_name="/World/Physics_Materials/physics_material",
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    physics_material = PhysicsMaterial(
        prim_path=physics_material_path,
        dynamic_friction=dynamic_friction,
        static_friction=static_friction,
        restitution=restitution,
    )
    cube = DynamicCuboid(
        name=name,
        prim_path=prim_path,
        scale=cube_params["scale"],
        size=1.0,
        color=np.array(cube_params["color"]),
        physics_material=physics_material,
    )
    cube.set_mass(mass)
    cube.set_local_pose(
        translation=cube_params["position"], orientation=cube_params["orientation"]
    )
    return cube


def create_object(name, object_params, prim_path, type, frame_tools):
    frame_tools: FrameTools
    name = find_unique_string_name(
        initial_name=name,
        is_unique_fn=lambda x: not is_prim_path_valid(prim_path + "/" + x),
    )
    xformprim = XFormPrim(prim_path)

    xformprim.set_local_pose(
        translation=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0])
    )
    if type == "DynamicCuboid":
        return create_dynamic_cube(name, object_params, prim_path + "/" + name)
    elif type == "usdz":
        if not os.path.exists(object_params["path"]):
            raise FileNotFoundError(f"File {object_params['path']} not found")
        create_prim(
            prim_path=prim_path + "/" + name,
            usd_path=object_params["path"],
            scale=object_params["scale"],
        )
        xformprim = XFormPrim(prim_path + "/" + name, name=name)
        xformprim.set_local_pose(
            translation=object_params["position"],
            orientation=object_params["orientation"],
        )
        return xformprim
    else:
        raise NotImplementedError(
            f"Object type {object_params['type']} not implemented"
        )


class Item:
    def __init__(
        self,
        name,
        type,
        params,
        prim_path,
        frame_tools,
        root_prim_path,
        randomize_params=None,
    ):
        self.RAMDOM_FUNC_PARAMS = {
            "random_position": self._random_position,
            "random_orientation": self._random_orientation,
        }
        self.name = name
        self.type = type
        self.prim_path = prim_path
        self.root_prim_path = root_prim_path
        self.randomize = params.get("randomize", False) and (
            len(randomize_params) > 0 if randomize_params is not None else False
        )
        self.randomize_params = randomize_params
        self.frame_tools = frame_tools
        # change prim path
        frame_name = params["frame_name"]
        frame_transform = frame_tools.get_frame_transform(frame_name)
        item_transform_in_frame = np.eye(4)
        item_transform_in_frame[:3, :3] = R.from_quat(
            params["orientation"], scalar_first=True
        ).as_matrix()
        item_transform_in_frame[:3, 3] = np.array(params["position"])
        item_transform_in_world = frame_transform @ item_transform_in_frame
        params["position"] = item_transform_in_world[:3, 3]
        params["orientation"] = R.from_matrix(
            item_transform_in_world[:3, :3]
            / np.linalg.det(item_transform_in_world[:3, :3]) ** (1 / 3)
        ).as_quat(scalar_first=True)
        self.params = params
        self.object = create_object(name, params, prim_path, type, frame_tools)
        self._xform = XFormPrim(prim_path)

    def _random_position(
        self, position_min, position_max, align_surface=False, **align_surface_params
    ):
        position = np.random.uniform(position_min, position_max)
        if align_surface:
            self.align_surface(
                self.params["position"][0] + position_min[0],
                self.params["position"][1] + position_min[1],
                **align_surface_params,
            )
        self._xform.set_local_pose(translation=position)
        return position.tolist()

    def _random_orientation(self, rotation_min, rotation_max):
        xyz_degree = np.random.uniform(rotation_min, rotation_max)
        xyz_radian = np.deg2rad(xyz_degree)
        random_rotation = R.from_euler("xyz", xyz_radian)
        # original_orientation = R.from_quat(self.params["orientation"], scalar_first=True)
        # new_orientation = random_rotation * original_orientation
        # new_quat = new_orientation.as_quat(scalar_first=True)
        self._xform.set_local_pose(
            orientation=random_rotation.as_quat(scalar_first=True)
        )
        return random_rotation.as_quat(scalar_first=True).tolist()

    def reset(self, align_surface=False, **align_surface_params):
        RigidPrimView(self.object.prim_path).set_linear_velocities(
            np.array([[0, 0, 0]])
        )
        RigidPrimView(self.object.prim_path).set_angular_velocities(
            np.array([[0, 0, 0]])
        )
        self.object.set_local_pose(
            translation=self.params["position"], orientation=self.params["orientation"]
        )
        self.object.set_local_pose(
            translation=self.params["position"], orientation=self.params["orientation"]
        )
        random_results = {}
        if self.randomize:
            if "random_position" in self.randomize_params:
                random_results["random_position"] = self.RAMDOM_FUNC_PARAMS[
                    "random_position"
                ](
                    **self.randomize_params["random_position"],
                    align_surface=align_surface,
                    **align_surface_params,
                )
            if "random_orientation" in self.randomize_params:
                random_results["random_orientation"] = self.RAMDOM_FUNC_PARAMS[
                    "random_orientation"
                ](**self.randomize_params["random_orientation"])
        return random_results

    def get_world_pose(self):
        return self.object.get_local_pose()

    def get_item_pcd(self) -> np.ndarray:
        if hasattr(self, "item_pcd"):
            return self.item_pcd
        else:
            self.item_pcd = get_points_at_path(
                self.object.prim_path, relative_frame_prim_path=self.object.prim_path
            )
            return self.item_pcd

    def align_surface(self, x, y, table_pcd, percent=95, z_offset=0.01):
        item_pcd = self.get_item_pcd()
        item_pcd_bbox = bbox_from_pcd(item_pcd)
        xy_min = item_pcd_bbox[0][:2] + np.array([x, y])
        xy_max = item_pcd_bbox[1][:2] + np.array([x, y])
        table_pcd_in_xy_bool = (
            (table_pcd[:, 0] > xy_min[0])
            & (table_pcd[:, 0] < xy_max[0])
            & (table_pcd[:, 1] > xy_min[1])
            & (table_pcd[:, 1] < xy_max[1])
        )
        table_pcd_in_xy = table_pcd[table_pcd_in_xy_bool]
        table_pcd_in_xy_top_percent = np.percentile(table_pcd_in_xy[:, 2], percent)
        self.params["position"][2] = (
            table_pcd_in_xy_top_percent + z_offset - item_pcd_bbox[0][2]
        )
        return self.params["position"][2]

    @property
    def pcd(self):
        return self.get_item_pcd()

    @property
    def xform(self):
        return self._xform


def create_camera(
    name,
    prim_path,
    position,
    orientation,
    camera_params,
    sub_xform_orientation=None,
    pixel_size=3,
    f_stop=2.0,
    focus_distance=0.3,
    D=None,
) -> Camera:
    import math

    print("Create camera at ", prim_path)
    xformprim = XFormPrim(prim_path)
    subxformprim = XFormPrim(prim_path + "/camera")
    camera = Camera(name=name, prim_path=prim_path + "/camera" + "/camera")
    xformprim.set_local_pose(translation=position, orientation=orientation)
    if sub_xform_orientation is not None:
        subxformprim.set_local_pose(
            translation=np.array([0, 0, 0]), orientation=sub_xform_orientation
        )
    else:
        subxformprim.set_local_pose(
            translation=np.array([0, 0, 0]), orientation=np.array([0.5, 0.5, -0.5, 0.5])
        )
    if D is None:
        D = np.zeros(8)
    # rotation = R.from_quat(orientation, scalar_first=True).as_matrix()
    # x 90 degree; z -90 degree; self rotation
    # rotation = rotation @ (rtb.ET.Rx(np.pi/2).A() @ rtb.ET.Rz(-np.pi/2).A())[:3, :3]
    # rotation = (rtb.ET.Rx(-np.pi/2).A() @ rtb.ET.Rz(-np.pi/2).A())[:3, :3] @ rotation
    # orientation = R.from_matrix(rotation).as_quat(scalar_first=True)
    # camera.set_local_pose(translation=position, orientation=orientation)
    camera.initialize()
    width, height = camera_params[4], camera_params[5]
    camera.set_resolution([width, height])
    camera.set_clipping_range(0.02, 5)
    fx, fy = camera_params[0], camera_params[1]
    cx, cy = camera_params[2], camera_params[3]
    horizontal_aperture = pixel_size * 1e-3 * width
    vertical_aperture = pixel_size * 1e-3 * height
    focal_length_x = fx * pixel_size * 1e-3
    focal_length_y = fy * pixel_size * 1e-3
    focal_length = (focal_length_x + focal_length_y) / 2  # in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    camera.set_focal_length(focal_length / 10.0)
    camera.set_focus_distance(focus_distance)
    camera.set_lens_aperture(f_stop * 100.0)
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)
    camera.set_vertical_aperture(vertical_aperture / 10.0)
    camera.set_clipping_range(0.05, 1.0e5)

    # Set the distortion coefficients, this is nessesary, when cx, cy are not in the center of the image
    diagonal = 2 * math.sqrt(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2)
    diagonal_fov = 2 * math.atan2(diagonal, fx + fy) * 180 / math.pi
    camera.set_projection_type("fisheyePolynomial")
    camera.set_rational_polynomial_properties(width, height, cx, cy, diagonal_fov, D)

    return camera
