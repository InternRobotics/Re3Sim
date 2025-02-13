# -*- coding=utf-8 -*-
# Author: w61
# Date: 2024-7-11

"""
# This script converts all .obj files under the Matterport3D/data/v1/scans/xxx/matterport_mesh directory to .usd files.
# The script includes transformations for scale and rotation.
# Note: The ground plane is not initialized during the stage export due to an issue with `stage.Export`.
# Instead, the ground plane is initialized when loading the USD in the task.
"""

"""Launch Isaac Sim Simulator first."""
import time
import argparse
import shutil

# from omni.isaac.lab.app import AppLauncher

import yaml

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Process all OBJ files in the Matterport3D directory."
)
parser.add_argument("--obj_path", type=str, help="The path to the obj file.")
parser.add_argument("--usd_dir", type=str, help="The path to the output usd file.")
# parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "none", "meshSimplification"],
    help=(
        'The method used for approximating collision mesh. Set to "none" '
        "to not add a collision mesh to the converted mesh."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)
parser.add_argument(
    "--scan",
    type=str,
    default=None,
    help="The specific scan in Matterport3D to research",
)
parser.add_argument(
    "--adjust_height_to_ground",
    action="store_true",
    default=False,
    help="Adjust the height of the model to the ground.",
)
parser.add_argument(
    "--headless", action="store_true", default=False, help="headless mode (without GUI)"
)

import isaacsim
from omni.isaac.kit import SimulationApp

# Initialize the simulation app
args_cli = parser.parse_args()
simulation_app = SimulationApp(vars(args_cli))


"""Rest everything follows."""
import os

import carb

import omni.usd
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
import omni.kit.commands

import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app

from omni.isaac.lab.sim.converters import MeshConverter, MeshConverterCfg
from omni.isaac.lab.sim.schemas import schemas_cfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.sim.converters.asset_converter_base import AssetConverterBase

import asyncio

from omni.isaac.core.utils.extensions import enable_extension

from omni.isaac.lab.sim.converters.asset_converter_base import AssetConverterBase
from omni.isaac.lab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from omni.isaac.lab.sim.schemas import schemas
from omni.isaac.lab.sim.utils import export_prim_to_file
from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg

from pxr import UsdGeom, Usd, Sdf, Gf, UsdPhysics, UsdUtils


def adjust_model_to_ground(stage, model_path):
    # Get the bounding box information of the model
    model_prim = stage.GetPrimAtPath(model_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(model_prim)

    # Get the minimum and maximum values of the bounding box
    min_point = bbox.GetRange().GetMin()
    # max_point = bbox.GetRange().GetMax()

    # Calculate the position of the bottom of the model relative to the origin
    bottom_z = min_point[2]
    print(f"Bottom_z: {bottom_z}")

    # Adjust the position of the model to make its bottom touch the ground
    translate = Gf.Vec3f(0.0, 0.0, -bottom_z)
    # xform = UsdGeom.Xformable(model_prim)
    # xform.AddTranslateOp().Set(translate)
    translate_attr = model_prim.GetAttribute("xformOp:translate")
    translate_attr.Set(translate)


def copy_and_rename_obj_file(obj_file_path):
    """The obj file is renamed because the Converter cannot be called with an asset that starts with a number"""
    # Get the directory and file name of the original OBJ file
    obj_dir = os.path.dirname(obj_file_path)
    obj_file_name = os.path.basename(obj_file_path)

    # Define the new file name with the "modified_" prefix
    modified_obj_file_name = "isaacsim_" + obj_file_name
    modified_obj_file_path = os.path.join(obj_dir, modified_obj_file_name)

    # Copy the original OBJ file to the new file with the "modified_" prefix
    shutil.copyfile(obj_file_path, modified_obj_file_path)

    return modified_obj_file_path


class MP3DMeshConverter(MeshConverter):
    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(
        in_file: str,
        out_file: str,
        prim_path: str = "/World",
        load_materials: bool = True,
    ) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to meters.

        The asset hierarchy is arranged as follows:

        .. code-block:: none
            prim_path (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            prim_path: The prim path of the mesh.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        enable_extension("omni.kit.asset_converter")
        enable_extension("omni.usd.metrics.assembler")

        import omni.kit.asset_converter
        import omni.usd
        from omni.metrics.assembler.core import get_metrics_assembler_interface

        # Create converter context
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # Set up converter settings
        # Don't import/export materials
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        # Merge all meshes into one
        converter_context.merge_all_meshes = False
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        # This does not work right now :(, so we need to scale the mesh manually
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        # Uses double precision for all transform ops.
        converter_context.use_double_precision_to_usd_transform_op = True

        # Create converter task
        instance = omni.kit.asset_converter.get_instance()
        out_file_non_metric = out_file.replace(".usd", "_non_metric.usd")
        task = instance.create_converter_task(
            in_file, out_file_non_metric, None, converter_context
        )
        # Start conversion task and wait for it to finish
        success = True
        while True:
            success = await task.wait_until_finished()
            if not success:
                await asyncio.sleep(0.1)
            else:
                break

        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)

        base_prim = temp_stage.DefinePrim(prim_path, "Xform")
        prim = temp_stage.DefinePrim(f"{prim_path}/geometry", "Xform")
        prim.GetReferences().AddReference(out_file_non_metric)
        cache = UsdUtils.StageCache.Get()
        cache.Insert(temp_stage)
        stage_id = cache.GetId(temp_stage).ToLongInt()
        get_metrics_assembler_interface().resolve_stage(stage_id)
        temp_stage.SetDefaultPrim(base_prim)

        # change prims' parameters
        mp3d_prim = prim
        # Set Rotate:unitsResolve to 0
        rotate_units_resolve_attr = mp3d_prim.GetAttribute(
            "xformOp:rotateX:unitsResolve"
        )
        if not rotate_units_resolve_attr:
            rotate_units_resolve_attr = mp3d_prim.CreateAttribute(
                "xformOp:rotateX:unitsResolve", Sdf.ValueTypeNames.Int
            )
        rotate_units_resolve_attr.Set(0)

        # Set Scale:unitsResolve to [1, 1, 1]
        scale_units_resolve_attr = mp3d_prim.GetAttribute("xformOp:scale:unitsResolve")
        if not scale_units_resolve_attr:
            scale_units_resolve_attr = mp3d_prim.CreateAttribute(
                "xformOp:scale:unitsResolve", Sdf.ValueTypeNames.Float3
            )
        scale_units_resolve_attr.Set(Gf.Vec3f(1, 1, 1))

        # modify the height to the ground
        if args_cli.adjust_height_to_ground:
            adjust_model_to_ground(temp_stage, mp3d_prim.GetPath())

        # Note: I don't know why this does not work after exporting and re-loading.
        # create physics material
        # physics_material = RigidBodyMaterialCfg(
        #     static_friction=0.5,
        #     dynamic_friction=0.5,
        #     restitution=0.0,
        #     improve_patch_friction=True,
        #     friction_combine_mode='average',
        #     restitution_combine_mode='average',
        #     compliant_contact_stiffness=0.0,
        #     compliant_contact_damping=0.0
        # )
        # physics_material_cfg: sim_utils.RigidBodyMaterialCfg = physics_material
        # # spawn the material
        # physics_prim = temp_stage.DefinePrim(f"{prim_path}/physicsMaterial", "Xform")
        # physics_material_cfg.func(f"{prim_path}/physicsMaterial", physics_material)
        # sim_utils.bind_physics_material(f"{prim_path}/geometry", f"{prim_path}/physicsMaterial", stage=temp_stage)

        # Collisions
        # Ensure the prim exists before defining collision properties
        geometry_prim_path = f"{prim_path}/geometry"
        if not temp_stage.GetPrimAtPath(geometry_prim_path):
            raise ValueError(f"Prim path '{geometry_prim_path}' does not exist.")
        collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        sim_utils.define_collision_properties(
            geometry_prim_path, collider_cfg, stage=temp_stage
        )

        # Add collision to the prim
        # collision_api = UsdPhysics.CollisionAPI.Apply(mp3d_prim)
        # # Define collision geometry (you can use Box, Sphere, Capsule, Mesh, etc.)
        # collision_prim = temp_stage.DefinePrim(f"{prim_path}/geometry/collision", "Mesh")
        # collision_api.CreateCollisionShapeAttr().Set(collision_prim.GetPath())

        # add ground plane
        # ground_plane_prim = temp_stage.DefinePrim(f"{prim_path}/GroundPlane", "Xform")
        # ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=physics_material)
        # ground_plane = ground_plane_cfg.func(f"{prim_path}/GroundPlane", ground_plane_cfg)
        # ground_plane.visible = False

        temp_stage.Export(out_file)
        return success


def process_obj_file(obj_file_path):
    # Ensure the mesh path is absolute
    if not os.path.isabs(obj_file_path):
        obj_file_path = os.path.abspath(obj_file_path)

    if not check_file_path(obj_file_path):
        raise ValueError(f"Invalid mesh file path: {obj_file_path}")

    # Determine the destination path for the USD file
    dest_path = os.path.splitext(obj_file_path)[0] + ".usd"
    usd_dir = args_cli.usd_dir
    usd_file_name = os.path.basename(dest_path)
    print("usd_dir: ", usd_dir)
    # Mass properties
    mass_props = (
        schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
        if args_cli.mass is not None
        else None
    )
    rigid_props = (
        schemas_cfg.RigidBodyPropertiesCfg() if args_cli.mass is not None else None
    )

    # Collision properties
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=args_cli.collision_approximation != "none"
    )

    # Create Mesh converter config
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=obj_file_path,
        force_usd_conversion=True,
        usd_dir=usd_dir,
        usd_file_name=usd_file_name,
        make_instanceable=args_cli.make_instanceable,
        collision_approximation=args_cli.collision_approximation,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input Mesh file: {obj_file_path}")
    print("Mesh importer config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Mesh converter and import the file
    mesh_converter = MP3DMeshConverter(mesh_converter_cfg)
    # Print output
    print("Mesh importer output:")
    print(f"Generated USD file: {mesh_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)


def main():
    process_obj_file(args_cli.obj_path)

    print("==========Convert finished!=============")


if __name__ == "__main__":
    main()
