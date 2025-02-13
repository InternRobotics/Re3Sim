from pxr import Usd, UsdGeom, Sdf, Gf, Vt, UsdShade, UsdPhysics, PhysxSchema, UsdUtils
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    tf_matrices_from_poses,
    pose_from_tf_matrix,
    get_relative_transform,
)
import numpy as np
from copy import deepcopy
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Sequence
import os
import tqdm
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

import base64


def to_list(data):
    res = []
    if data is not None:
        res = [_ for _ in data]
    return res


def recursive_parse(prim):
    # import open3d as o3d
    # print(prim.GetPath())
    translation = prim.GetAttribute("xformOp:translate").Get()
    if translation is None:
        translation = np.zeros(3)
    else:
        translation = np.array(translation)

    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is None:
        scale = np.ones(3)
    else:
        scale = np.array(scale)

    orient = prim.GetAttribute("xformOp:orient").Get()
    rotation = prim.GetAttribute("xformOp:rotateXYZ").Get()
    if orient is None:
        if rotation is None:
            orient = np.zeros([4, 1])
            orient[0] = 1.0
        else:
            orient = np.array(
                R.from_euler("xyz", rotation, degrees=True).as_quat(scalar_first=True)
            ).reshape(4, 1)
    else:
        # print(orient)
        r = orient.GetReal()
        i, j, k = orient.GetImaginary()

        orient = np.array([r, i, j, k]).reshape(4, 1)

    transform = prim.GetAttribute("xformOp:transform").Get()
    if transform is None:
        transform = np.eye(4)
    else:
        transform = np.array(transform)

    rotation_matrix = R.from_quat(orient.reshape(4), scalar_first=True).as_matrix()

    points_total = []
    faceuv_total = []
    normals_total = []
    faceVertexCounts_total = []
    faceVertexIndices_total = []
    mesh_total = []
    if prim.IsA(UsdGeom.Mesh):
        mesh_path = str(prim.GetPath()).split("/")[-1]
        if not mesh_path == "SM_Dummy":
            mesh_total.append(mesh_path)
            points = prim.GetAttribute("points").Get()
            normals = prim.GetAttribute("normals").Get()
            faceVertexCounts = prim.GetAttribute("faceVertexCounts").Get()
            faceVertexIndices = prim.GetAttribute("faceVertexIndices").Get()
            faceuv = prim.GetAttribute("primvars:st").Get()
            normals = to_list(normals)
            faceVertexCounts = to_list(faceVertexCounts)
            faceVertexIndices = to_list(faceVertexIndices)
            faceuv = to_list(faceuv)
            points = to_list(points)
            ps = []
            for p in points:
                x, y, z = p
                p = np.array((x, y, z))
                ps.append(p)

            points = ps

            base_num = len(points_total)
            for idx in faceVertexIndices:
                faceVertexIndices_total.append(base_num + idx)

            faceVertexCounts_total += faceVertexCounts
            faceuv_total += faceuv
            normals_total += normals
            points_total += points

    # else:

    children = prim.GetChildren()

    for child in children:

        points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_list = (
            recursive_parse(child)
        )
        # child_path = child.GetPath()
        # if len(normals) > len(points):
        #     print(f"points is less than their normals, the prim is {child_path}")
        #     print(len(points), len(normals), len(points_total), len(normals_total), len(faceVertexCounts))

        base_num = len(points_total)
        for idx in faceVertexIndices:
            faceVertexIndices_total.append(base_num + idx)

        faceVertexCounts_total += faceVertexCounts
        faceuv_total += faceuv
        normals_total += normals[: len(points)]
        points_total = np.concatenate(
            [np.array(points_total).reshape(-1, 3), np.array(points).reshape(-1, 3)],
            axis=0,
        )
        mesh_total += mesh_list

    new_points = []
    points_total = np.array(points_total).T
    if len(points_total) > 0:
        points_total = points_total * scale.reshape(3, 1)
        points_total = np.matmul(rotation_matrix, points_total)
        points_total += translation.reshape(3, 1)
        new_points = points_total.T
        points_mat = np.ones((len(new_points), 4)).astype(np.float32)
        points_mat[:, :3] = np.array(new_points)
        # print(points_mat.shape, transform.shape)
        points_mat = np.matmul(points_mat, transform)
        new_points = points_mat[:, :3]

    return (
        new_points,
        faceuv_total,
        normals_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
        mesh_total,
    )


def extract_obj_mesh(stage, scene_name, black_list=[]):
    bbox_list = []
    mesh_paths = []

    meshes = stage.GetPrimAtPath("/Root/Meshes")
    for scope in tqdm.tqdm(meshes.GetChildren()):
        # if flag:
        #     break
        scope_name = scope.GetName()
        for category in scope.GetChildren():
            cate = category.GetName()
            # print('cate', cate)
            instances = category.GetChildren()

            if not cate in black_list:
                # if cate in allow_list:
                path_prefix = os.path.join("data/models", cate)
                if not os.path.exists(path_prefix):
                    os.makedirs(path_prefix)
                for inst in instances:
                    filename = str(inst.GetPath()).split("/")[-1]
                    # obj_prefix = os.path.join(path_prefix, filename)
                    # mat_prefix = os.path.join(obj_prefix, "mtl")
                    # if not os.path.exists(obj_prefix):
                    #     os.makedirs(obj_prefix)
                    # if not os.path.exists(mat_prefix):
                    #     os.makedirs(mat_prefix)

                    (
                        points_total,
                        faceuv_total,
                        normals_total,
                        faceVertexCounts_total,
                        faceVertexIndices_total,
                        mesh_total,
                    ) = recursive_parse(inst)

                    mesh_total = sorted(mesh_total)
                    mesh_string = ""
                    for s in mesh_total:
                        mesh_string += s
                    item = {
                        "instance_id": filename,
                        "object_type": cate,
                        "scope": scope_name,
                    }
                    max_point = np.array(points_total).max(0)
                    min_point = np.array(points_total).min(0)
                    item["min_point"] = [_ for _ in min_point]
                    item["max_point"] = [_ for _ in max_point]
                    item["mesh_hash"] = hash(mesh_string)

                    bbox_list.append(item)

                    mesh_path = os.path.join(path_prefix, f"{filename}.obj")

                    write_obj(
                        mesh_path,
                        None,
                        points_total,
                        faceuv_total,
                        normals_total,
                        faceVertexCounts_total,
                        faceVertexIndices_total,
                    )

                    mesh_paths.append(mesh_path)
                    # break

                    continue

        # break
    scene_path = os.path.join("data", "meta", scene_name)
    if not os.path.exists(scene_path):
        os.makedirs(scene_path)

    with open(os.path.join(scene_path, "bbox.json"), "w") as fp:
        json.dump(bbox_list, fp, indent=4)

    return mesh_paths


def write_obj(path, name, points, faceuv, normals, faceVertexCounts, faceVertexIndices):
    with open(path, "w") as fp:

        # mtl_path = f"./mtl/{name}.mtl"
        # fp.write(f"mtllib {mtl_path}\n\n")

        for p in points:
            x, y, z = p
            fp.write(f"v {x} {y} {z}\n")

        fp.write("\n\n")
        for p in faceuv:
            u, v = p
            fp.write(f"vt {u} {v}\n")

        fp.write("\n\n")
        for p in normals:
            x, y, z = p
            fp.write(f"vn {x} {y} {z}\n")

        # fp.write(f"usemtl {material_name}\n\n")
        idx = 0
        for n in faceVertexCounts:
            a, b, c = faceVertexIndices[idx : idx + n]
            a += 1
            b += 1
            c += 1
            idx += n
            # fp.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            fp.write(f"f {a} {b} {c}\n")

        fp.write("\n\n")


def get_points_at_path(
    prim_path, relative_frame_prim_path=None, change_father_prim=True
):
    """
    Get the points of the mesh at the given prim path.
    Parameters:
        prim_path: The path to the prim.
    Returns:
        points_total: The points of the mesh.
        faceuv_total: The faceuv of the mesh.
        normals_total: The normals of the mesh.
        faceVertexCounts_total: The faceVertexCounts of the mesh.
        faceVertexIndices_total: The faceVertexIndices of the mesh.
        mesh_total: The mesh of the mesh.
    """
    prim = get_prim_at_path(prim_path)
    (
        points_total,
        faceuv_total,
        normals_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
        mesh_total,
    ) = recursive_parse(prim)
    father_prim = prim.GetParent()
    if change_father_prim:
        transform = get_relative_transform(father_prim, prim)
        points_total = (
            np.matmul(transform[:3, :3], np.array(points_total).T).T + transform[:3, 3]
        )
    if relative_frame_prim_path is not None:
        local_to_world = get_relative_transform(
            get_prim_at_path(prim_path), get_prim_at_path(relative_frame_prim_path)
        )
        points_total = (
            np.matmul(local_to_world[:3, :3], np.array(points_total).T).T
            + local_to_world[:3, 3]
        )
    return points_total


def franka_picked_up(
    franka_prim_path,
    item_prim_path,
    table_pcd=None,
    threshold=0.1,
    table_pcd_frame_prim_path=None,
):  # to be checked
    """
    Check if the franka has picked up the item.
    Parameters:
        franka_prim_path: The path to the franka prim.
        item_prim_path: The path to the item prim.
    Returns:
        True if the franka has picked up the item, False otherwise.
    """
    if table_pcd_frame_prim_path is None:
        table_pcd_frame_prim_path = franka_prim_path
    left_finger_points = get_points_at_path(
        franka_prim_path + "/panda_leftfinger",
        relative_frame_prim_path=table_pcd_frame_prim_path,
    )
    right_finger_points = get_points_at_path(
        franka_prim_path + "/panda_rightfinger",
        relative_frame_prim_path=table_pcd_frame_prim_path,
    )
    item_points = get_points_at_path(
        item_prim_path, relative_frame_prim_path=table_pcd_frame_prim_path
    )
    gripper = np.concatenate([left_finger_points, right_finger_points], axis=0)
    gripper_bbox = np.concatenate([gripper.min(0), gripper.max(0)], axis=0)
    item_bbox = np.concatenate([item_points.min(0), item_points.max(0)], axis=0)
    # if has overlap then return True
    item_in_gripper = _is_overlap(gripper_bbox, item_bbox)
    if not item_in_gripper:
        return False
    if table_pcd is None:
        return True
    item_low_height = item_points[:, 2].min()
    item_xy_bbox = item_points[:, :2].min(0), item_points[:, :2].max(0)
    table_pcd_in_item_xy_bbox = (
        (table_pcd[:, 0] > item_xy_bbox[0][0])
        & (table_pcd[:, 0] < item_xy_bbox[1][0])
        & (table_pcd[:, 1] > item_xy_bbox[0][1])
        & (table_pcd[:, 1] < item_xy_bbox[1][1])
    )
    table_pcd_in_item_xy_bbox = table_pcd[table_pcd_in_item_xy_bbox]
    table_pcd_in_item_xy_bbox_max_height = table_pcd_in_item_xy_bbox[:, 2].max()
    if item_low_height > table_pcd_in_item_xy_bbox_max_height + threshold:
        return True
    return False


def _is_overlap(boxA, boxB):
    """
    Check if two 3D boxes overlap
    
    Args:
        boxA: First box coordinates [x1, y1, z1, x2, y2, z2]
        boxB: Second box coordinates [x1, y1, z1, x2, y2, z2]
        
    Returns:
        bool: True if boxes overlap, False otherwise
    """
    x1_A, y1_A, z1_A, x2_A, y2_A, z2_A = boxA
    x1_B, y1_B, z1_B, x2_B, y2_B, z2_B = boxB

    # Check overlap along x, y, z axes
    if (
        x1_A <= x2_B
        and x1_B <= x2_A  # X-axis overlap 
        and y1_A <= y2_B
        and y1_B <= y2_A  # Y-axis overlap
        and z1_A <= z2_B
        and z1_B <= z2_A  # Z-axis overlap
    ):
        return True
    return False


def update_physics_params_of_usd(stage, prim_path, mass, friction, restitution):
    """
    Update the physics parameters of the prim at the given prim path.
    Parameters:
        prim_path: The path to the prim.
        mass: The mass of the prim.
        friction: The friction of the prim.
        restitution: The restitution of the prim.
    """
    prim = get_prim_at_path(prim_path)
    physics_schema = UsdPhysics.PhysicsSceneAPI.Get(
        stage, stage.GetDefaultPrim().GetPath()
    )
    physics_schema.GetMassAttr().Set(mass)
    physics_schema.GetFrictionAttr().Set(friction)
    physics_schema.GetRestitutionAttr().Set(restitution)
    return


def filter_collisions(
    stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: List[str],
    global_paths: List[str] = [],
):
    """Filters collisions between clones. Clones will not collide with each other, but can collide with objects specified in global_paths.

    Args:
        physicsscene_path (str): Path to PhysicsScene object in stage.
        collision_root_path (str): Path to place collision groups under.
        prim_paths (List[str]): Paths of objects to filter out collision.
        global_paths (List[str]): Paths of objects to generate collision (e.g. ground plane).

    """

    physx_scene = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(physicsscene_path))

    # We invert the collision group filters for more efficient collision filtering across environments
    physx_scene.CreateInvertCollisionGroupFilterAttr().Set(True)

    collision_scope = UsdGeom.Scope.Define(stage, collision_root_path)

    with Sdf.ChangeBlock():
        if len(global_paths) > 0:
            global_collision_group_path = collision_root_path + "/global_group"
            # add collision group prim
            global_collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                "global_group",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            global_collision_group.SetInfo(
                Usd.Tokens.apiSchemas,
                Sdf.TokenListOp.Create({"CollectionAPI:colliders"}),
            )

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                global_collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            global_includes_rel = Sdf.RelationshipSpec(
                global_collision_group, "collection:colliders:includes", False
            )
            for global_path in global_paths:
                global_includes_rel.targetPathList.Append(global_path)

            # filteredGroups rel
            global_filtered_groups = Sdf.RelationshipSpec(
                global_collision_group, "physics:filteredGroups", False
            )
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            global_filtered_groups.targetPathList.Append(global_collision_group_path)

        # set collision groups and filters
        for i, prim_path in enumerate(prim_paths):
            collision_group_path = collision_root_path + f"/group{i}"
            # add collision group prim
            collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                f"group{i}",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            collision_group.SetInfo(
                Usd.Tokens.apiSchemas,
                Sdf.TokenListOp.Create({"CollectionAPI:colliders"}),
            )

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            includes_rel = Sdf.RelationshipSpec(
                collision_group, "collection:colliders:includes", False
            )
            includes_rel.targetPathList.Append(prim_path)

            # filteredGroups rel
            filtered_groups = Sdf.RelationshipSpec(
                collision_group, "physics:filteredGroups", False
            )
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            filtered_groups.targetPathList.Append(collision_group_path)
            if len(global_paths) > 0:
                filtered_groups.targetPathList.Append(global_collision_group_path)
                global_filtered_groups.targetPathList.Append(collision_group_path)


def disable_collision_recursively(stage, prim_path: str):
    """
    Disable collision for the prim at the given prim path and all its children.
    Parameters:
        prim_path: The path to the prim.
    """
    prim = get_prim_at_path(prim_path)
    for child in prim.GetChildren():
        disable_collision_recursively(stage, child.GetPath())
    try:
        collisionAPI = UsdPhysics.Get(stage, prim_path)
        collisionAPI.GetCollisionEnabledAttr().Set(False)
    except:
        pass


def check_overlap_by_pcd_bbox(
    pcd: np.ndarray, table_pcd: np.ndarray, placed_items_pcd: List[np.ndarray]
) -> bool:
    """
    Check for overlapping by comparing point cloud bounding boxes

    Args:
        pcd: Point cloud to check
        table_pcd: Point cloud of the table
        placed_items_pcd: List of point clouds for already placed items
        
    Returns:
        bool: True if overlap found, False otherwise
    """
    # Calculate bounding box of point cloud to check
    pcd_min = np.min(pcd, axis=0)
    pcd_max = np.max(pcd, axis=0)

    # Check overlap with already placed items
    for placed_pcd in placed_items_pcd:
        placed_min = np.min(placed_pcd, axis=0)
        placed_max = np.max(placed_pcd, axis=0)

        # Check overlap along x, y, z axes
        overlap = all(
            pcd_min[i] <= placed_max[i] and pcd_max[i] >= placed_min[i]
            for i in range(3)
        )

        if overlap:
            return True

    return False


def check_overlap_efficient(
    pcd: np.ndarray,
    table_tree: cKDTree,
    placed_items_pcd: List[np.ndarray],
    threshold: float = 0.01,
) -> bool:
    """
    Efficiently check if point cloud overlaps with table or placed items

    Args:
        pcd: Point cloud to check
        table_tree: KDTree of table point cloud
        placed_items_pcd: List of point clouds for placed items 
        threshold: Overlap threshold distance

    Returns:
        bool: True if overlap found, False otherwise
    """
    # Check overlap with table
    distances, _ = table_tree.query(pcd[:, :2], k=1)
    if np.any(distances < threshold):
        return True

    # Check overlap with placed items
    pcd_tree = cKDTree(pcd)
    for placed_pcd in placed_items_pcd:
        distances, _ = pcd_tree.query(placed_pcd, k=1)
        if np.any(distances < threshold):
            return True

    return False
