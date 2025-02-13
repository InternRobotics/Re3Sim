from .base import Randomize
from typing import Dict, Any
from ..tasks.task import BaseTask
from ..utils.prim import check_overlap_by_pcd_bbox
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple


def random_place_items(
    table_pcd: np.ndarray,
    random_range: Tuple[Tuple[float, float], Tuple[float, float]],
    item_pcds: List[np.ndarray],
    items_rotation_range: List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ],
    height_offset: float = 0.02,
):
    """
    Efficiently place items randomly on a table surface, ensuring the entire point cloud of each item
    stays within the specified range and doesn't overlap with the table or other items.

    Args:
        table_pcd: Point cloud of the table. Shape: (N, 3)
        random_range: Range for random placement. Shape: ((x_min, x_max), (y_min, y_max))
        item_pcds: List of item point clouds. Each element shape: (N, 3)
        items_rotation_range: Random rotation ranges for items. Shape: List[((roll_min, roll_max), (pitch_min, pitch_max), (yaw_min, yaw_max))]
        height_offset: Height offset for item placement

    Returns:
        item_positions, item_orientations: Positions and orientations of items. Shape: (M, 3) and (M, 4)
    """
    num_items = len(item_pcds)
    item_positions = []
    item_orientations = []
    placed_items_pcd = []

    table_tree = cKDTree(table_pcd[:, :2])

    x_range, y_range = random_range

    for i in range(num_items):
        item_pcd = item_pcds[i]
        rotation_range = items_rotation_range[i]

        item_min = np.min(item_pcd, axis=0)
        item_max = np.max(item_pcd, axis=0)
        item_size = item_max - item_min

        max_attempts = 100
        for _ in range(max_attempts):
            roll = np.random.uniform(rotation_range[0][0], rotation_range[0][1])
            pitch = np.random.uniform(rotation_range[1][0], rotation_range[1][1])
            yaw = np.random.uniform(rotation_range[2][0], rotation_range[2][1])
            rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=True)

            item_pcd_mean = np.mean(item_pcd, axis=0)
            rotated_item = rotation.apply(item_pcd - item_pcd_mean)
            rotated_min = np.min(rotated_item, axis=0)
            rotated_max = np.max(rotated_item, axis=0)
            rotated_size = rotated_max - rotated_min

            x = (
                np.random.uniform(
                    x_range[0] - rotated_min[0], x_range[1] - rotated_max[0]
                )
                - item_pcd_mean[0]
            )
            y = (
                np.random.uniform(
                    y_range[0] - rotated_min[1], y_range[1] - rotated_max[1]
                )
                - item_pcd_mean[1]
            )

            item_center_xy = np.array([x, y])
            xy_min = rotated_min[:2] + item_center_xy + item_pcd_mean[:2]
            xy_max = rotated_max[:2] + item_center_xy + item_pcd_mean[:2]

            table_pcd_in_xy_bool = (
                (table_pcd[:, 0] > xy_min[0])
                & (table_pcd[:, 0] < xy_max[0])
                & (table_pcd[:, 1] > xy_min[1])
                & (table_pcd[:, 1] < xy_max[1])
            )
            table_pcd_in_xy = table_pcd[table_pcd_in_xy_bool]

            if len(table_pcd_in_xy) > 0:
                table_pcd_in_xy_max_height = np.max(table_pcd_in_xy[:, 2])
                z = (
                    table_pcd_in_xy_max_height
                    + height_offset
                    - rotated_min[2]
                    - item_pcd_mean[2]
                )
            else:
                z = (
                    np.max(table_pcd[:, 2])
                    + height_offset
                    - rotated_min[2]
                    - item_pcd_mean[2]
                )

            transform = np.eye(4)
            transform[:3, :3] = rotation.as_matrix()
            transform[:3, 3] = [x, y, z]

            homogeneous_pcd = np.hstack((item_pcd, np.ones((item_pcd.shape[0], 1))))
            transformed_pcd = (transform @ homogeneous_pcd.T).T[:, :3]

            if not check_overlap_by_pcd_bbox(
                transformed_pcd, table_pcd, placed_items_pcd
            ):
                item_positions.append([x, y, z])
                item_orientations.append(rotation.as_quat(scalar_first=True))
                placed_items_pcd.append(transformed_pcd)
                break
        else:
            print(f"Warning: Cannot find a suitable position for item {i+1}")

    return np.array(item_positions), np.array(item_orientations)


@Randomize.register("ItemPoseRandom")
class ItemPoseRandom(Randomize):
    def __init__(self, random_params: Dict[str, Any]):
        super().__init__(random_params)
        item_random_range = random_params["item_random_range"]
        self.x_range = item_random_range[0]
        self.y_range = item_random_range[1]
        self.name = "ItemPoseRandom"

    def randomize(self, task: BaseTask):
        items_rotation_range = []
        item_pcds = []
        for item in task.items:
            range_min = item.randomize_params["random_orientation"]["rotation_min"]
            range_max = item.randomize_params["random_orientation"]["rotation_max"]
            items_rotation_range.append(
                (
                    (range_min[0], range_max[0]),
                    (range_min[1], range_max[1]),
                    (range_min[2], range_max[2]),
                )
            )
            item_pcds.append(item.pcd)
        positions, orientations = random_place_items(
            task.bg_pcd,
            (self.x_range, self.y_range),
            item_pcds,
            items_rotation_range,
            height_offset=0.02,
        )
        for item, position, orientation in zip(task.items, positions, orientations):
            item.xform.set_local_pose(translation=position, orientation=orientation)
        return positions.tolist(), orientations.tolist()

    def apply_random_result(self, task: BaseTask, random_result: Any) -> None:
        positions, orientations = random_result
        positions = np.array(positions)
        orientations = np.array(orientations)
        for item, position, orientation in zip(task.items, positions, orientations):
            item.xform.set_local_pose(translation=position, orientation=orientation)
