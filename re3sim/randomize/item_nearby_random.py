from typing import Dict, Any, Callable, List, Tuple
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from . import Randomize
from ..tasks.task import BaseTask
from scipy.spatial.transform import Rotation as R


@Randomize.register("ItemNearbyRandom")
class ItemNearbyRandom(Randomize):
    def __init__(self, random_params: Dict[str, Any]):
        super().__init__(random_params)
        item_random_range = random_params["item_random_range"]
        self.x_range = item_random_range[0]
        self.y_range = item_random_range[1]
        self.name = "ItemNearbyRandom"

    def randomize(self, task: BaseTask):
        items_rotation_range = []
        positions = []
        orientations = []
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
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])
            positions.append([x, y, 0])
            orientations.append(np.random.uniform(range_min, range_max))

        for item, position, orientation in zip(task.items, positions, orientations):
            item.object.set_local_pose(
                translation=position + item.object.get_local_pose()[0],
                orientation=(
                    R.from_euler("xyz", orientation)
                    * R.from_quat(item.object.get_local_pose()[1], scalar_first=True)
                ).as_quat(scalar_first=True),
            )
        return np.array(positions).tolist(), np.array(orientations).tolist()

    def apply_random_result(self, task: BaseTask, random_result: Any) -> None:
        positions, orientations = random_result
        positions = np.array(positions)
        orientations = np.array(orientations)
        for item, position, orientation in zip(task.items, positions, orientations):
            item.xform.set_local_pose(translation=position, orientation=orientation)
