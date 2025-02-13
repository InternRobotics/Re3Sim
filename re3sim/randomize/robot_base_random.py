from .base import Randomize
from typing import Dict, Any
from ..tasks.task import BaseTask
import numpy as np
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import get_prim_parent, get_prim_at_path


@Randomize.register("RobotBaseRandom")
class RobotBaseRandom(Randomize):
    def __init__(self, random_params: Dict[str, Any]):
        super().__init__(random_params)
        self.name = "RobotBaseRandom"
        self.position_min = random_params["position_min"]
        self.position_max = random_params["position_max"]

    def randomize(self, task: BaseTask):
        position = np.random.uniform(self.position_min, self.position_max)
        franka_prim = get_prim_at_path(task.robot.prim_path)
        franka_parent_prim = get_prim_parent(franka_prim)
        franka_parent_prim = XFormPrim(str(franka_parent_prim.GetPrimPath()))
        franka_parent_prim.set_local_pose(translation=position)
        return position.tolist()

    def apply_random_result(self, task: BaseTask, random_result: Any) -> None:
        position = np.array(random_result)
        franka_prim = get_prim_at_path(task.robot.prim_path)
        franka_parent_prim = get_prim_parent(franka_prim)
        franka_parent_prim = XFormPrim(str(franka_parent_prim.GetPrimPath()))
        franka_parent_prim.set_local_pose(translation=position)
