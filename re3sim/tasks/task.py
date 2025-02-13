# import random
import traceback
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, Tuple, List

from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask as OmniBaseTask
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.articulations import Articulation, ArticulationView

from ..envs.config import TaskUserConfig
from ..envs.scene import create_object, create_scene
from ..tasks.metric import BaseMetric, create_metric
from ..utils.logger import log
from ..utils.items import Item
from ..background import Background


class BaseTask(OmniBaseTask, ABC):
    """
    wrap of omniverse isaac sim's base task

    * enable register for auto register task
    * contains robots
    """

    tasks = {}

    def __init__(self, config: TaskUserConfig, scene: Scene):
        self.background: Background = None
        self.items: List[Item] = None
        self._root_prim_path = None
        name = config.name
        offset = config.offset
        super().__init__(name=name, offset=offset)
        self._scene = scene
        self.config = config

        self.metrics: dict[str, BaseMetric] = {}
        self.steps = 0
        self.work = True

        for metric_config in config.metrics:
            self.metrics[metric_config.name] = create_metric(metric_config)

    def initialize(self) -> None:
        pass

    def load(self):
        if self.config.scene_asset_path is not None:
            source, prim_path = create_scene(
                self.config.scene_asset_path,
                prim_path_root=f"World/env_{self.config.env_id}/scene",
            )
            create_prim(
                prim_path,
                usd_path=source,
                scale=self.config.scene_scale,
                translation=[
                    self.config.offset[idx] + i
                    for idx, i in enumerate(self.config.scene_position)
                ],
            )

        self.robot: Articulation = None
        # self.robots = init_robots(self.config, self._scene)
        for obj in self.config.objects:
            _object = create_object(obj)
            _object.set_up_scene(self._scene)
            self.objects[obj.name] = _object
        log.info(self.robots)
        log.info(self.objects)

    def set_up_scene(self, scene: Scene) -> None:
        self._scene = scene
        self.load()

    def get_observations(self) -> Dict[str, Any]:
        """
        Returns current observations from the objects needed for the behavioral layer.

        Return:
            Dict[str, Any]: observation of robots in this task
        """
        if not self.work:
            return {}
        obs = {}
        for robot_name, robot in self.robots.items():
            try:
                obs[robot_name] = robot.get_obs()
            except Exception as e:
                log.error(self.name)
                log.error(e)
                traceback.print_exc()
                return {}
        return obs

    def get_max_reward(self):
        return 10

    # @abstractmethod
    # def get_observations_for_controller(self):
    #     raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns True of the task is done.

        Raises:
            NotImplementedError: this must be overridden.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: Dict[str, Any]) -> None:
        """
        Apply action to the robots in the task.

        Args:
            action (Dict[str, Any]): action to apply to the robots.
        """
        raise NotImplementedError
        # for name, action in action.items():
        #     if name in self.robots:
        #         self.robots[name].apply_action(action)

    @abstractmethod
    def individual_reset(self):
        """
        reload this task individually without reloading whole world.
        """
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        """
        Register a task with its name(decorator).
        Args:
            name(str): name of the task
        """

        def decorator(tasks_class):
            cls.tasks[name] = tasks_class

            @wraps(tasks_class)
            def wrapped_function(*args, **kwargs):
                return tasks_class(*args, **kwargs)

            return wrapped_function

        return decorator

    @property
    def bg_pcd(self):
        raise NotImplementedError


def create_task(config: TaskUserConfig, scene: Scene):
    task_cls = BaseTask.tasks[config.type]
    return task_cls(config, scene)
