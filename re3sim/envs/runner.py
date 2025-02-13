from typing import List
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa F401
from omni.physx.scripts import utils
from pxr import Usd  # noqa

# Init
from ..envs.config import SimulatorConfig, TaskUserConfig, ObjectConfig, Scene
from ..tasks import BaseTask, create_task
from ..envs.scene import create_scene, create_object
from ..utils.logger import log, get_log_folder, get_json_log_path
from ..utils.utils import remove_ndarray_in_dict
from ..utils.prim import filter_collisions
from omni.isaac.core.loggers.data_logger import DataLogger
from pathos.multiprocessing import ThreadingPool as Pool
from pathos import multiprocessing
import tqdm


class SimulatorRunner:

    def __init__(self, config: SimulatorConfig, to_log: bool = False) -> None:

        self._simulator_config = config.config
        physics_dt = (
            self._simulator_config.simulator.physics_dt
            if self._simulator_config.simulator.physics_dt is not None
            else None
        )
        rendering_dt = (
            self._simulator_config.simulator.rendering_dt
            if self._simulator_config.simulator.rendering_dt is not None
            else None
        )
        physics_dt = eval(physics_dt) if isinstance(physics_dt, str) else physics_dt
        rendering_dt = (
            eval(rendering_dt) if isinstance(rendering_dt, str) else rendering_dt
        )
        self.dt = physics_dt
        log.debug(f"Simulator physics dt: {self.dt}")
        self._world = World(
            physics_dt=self.dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0
        )
        self._scene = self._world.scene
        self._stage = self._world.stage
        self._render = self._simulator_config.render

        # setup scene
        prim_path = "/"
        if self._simulator_config.env_set.bg_type is None:
            self._ground_plane = self._scene.add(
                GroundPlane(prim_path="/World/groundPlane")
            )
            create_prim(
                "/World/Light",
                "DomeLight",
                attributes={
                    # "inputs:radius": 0.01,
                    "inputs:intensity": 1e3,
                    "inputs:color": (0.75, 0.75, 0.75),
                },
            )
            # self._scene.add_default_ground_plane()
        elif self._simulator_config.env_set.bg_type != "default":
            source, prim_path = create_scene(
                self._simulator_config.env_set.bg_path, prim_path_root="background"
            )
            add_reference_to_stage(source, prim_path)

        self.render_interval = (
            self._simulator_config.simulator.rendering_interval
            if self._simulator_config.simulator.rendering_interval is not None
            else 5
        )
        print(f"rendering interval: {self.render_interval}")
        self.render_trigger = 0
        self._to_log = to_log
        self._data_log_dir_path = None
        if self._simulator_config.data_log_root_path:
            self._data_log_dir_path = get_log_folder(
                self._simulator_config.data_log_root_path
            )
        self.npc = []
        self._data_loggers = {}
        self._last_obs = {}
        self._filter_collisions = True
        self._collision_root_path = "/World/collisions"

    @property
    def current_tasks(self) -> dict[str, BaseTask]:
        return self._world._current_tasks

    def _warm_up(self, steps=10, render=True):
        with tqdm.tqdm(total=steps, desc="Warming up") as pbar:
            for _ in range(steps):
                self._world.step(render=render and self._render)
                pbar.update(1)

    def _init_data_loggers(self):
        if self._data_log_dir_path:
            for task_name, task in self.current_tasks.items():
                self._data_loggers[task_name] = DataLogger()

    def add_tasks(self, configs: List[TaskUserConfig]):
        p = Pool(multiprocessing.cpu_count())
        tasks = p.map(lambda config: create_task(config, self._scene), configs)
        with tqdm.tqdm(total=len(tasks), desc="Initializing tasks") as pbar:
            for task in tasks:
                self._world.add_task(task)
                pbar.update(1)

        self._world.reset()
        self._warm_up()
        if self._to_log:
            self._init_data_loggers()

        # filter collisions
        if self._filter_collisions:
            global_collision_paths = []
            prim_paths = []
            global_collision_paths.append("/World/groundPlane")
            for task_name, task in self.current_tasks.items():
                prim_paths.append(task.config.root_path + "/background")
                prim_paths.append(task.config.root_path + "/robots")
                global_collision_paths.append(task.config.root_path + "/objects")
            filter_collisions(
                self._stage,
                self._world.get_physics_context().prim_path,
                self._collision_root_path,
                prim_paths,
                global_collision_paths,
            )

    def step(self, actions: dict, render: bool = True, only_step=False):
        if only_step:
            self._world.step(render=False)
            return
        for task_name, action_dict in actions.items():
            task = self.current_tasks.get(task_name)
            task.apply_action(action_dict)
        self.render_trigger += 1
        render = render and self.render_trigger >= self.render_interval and self._render
        if self.render_trigger > self.render_interval:
            self.render_trigger = 0
        if self._data_loggers and self._to_log:
            tmp_obs = self._last_obs
            for task_name, _ in actions.items():
                data_logger: DataLogger = self._data_loggers[task_name]
                data = {"obs": {}, "actions": actions[task_name]}
                for k, v in tmp_obs[task_name].items():
                    if k != "info":
                        data["obs"][k] = v
                data = remove_ndarray_in_dict(data)
                data_logger.add_data(
                    data, self._world.current_time_step_index, self._world.current_time
                )
        self._world.step(render=render)

    def get_obs(self):
        obs = {}
        for task_name, task in self.current_tasks.items():
            obs[task_name] = task.get_observations()
            obs[task_name]["current_world_time"] = self._world.current_time
        self._last_obs = obs
        return obs

    def get_current_time_step_index(self) -> int:
        return self._world.current_time_step_index

    def reset(self, tasks: List[str] = None):
        if tasks is None:
            self._world.reset()

            for task in self.current_tasks.values():
                if not task._initialized:
                    task.initialize()
            return
        for task in tasks:
            self.current_tasks[task].individual_reset()

    def get_obj(self, name: str) -> XFormPrim:
        return self._world.scene.get_object(name)

    def remove_collider(self, prim_path: str):
        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.removeCollider(build)

    def add_collider(self, prim_path: str):
        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.setCollider(build, approximationShape=None)

    def log_data(self, task_name: str):
        log_path = get_json_log_path(self._data_log_dir_path)
        if self._to_log:
            assert (
                task_name in self._data_loggers
            ), f"No data logger for task {task_name}"
            log_path = get_json_log_path(self._data_log_dir_path)
            self._data_loggers[task_name].save(log_path)
        return log_path

    def reset_data_logger(self, task_name: str):
        if self._to_log:
            assert (
                task_name in self._data_loggers
            ), f"No data logger for task {task_name}"
            self._data_loggers[task_name].reset()
