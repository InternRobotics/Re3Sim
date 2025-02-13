from copy import deepcopy
from typing import List, Optional, Union, Any
from typing import List, Dict, Tuple
from pydantic import BaseModel
import carb
import yaml
from real2sim2real.utils.logger import log
import numpy as np


class MetricUserConfig(BaseModel):
    """
    MetricUserConfig
    """

    type: str
    name: Optional[str]


class TaskUserConfig(BaseModel):
    type: str
    name: str

    # scene
    scene_asset_path: Optional[str] = None
    scene_scale: Optional[List[float]] = [1.0, 1.0, 1.0]
    scene_position: Optional[List[float]] = [0, 0, 0]
    scene_orientation: Optional[List[float]] = [1.0, 0, 0, 0]

    # inherit
    robots: Optional[List[Any]] = []
    objects: Optional[List[Any]] = []
    metrics: Optional[List[MetricUserConfig]] = []

    # path
    root_path: str
    scene_root_path: str = "/scene"
    robots_root_path: str = "/robots"
    objects_root_path: str = "/objects"

    # offset
    offset: Optional[List[float]] = None
    offset_size: float = 10.0

    # id
    env_id: int = 0

    params: Optional[Dict[str, Any]] = None
    items: List[Dict[str, Any]] = []
    place_items: List[Dict[str, Any]] = []
    cameras: List[Dict[str, Any]] = []
    background: Optional[Dict[str, Any]] = None


class Env(BaseModel):
    """
    Env config
    """

    # background config(type None for nothing)
    bg_type: Union[str, None] = None
    bg_path: Optional[Union[str, None]] = None


class SimConfig(BaseModel):
    """
    SimConfig
    """

    physics_dt: Optional[float | str] = None
    rendering_dt: Optional[float | str] = None
    rendering_interval: Optional[int] = None


class Config(BaseModel):
    """
    Config
    """

    simulator: Optional[SimConfig]
    env_set: Optional[Env] = None
    tasks: List[TaskUserConfig]
    data_log_root_path: Optional[str] = ""
    save_depth: Optional[bool] = False
    global_params: Optional[Dict[str, Any]] = {}
    render: Optional[bool] = True
    headless: Optional[bool] = False
    save_depth_as_png: Optional[bool] = False
    # npc: List[NPCUserConfig] = []


class Scene(BaseModel):
    type: str
    name: Optional[str]
    path: Optional[str]


class ControllerParams(BaseModel):
    """
    Controller config validator
    """

    name: str
    joint_names: Optional[List[str]]
    map_data_path: Optional[str]  # navi only, npy BOG (binary occupancy grid) file
    reference: Optional[str]  # ik only, world/robot/arm_base, default to world
    threshold: Optional[float]  # threshold to judge if action has been finished.

    # Planner controller
    planner: Optional[str]  # for planning policy.
    model: Optional[str]  # for planning policy model
    model_path: Optional[str]  # for planning policy, weight path of model


class UsdObj(BaseModel):
    usd_path: str


class DynamicCube(BaseModel):
    color: Optional[List[float]] = None


class ObjectConfig(BaseModel):
    # common
    name: str
    prim_path: str
    position: Optional[List[float]] = [0.0, 0.0, 0.0]
    scale: Optional[List[float]] = [1.0, 1.0, 1.0]

    # physics
    mass: Optional[float] = None
    density: Optional[float] = None
    collider: Optional[bool] = True

    # Set type in ["UsdObj", "DynamicCube"]
    # If not, raise error
    type: str

    # params for each type of
    usd_obj_param: Optional[UsdObj] = None
    color: Optional[List[float]] = None
    dynamic_cube_param: Optional[DynamicCube] = None


class SimulatorConfig:

    def __init__(self, path: str = None):
        """

        Args:
            path: config file path
        """
        self.env_num = 1
        self.offset_size = None
        self.config_file_path = path
        self.config_dict = None
        self.config: Config = self.validate(self.get_config_from_file())

    def get_config_from_file(self):
        if self.config_file_path:
            if not self.config_file_path.endswith(
                ".yaml"
            ) or self.config_file_path.endswith(".yml"):
                log.error("config file not end with .yaml or .yml")
                raise FileNotFoundError("config file not end with .yaml or .yml")
            with open(self.config_file_path, "r") as f:
                self.config_dict = yaml.load(f.read(), yaml.FullLoader)
            return self.config_dict
        log.error("Config file path is not set")
        raise FileNotFoundError("Config file path is not set")

    def validate(self, config_dict: dict) -> Config:
        data_log_root_path = config_dict.get("data_log_root_path", "")
        save_depth = config_dict.get("save_depth", False)
        save_depth_as_png = config_dict.get("save_depth_as_png", False)
        global_params = config_dict.get("global_params", {})
        _env = None
        _render = None
        _headless = config_dict.get("headless", False)
        self.env_num = 0
        if "tasks" not in config_dict:
            raise KeyError("tasks are not set in config path")

        if "render" in config_dict:
            _render = config_dict["render"]
        else:
            _render = True

        if "env" in config_dict:
            _env = config_dict["env"]
        else:
            _env = {}

        if "npc" in config_dict:
            _npc = config_dict["npc"]
        else:
            _npc = []

        for _task in config_dict["tasks"]:
            self.env_num += _task["env_num"]
            if "offset_size" in _task and isinstance(_task["offset_size"], float):
                if self.offset_size is None:
                    self.offset_size = _task["offset_size"]
                else:
                    self.offset_size = max(self.offset_size, _task["offset_size"])
        _column_length = int(np.sqrt(self.env_num))
        if self.offset_size is None:
            # default
            self.offset_size = 10.0

        env_count = 0
        tasks = []
        for _task in config_dict["tasks"]:
            for i in range(_task["env_num"]):
                task_copy = deepcopy(_task)

                row = int(i // _column_length)
                column = i % _column_length
                offset = [row * self.offset_size, column * self.offset_size, 0]

                task_copy["name"] = f"{task_copy['name']}_{env_count}"

                task_copy["root_path"] = f"/World/env_{env_count}"
                task_copy["env_id"] = env_count

                task_copy["offset"] = offset
                if "scene_root_path" not in _task:
                    task_copy["scene_root_path"] = "/scene"
                if "robots_root_path" not in _task:
                    task_copy["robots_root_path"] = "/robots"
                if "objects_root_path" not in _task:
                    task_copy["objects_root_path"] = "/objects"

                for r in task_copy["robots"]:
                    r["name"] = f"{r['name']}_{env_count}"
                    r["prim_path"] = (
                        task_copy["root_path"]
                        + task_copy["robots_root_path"]
                        + r["prim_path"]
                    )
                    r["position"] = [
                        task_copy["offset"][idx] + i
                        for idx, i in enumerate(r["position"])
                    ]
                if "objects" in task_copy and task_copy["objects"] is not None:
                    for o in task_copy["objects"]:
                        o["name"] = f"{o['name']}_{env_count}"
                        o["prim_path"] = (
                            task_copy["root_path"]
                            + task_copy["objects_root_path"]
                            + o["prim_path"]
                        )
                        o["position"] = [
                            task_copy["offset"][idx] + i
                            for idx, i in enumerate(o["position"])
                        ]
                items = task_copy.get("items", [])
                for item in items:
                    item["name"] = f"{item['name']}_{env_count}"
                    item["prim_path"] = (
                        task_copy["root_path"]
                        + task_copy["objects_root_path"]
                        + item["prim_path"]
                    )
                task_copy["items"] = items
                place_items = task_copy.get("place_items", [])
                for place_item in place_items:
                    place_item["name"] = f"{place_item['name']}_{env_count}"
                    place_item["prim_path"] = (
                        task_copy["root_path"]
                        + task_copy["objects_root_path"]
                        + place_item["prim_path"]
                    )
                task_copy["place_items"] = place_items

                print(task_copy["items"])
                for camera in task_copy.get("cameras", []):
                    camera["name"] = f"{camera['name']}_{env_count}"
                tasks.append(TaskUserConfig(**task_copy))
                env_count += 1

                background = task_copy.get("background", None)

        # log.debug(tasks)
        return Config(
            simulator=SimConfig(**config_dict["simulator"]),
            render=_render,
            env_set=Env(**_env),
            tasks=tasks,
            npc=_npc,
            data_log_root_path=data_log_root_path,
            global_params=global_params,
            headless=_headless,
            background=background,
            save_depth=save_depth,
            save_depth_as_png=save_depth_as_png,
        )


class SensorParams(BaseModel):
    """
    Sensor config validator
    """

    name: str
    enable: Optional[bool] = True
    size: Optional[Tuple[int, int]]  # Camera only
    scan_rate: Optional[int]  # RPS. Lidar only


class RobotUserConfig(BaseModel):
    # meta info
    name: str
    type: str
    prim_path: str
    create_robot: bool = True

    # common config
    position: Optional[List[float]] = [0.0, 0.0, 0.0]
    orientation: Optional[List[float]]
    scale: Optional[List[float]]

    # Parameters
    controller_params: Optional[List[ControllerParams]] = None
    sensor_params: Optional[List[SensorParams]] = None
