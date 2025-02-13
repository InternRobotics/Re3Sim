from functools import wraps
from ..envs.config import SimulatorConfig, TaskUserConfig, ObjectConfig, Scene


class ObjectCommon:
    """
    Object common class.
    """

    objs = {}

    def __init__(self, config: ObjectConfig):
        self._config = config

    def set_up_scene(self, scene: Scene):
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        """
        Register an object class with the given name(decorator).

        Args:
            name(str): name of the object
        """

        def decorator(object_class):
            cls.objs[name] = object_class

            @wraps(object_class)
            def wrapped_function(*args, **kwargs):
                return object_class(*args, **kwargs)

            return wrapped_function

        return decorator


def create_object(config: ObjectConfig):
    """
    Create an object.
    Args:
        config (ObjectConfig): configuration of the objects
    """
    assert config.type in ObjectCommon.objs, "unknown objects type {}".format(
        config.type
    )
    cls = ObjectCommon.objs[config.type]
    return cls(config)


def create_scene(config_json_path: str, prim_path_root: str = "background"):
    """
    Create a scene from config.(But just input usd file yet.)
    Args:
        config_json_path (str): path to scene config file(use to be a .usd file)
        prim_path_root (str): path to root prim

    Returns:
        config_json_path (str): path to config file
        world_prim_path (str): path to world prim
    """
    world_prim_path = "/" + prim_path_root
    if (
        config_json_path.endswith("usd")
        or config_json_path.endswith("usda")
        or config_json_path.endswith("usdc")
    ):
        # Add usd directly
        return config_json_path, world_prim_path
    raise RuntimeError("Env file path needs to end with .usd, .usda or .usdc .")
