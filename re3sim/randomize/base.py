from typing import Dict, Any, Callable, List, Tuple
from abc import ABC, abstractmethod
from functools import wraps

from ..tasks.task import BaseTask


class Randomize(ABC):
    randomize_dict = {}

    def __init__(self, random_params: Dict[str, Any]):
        self.random_params = random_params
        self.name = "BaseRandom"

    @abstractmethod
    def randomize(self, task: BaseTask) -> Any:
        raise NotImplementedError

    @abstractmethod
    # a method to apply a former randomize result to the task
    def apply_random_result(self, task: BaseTask, random_result: Any) -> None:
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        def decorator(randomize_class):
            cls.randomize_dict[name] = randomize_class

            @wraps(randomize_class)
            def wrapped_function(*args, **kwargs):
                return randomize_class(*args, **kwargs)

            return wrapped_function

        return decorator
