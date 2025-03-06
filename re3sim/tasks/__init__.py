from .task import BaseTask, create_task
from .metric import *

try:
    from .pick_and_place import *
    from .eval_pick_and_place import *
except Exception as e:
    import traceback

    print(traceback.format_exc())
    from .pick_and_place_extension import *
