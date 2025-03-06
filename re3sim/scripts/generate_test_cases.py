import numpy as np
import os
from real2sim2real.envs.config import SimulatorConfig
from real2sim2real.envs.env import BaseEnv

item_position_range = [[-0.3, -0.05], [0.0, 0.3]]
test_times = 100
save_path = "/isaac-sim/src/assets/test_cases/1113-multi-item.npy"

file_path = "/isaac-sim/src/configs/pick_and_place/collect_data_render_1113_eggplant_low_res_continuous_multi_item.yaml"
sim_config = SimulatorConfig(file_path)
headless = sim_config.config.headless
webrtc = False

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

from real2sim2real.tasks.multi_item_pick_and_place import (
    FrankaTableMultiItemPickAndPlaceForEnv,
)
from real2sim2real.randomize.item_pose_random import random_place_items


env.reset()
task: FrankaTableMultiItemPickAndPlaceForEnv = list(env.runner.current_tasks.values())[
    0
]


# set seed
np.random.seed(20241106)

item_pcds = []
items_orientation_ranges = []
for item in task._items:
    item_pcds.append(item.pcd)
    range_min = item.randomize_params["random_orientation"]["rotation_min"]
    range_max = item.randomize_params["random_orientation"]["rotation_max"]
    items_orientation_ranges.append(
        (
            (range_min[0], range_max[0]),
            (range_min[1], range_max[1]),
            (range_min[2], range_max[2]),
        )
    )
result = np.empty((test_times, len(task._items), 7))
item_num = len(task._items)
i = 0
while i < test_times:
    positions, orientations = random_place_items(
        task.table_pcd,
        item_position_range,
        item_pcds,
        items_orientation_ranges,
        height_offset=0.02,
    )
    if len(positions) != item_num:
        print(
            f"Test case {i} failed, positions length: {len(positions)}, item_num: {item_num}"
        )
        continue
    result[i, :, :3] = positions
    result[i, :, 3:] = orientations
    i += 1

print(f"Save test cases to {save_path}")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, result)
env.close()
