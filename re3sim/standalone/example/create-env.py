from real2sim2real.envs.config import SimulatorConfig
from real2sim2real.envs.env import BaseEnv



file_path = "configs/example/pick_into_basket/collect_data_render_1_16_one_item.yaml"
sim_config = SimulatorConfig(file_path)

headless = True
webrtc = False

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

i = 0
env.reset()
while env.simulation_app.is_running():
    i += 1
    env._runner._world.step(render=True)
env.simulation_app.close()
