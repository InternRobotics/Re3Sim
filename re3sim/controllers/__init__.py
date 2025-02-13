from .multi_item_controller import MultiItemController
import numpy as np


def get_action(controller, obs, graspnet=False):
    try:
        render_obs = obs["info"]["render"]
        current_time = obs["current_world_time"]
        tmp_obs = obs["observations"]
        controller: MultiItemController
        controller._item_idx = obs["info"].get("picking_item", 0)
        if graspnet:
            extra_params = {
                "rgb": (
                    render_obs["images"]["wrist_camera"]
                    if "wrist_camera" in render_obs["images"]
                    else render_obs["images"]["camera_0"]
                ),
                "depth": (
                    render_obs["depths"]["wrist_camera"]
                    if "wrist_camera" in render_obs["depths"]
                    else render_obs["depths"]["camera_0"]
                ),
                "info": (
                    render_obs["root_info"]["wrist_camera"]
                    if "wrist_camera" in render_obs["root_info"]
                    else render_obs["root_info"]["camera_0"]
                ),
            }
        else:
            extra_params = {}
        replan = True
        replan_count = 0
        while replan:
            raw_action, info = controller.forward(
                picking_position=tmp_obs["cube"]["position"],
                picking_orientation=tmp_obs["cube"]["orientation"],
                placing_position=tmp_obs["target"]["position"],
                placing_orientation=tmp_obs["target"]["orientation"],
                current_joint_positions=tmp_obs["robot"]["qpos"],
                gripper_position=tmp_obs["robot"]["gripper"]["position"],
                gripper_orientation=tmp_obs["robot"]["gripper"]["orientation"],
                end_effector_offset=np.array([0, 0.005, 0]),
                current_time=current_time,
                **extra_params
            )
            if info is None or not info.get("replan", False):
                break
            replan = info["replan"]
            replan_count += 1
            if replan_count > 20:
                raw_action = -1
                info = None
                break
        if isinstance(raw_action, int) and raw_action == -1:
            positions = tmp_obs["robot"]["qpos"]
            controller_done = True
        else:
            positions = raw_action
            controller_done = False

        action = {"robot": positions}
        if obs["done"] or controller_done:
            print("controller done")
            controller.reset()
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(e)
        return (e, tb)
    return (action, controller_done)


def get_action_for_multi_item_continuous(controller, obs, graspnet=False):
    try:
        render_obs = obs["info"]["render"]
        current_time = obs["current_world_time"]
        tmp_obs = obs["observations"]
        controller: MultiItemController
        controller._item_idx = obs["info"].get("picking_item", 0)
        if graspnet:
            extra_params = {
                "rgb": (
                    render_obs["images"]["wrist_camera"]
                    if "wrist_camera" in render_obs["images"]
                    else render_obs["images"]["camera_0"]
                ),
                "depth": (
                    render_obs["depths"]["wrist_camera"]
                    if "wrist_camera" in render_obs["depths"]
                    else render_obs["depths"]["camera_0"]
                ),
                "info": (
                    render_obs["root_info"]["wrist_camera"]
                    if "wrist_camera" in render_obs["root_info"]
                    else render_obs["root_info"]["camera_0"]
                ),
            }
        else:
            extra_params = {}
        replan = True
        replan_count = 0
        while replan:
            raw_action, info = controller.forward(
                picking_position=tmp_obs["cube"]["position"],
                picking_orientation=tmp_obs["cube"]["orientation"],
                placing_position=tmp_obs["target"]["position"],
                placing_orientation=tmp_obs["target"]["orientation"],
                current_joint_positions=tmp_obs["robot"]["qpos"],
                gripper_position=tmp_obs["robot"]["gripper"]["position"],
                gripper_orientation=tmp_obs["robot"]["gripper"]["orientation"],
                end_effector_offset=np.array([0, 0.005, 0]),
                current_time=current_time,
                **extra_params
            )
            if not info:
                break
            replan = info.get("replan", False)
            replan_count += 1
            if not replan:
                break
            if replan_count > 20:
                raw_action = -1
                break
        if isinstance(raw_action, int) and raw_action == -1:
            positions = tmp_obs["robot"]["qpos"]
            controller_done = True
        else:
            positions = raw_action
            controller_done = controller.is_done()

        action = {"robot": positions}
        if obs["done"] or controller_done:
            print("controller done")
            controller.reset()
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(e)
        return (e, tb, {})
    return (action, controller_done, info)


try:
    from real2sim2real.controllers.stacking_curobo import StackingCuroboController

    def get_action_for_curobo(controller: StackingCuroboController, obs):
        action, controller_done = controller.forward(obs)
        return {"robot": action}, controller_done, {}

except:
    import traceback

    tb = traceback.format_exc()
    print(tb)


def get_action_for_place_on_the_box(controller, obs, graspnet=False):
    try:
        render_obs = obs["info"]["render"]
        current_time = obs["current_world_time"]
        tmp_obs = obs["observations"]
        controller: MultiItemController
        controller._item_idx = obs["info"].get("picking_item", 0)
        if graspnet:
            extra_params = {
                "rgb": (
                    render_obs["images"]["wrist_camera"]
                    if "wrist_camera" in render_obs["images"]
                    else render_obs["images"]["camera_0"]
                ),
                "depth": (
                    render_obs["depths"]["wrist_camera"]
                    if "wrist_camera" in render_obs["depths"]
                    else render_obs["depths"]["camera_0"]
                ),
                "info": (
                    render_obs["root_info"]["wrist_camera"]
                    if "wrist_camera" in render_obs["root_info"]
                    else render_obs["root_info"]["camera_0"]
                ),
            }
        else:
            extra_params = {}
        replan = True
        replan_count = 0
        while replan:
            raw_action, info = controller.forward(
                picking_position=tmp_obs["cube"]["position"],
                picking_orientation=tmp_obs["cube"]["orientation"],
                placing_position=tmp_obs["target"]["position"],
                placing_orientation=tmp_obs["target"]["orientation"],
                current_joint_positions=tmp_obs["robot"]["qpos"],
                gripper_position=tmp_obs["robot"]["gripper"]["position"],
                gripper_orientation=tmp_obs["robot"]["gripper"]["orientation"],
                end_effector_offset=np.array([0, 0.005, 0]),
                current_time=current_time,
                **extra_params
            )
            if not info:
                break
            replan = info.get("replan", False)
            replan_count += 1
            if not replan:
                break
            if replan_count > 20:
                raw_action = -1
                break
        if isinstance(raw_action, int) and raw_action == -1:
            positions = tmp_obs["robot"]["qpos"]
            controller_done = True
        else:
            positions = raw_action
            controller_done = controller.is_done()
        action = {"robot": positions}
        if obs["done"] or controller_done:
            print("controller done")
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(e)
        return (e, tb, {})
    return (action, controller_done, info)
