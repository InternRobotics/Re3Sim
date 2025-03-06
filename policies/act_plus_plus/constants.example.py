import pathlib
import os

try:
    from real2sim2real.act_plus_plus.tasks.real_task import TASK_CONFIGS as REAL_TASK_CONFIGS
except:
    from tasks.real_task import TASK_CONFIGS as REAL_TASK_CONFIGS

### Task parameters
DATA_DIR = "/path/to/data"
SIM_TASK_CONFIGS = {
    # pick_one_from_multi_1128_old_align
    "sim_pick_one_from_multi_1128_old_align_huge": {
        "dataset_dir": DATA_DIR + "/pick_one_from_multi_1128_old_align_3/huge",
        "num_episodes": 7504,
        "episode_len": -1,
        "camera_names": ["wrist_camera", "camera_0"],
    },
    "sim_pick_one_from_multi_1128_old_align_large": {
        "dataset_dir": DATA_DIR + "/pick_one_from_multi_1128_old_align_3/large",
        "num_episodes": 7504,
        "episode_len": -1,
        "camera_names": ["wrist_camera", "camera_0"],
    },
    "sim_pick_one_from_multi_1128_old_align": {
        "dataset_dir": DATA_DIR + "/pick_one_from_multi_1128_old_align_3/mid",
        "num_episodes": 1555,
        "episode_len": -1,
        "camera_names": ["wrist_camera", "camera_0"],
    },
    # low res
    "sim_random_position_1027_1_low_res_wo13": {
        "dataset_dir": DATA_DIR + "/random_position_1027_1_low_res/mid",
        "num_episodes": 1166,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    # polycam
    "sim_random_position_1109_1_little_orientation_mid_wo13_polycam": {
        "dataset_dir": DATA_DIR + "/random_position_1109_1_polycam/mid",
        "num_episodes": 759,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1027_1_polycam",
    },
    # place
    "sim_more_item_place_huge": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_place/huge",
        "num_episodes": 1840,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_place",
    },
    "sim_more_item_place_large": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_place/large",
        "num_episodes": 1537,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_place",
    },
    "sim_more_item_place_mid": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_place/mid",
        "num_episodes": 323,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_place",
    },
    # more-item
    "sim_more_item_filter_angle_all_good_data": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item/all_good_data",
        "num_episodes": 3097,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_more_item_filter_angle": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item/filter_angle",
        "num_episodes": 1498,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item",
    },
    "sim_more_item_huge2": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item/huge2",
        "num_episodes": 4243,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item",
    },
    "sim_more_item_huge": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item/huge",
        "num_episodes": 1840,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item",
    },
    "sim_more_item_large": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item/large",
        "num_episodes": 323,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item",
    },
    "sim_more_item_mid": {
        "dataset_dir": DATA_DIR
        + "/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1113_1_eggplant_low_res_continuous_better_controller2_multi_item",
    },
    # multi-item-low_res-continuous-iter_2
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_100": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/100",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_200": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/200",
        "num_episodes": 200,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_400": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/400",
        "num_episodes": 400,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_800": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/800",
        "num_episodes": 800,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_1200": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/1200",
        "num_episodes": 1200,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_1600": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/1600",
        "num_episodes": 1600,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_all": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/all",
        "num_episodes": 1953,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_iter_2_large": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2/large",
        "num_episodes": 1144,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1021_1_eggplant_low_res_continuous_better_controller_iter_2",
    },
    # multi-item-low_res-continuous-better-controller
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_huge": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller/huge",
        "num_episodes": 1944,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1021_1_eggplant_low_res_continuous_better_controller",
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_large": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller/large",
        "num_episodes": 939,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1021_1_eggplant_low_res_continuous_better_controller",
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_better_controller_mid": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous_better_controller/mid",
        "num_episodes": 321,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1021_1_eggplant_low_res_continuous_better_controller",
    },
    # multi-item-low_res-continuous
    "sim_multi_item_1021_1_eggplant_low_res_continuous_large": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous/large",
        "num_episodes": 2143,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1021_1_eggplant_low_res_continuous",
    },
    "sim_multi_item_1021_1_eggplant_low_res_continuous_mid": {
        "dataset_dir": DATA_DIR
        + "/random_position_1021_1_eggplant_low_res_continuous/mid",
        "num_episodes": 463,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1021_1_eggplant_low_res_continuous",
    },
    # muti-item-low_res
    "sim_multi_item_1021_1_eggplant_low_res": {
        "dataset_dir": DATA_DIR + "/random_position_1021_1_eggplant_low_res/mid",
        "num_episodes": 1853,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1021_1_eggplant_low_res/clean_data",
    },
    # multi-item
    "sim_multi_item_1021_1_eggplant": {
        "dataset_dir": DATA_DIR + "/random_position_1021_1_eggplant/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_random_position_1027_1_little_orientation_h264_0.3m_mid_wo03": {
        "dataset_dir": DATA_DIR
        + "/random_position_1027_1_little_orientation_h264_0.3m/mid",
        "num_episodes": 759,
        "episode_len": -1,
        "camera_names": ["camera_1", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1027_1_little_orientation_h264_0.3m",
    },
    "sim_random_position_1027_1_little_orientation_h264_0.3m_mid_wo13": {
        "dataset_dir": DATA_DIR
        + "/random_position_1027_1_little_orientation_h264_0.3m/mid",
        "num_episodes": 759,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
        "original_dataset_root_path": "/path/to/isaacsim-data/random_position_1027_1_little_orientation_h264_0.3m",
    },
    # random_position_1027_1_little_orientation
    "sim_random_position_1027_1_little_orientation_mid_wo3": {
        "dataset_dir": DATA_DIR + "/random_position_1027_1_little_orientation/mid",
        "num_episodes": 759,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "wrist_camera"],
    },
    "sim_random_position_1027_1_little_orientation_mid_wo03": {
        "dataset_dir": DATA_DIR + "/random_position_1027_1_little_orientation/mid",
        "num_episodes": 759,
        "episode_len": -1,
        "camera_names": ["camera_1", "wrist_camera"],
    },
    "sim_random_position_1027_1_little_orientation_mid_wo13": {
        "dataset_dir": DATA_DIR + "/random_position_1027_1_little_orientation/mid",
        "num_episodes": 759,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_random_position_1027_1_little_orientation_mid_wo013": {
        "dataset_dir": DATA_DIR + "/random_position_1027_1_little_orientation/mid",
        "num_episodes": 759,
        "episode_len": -1,
        "camera_names": ["wrist_camera"],
    },
    # random_position_1027_1_random_orientation
    "sim_random_position_1027_1_random_orientation_mid_wo3": {
        "dataset_dir": DATA_DIR + "/random_position_1027_1_random_orientation/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "wrist_camera"],
    },
    "sim_random_position_1024_1": {
        "dataset_dir": DATA_DIR + "/random_position_1024_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "camera_3", "wrist_camera"],
    },
    # multi-task-corn-1
    "sim_random_position_1023_1_corn": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1023_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1023_1_corn_wo01": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1023_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_3", "wrist_camera"],
    },
    "sim_random_position_1023_1_corn_wo03": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1023_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_1", "wrist_camera"],
    },
    "sim_random_position_1023_1_corn_wo13": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1023_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_random_position_1023_1_corn_wo0": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1023_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1023_1_corn_wo1": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1023_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1023_1_corn_wo3": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1023_1_corn/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "wrist_camera"],
    },
    # 1019_1_large
    "sim_random_position_1019_1_large": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1019_1_large_wo01": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_3", "wrist_camera"],
    },
    "sim_random_position_1019_1_large_wo03": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_random_position_1019_1_large_wo13": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_random_position_1019_1_large_wo0": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1019_1_large_wo1": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1019_1_large_wo3": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "wrist_camera"],
    },
    # 1019_1
    "sim_random_position_1019_1_mid_wo01": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_3", "wrist_camera"],
    },
    "sim_random_position_1019_1_mid_wo03": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_1", "wrist_camera"],
    },
    "sim_random_position_1019_1_mid_wo13": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_random_position_1019_1_mid_wo0": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1019_1_mid_wo1": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1019_1_mid_wo3": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "wrist_camera"],
    },
    "sim_random_position_1019_1_mid": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1019_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "camera_3", "wrist_camera"],
    },
    # 1017_1
    "sim_random_position_1017_1_large": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/large",
        "num_episodes": 500,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1017_1_mid_wo13": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "sim_random_position_1017_1_mid_wo03": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_1", "wrist_camera"],
    },
    "sim_random_position_1017_1_mid_wo01": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_3", "wrist_camera"],
    },
    "sim_random_position_1017_1_mid_wo0": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1017_1_mid_wo1": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1017_1_mid_wo3": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "wrist_camera"],
    },
    "sim_random_position_1017_1_mid": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1017_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "camera_3", "wrist_camera"],
    },
    "sim_random_position_1010_1_large": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1010_1/large",
        "num_episodes": 1628,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_position_1010_1_mid": {
        "dataset_dir": DATA_DIR + "/sim_random_position_1010_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_105_debug": {
        "dataset_dir": DATA_DIR + "/sim_random_position_105_1/debug",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_105_debug_ns": {
        "dataset_dir": DATA_DIR + "/sim_random_position_105_1/debug-new-storage",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_position_105_1_large": {
        "dataset_dir": DATA_DIR + "/sim_random_position_105_1/large",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_position_105_1_mid": {
        "dataset_dir": DATA_DIR + "/sim_random_position_105_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_position_102_1_large": {
        "dataset_dir": DATA_DIR + "/sim_random_position_102_1/large",
        "num_episodes": 1271,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_position_102_1_mid": {
        "dataset_dir": DATA_DIR + "/sim_random_position_102_1/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_position_101_3_mid": {
        "dataset_dir": DATA_DIR + "/sim_random_position_101_3_mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_robot_base_930_large": {
        "dataset_dir": DATA_DIR + "/sim_random_robot_base_930_large",
        "num_episodes": 1049,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_robot_base_930_mid": {
        "dataset_dir": DATA_DIR + "/sim_random_robot_base_930/mid",
        "num_episodes": 1049,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_robot_base_926_large": {
        "dataset_dir": DATA_DIR + "/sim_random_robot_base_926_large",
        "num_episodes": 1049,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_robot_base_926": {
        "dataset_dir": DATA_DIR + "/sim_random_robot_base_926",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["right_camera", "wrist_camera"],
    },
    "sim_random_position_cloth_table_large": {
        "dataset_dir": DATA_DIR + "/new_table_with_cloth/large",
        "num_episodes": 1000,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_random_position_cloth_table_mid": {
        "dataset_dir": DATA_DIR + "/new_table_with_cloth/mid",
        "num_episodes": 113,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_random_position_oracle_mid": {
        "dataset_dir": DATA_DIR + "/new_planner/oracle/mid",
        "num_episodes": 100,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_random_position_oracle_large": {
        "dataset_dir": DATA_DIR + "/new_planner/oracle/large",
        "num_episodes": 999,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_random_position_new_planner_abs_large": {
        "dataset_dir": DATA_DIR + "/new_planner/simple_green_table_large",
        "num_episodes": 873,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_random_position_new_planner_abs": {
        "dataset_dir": DATA_DIR + "/new_planner/simple_green_table_large",
        "num_episodes": 132,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_random_position_abs": {
        "dataset_dir": DATA_DIR + "/old_planner/sim_random_position_abs",
        "num_episodes": 77,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_random_position_rel": {
        "dataset_dir": DATA_DIR + "/old_planner/sim_random_position_rel",
        "num_episodes": 77,
        "episode_len": -1,
        "camera_names": ["left_camera", "right_camera", "wrist_camera"],
    },
    "sim_transfer_cube_scripted": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_transfer_cube_human": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_human",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
    },
    "sim_insertion_scripted": {
        "dataset_dir": DATA_DIR + "/sim_insertion_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_insertion_human": {
        "dataset_dir": DATA_DIR + "/sim_insertion_human",
        "num_episodes": 50,
        "episode_len": 500,
        "camera_names": ["top"],
    },
    "all": {
        "dataset_dir": DATA_DIR + "/",
        "num_episodes": None,
        "episode_len": None,
        "name_filter": lambda n: "sim" not in n,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
    "sim_transfer_cube_scripted_mirror": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted_mirror",
        "num_episodes": None,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_insertion_scripted_mirror": {
        "dataset_dir": DATA_DIR + "/sim_insertion_scripted_mirror",
        "num_episodes": None,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
}
SIM_TASK_CONFIGS.update(REAL_TASK_CONFIGS)
### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
]

XML_DIR = (
    str(pathlib.Path(__file__).parent.resolve()) + "/assets/"
)  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    + MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    + PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
)

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
    MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
    PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
    MASTER_GRIPPER_JOINT_NORMALIZE_FN(x)
)

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)

MASTER_POS2JOINT = (
    lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE)
    / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
)
PUPPET_POS2JOINT = (
    lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE)
    / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
)

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
