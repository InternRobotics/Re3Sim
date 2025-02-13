DATA_DIR = "/path/to/real_data"
TASK_CONFIGS = {
    "real_random_position_1024_1": {
        "dataset_dir": DATA_DIR + "/real_random_position_1024_1",
        "num_episodes": 50,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "camera_3", "wrist_camera"],
    },
    "real_random_position_1024_1_wo01": {
        "dataset_dir": DATA_DIR + "/real_random_position_1024_1",
        "num_episodes": 50,
        "episode_len": -1,
        "camera_names": ["camera_3", "wrist_camera"],
    },
    "real_random_position_1024_1_wo03": {
        "dataset_dir": DATA_DIR + "/real_random_position_1024_1",
        "num_episodes": 50,
        "episode_len": -1,
        "camera_names": ["camera_1", "wrist_camera"],
    },
    "real_random_position_1024_1_wo13": {
        "dataset_dir": DATA_DIR + "/real_random_position_1024_1",
        "num_episodes": 50,
        "episode_len": -1,
        "camera_names": ["camera_0", "wrist_camera"],
    },
    "real_random_position_1024_1_wo0": {
        "dataset_dir": DATA_DIR + "/real_random_position_1024_1",
        "num_episodes": 50,
        "episode_len": -1,
        "camera_names": ["camera_1", "camera_3", "wrist_camera"],
    },
    "real_random_position_1024_1_wo1": {
        "dataset_dir": DATA_DIR + "/real_random_position_1024_1",
        "num_episodes": 50,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_3", "wrist_camera"],
    },
    "real_random_position_1024_1_wo3": {
        "dataset_dir": DATA_DIR + "/real_random_position_1024_1",
        "num_episodes": 50,
        "episode_len": -1,
        "camera_names": ["camera_0", "camera_1", "wrist_camera"],
    },
}
