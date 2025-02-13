from loguru import logger
from pathlib import Path
import os
import time
import re
import random
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Any


def get_log_folder(log_root: str):
    log_folder = Path(log_root) / (
        time.strftime("%Y-%m-%d_%H-%M-%S_")
        + f"{int(time.time() * 1000000) % 1000000:06d}"
    )
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def get_sub_log_path(log_folder: Path):
    log_folder = Path(log_folder)
    files = os.listdir(log_folder)
    pattern = r"log-(\d{6})-\d{4}"
    existing_numbers = [
        int(re.match(pattern, file).group(1))
        for file in files
        if re.match(pattern, file)
    ]
    if not existing_numbers:
        next_number = 1
    else:
        existing_numbers.sort()
        next_number = existing_numbers[-1] + 1
    random_id = random.randint(1000, 9999)
    dir_path = log_folder / f"log-{next_number:06d}-{random_id}"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


class BaseLogger(ABC):
    def __init__(
        self,
        log_root_path: str,
        disk_space_threshold: int = 5,
        save_depth: bool = False,
        save_depth_as_png: bool = False,
    ):
        """
        Args:
            log_root_path: The root path of the log folder
            disk_space_threshold: The threshold of disk space (GB)
        """
        self.log_root_path = get_log_folder(Path(log_root_path))
        self.log_root_path.mkdir(parents=True, exist_ok=True)
        logger.add(self.log_root_path / "info.log", level="INFO")
        logger.add(self.log_root_path / "debug.log", level="DEBUG")
        logger.add(self.log_root_path / "error.log", level="ERROR")
        self.disk_space_threshold = disk_space_threshold
        self.scalar_data_logger: Dict[str, List[Any]] = {}
        self.image_data_logger_jpg: Dict[str, List[Any]] = {}
        self.image_data_logger_png: Dict[str, List[Any]] = {}
        self.depth_data_logger: Dict[str, List[Any]] = {}
        self.json_data_logger: Dict[str, Any] = {}
        self.save_depth = save_depth
        self.save_depth_as_png = save_depth_as_png

    def check_disk_space(self):
        total, used, free = shutil.disk_usage(self.log_root_path)
        free_gb = free / (2**30)  # 转换为GB
        logger.warning(f"Disk space check: {free_gb:.2f} GB left")
        return free_gb >= self.disk_space_threshold

    def print(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def info(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        logger.debug(*args, **kwargs)

    def error(self, *args, **kwargs):
        logger.error(*args, **kwargs)

    def warning(self, *args, **kwargs):
        logger.warning(*args, **kwargs)

    def critical(self, *args, **kwargs):
        logger.critical(*args, **kwargs)

    def add_scalar_data(self, key, value):
        if key not in self.scalar_data_logger:
            self.scalar_data_logger[key] = []
        self.scalar_data_logger[key].append(value)

    def add_image_data_jpg(self, key, image):
        if key not in self.image_data_logger_jpg:
            self.image_data_logger_jpg[key] = []
        self.image_data_logger_jpg[key].append(image)

    def add_image_data_png(self, key, image):
        if key not in self.image_data_logger_png:
            self.image_data_logger_png[key] = []
        self.image_data_logger_png[key].append(image)

    def add_depth_data(self, key, depth):
        if key not in self.depth_data_logger:
            self.depth_data_logger[key] = []
        self.depth_data_logger[key].append(depth)

    def add_json_data(self, key, data):
        self.json_data_logger[key] = data

    def clear(self):
        self.scalar_data_logger = {}
        self.image_data_logger_jpg = {}
        self.image_data_logger_png = {}
        self.depth_data_logger = {}
        self.json_data_logger = {}

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def save(self):
        pass
