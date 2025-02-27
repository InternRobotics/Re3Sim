import numpy as np
import torch
import os
from pathlib import Path
from typing import Dict, Optional, Set, List, Union, Tuple
from collections import OrderedDict
import io
import logging
import psutil
import queue
from torch.multiprocessing import Lock
from concurrent.futures import ThreadPoolExecutor
import heapq
import json
import shutil
import mmap
import time


class SingletonMeta(type):
    """Singleton meta class, supporting cross-process sharing"""

    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        # Use folder_path as key    
        key = args[0] if args else kwargs.get("folder_path", None)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[key] = instance
                    # Print log only when the instance is first created
                    instance._log_initialized = True
        return cls._instances[key]


class JpgToShm:
    """Utility class to copy video files to shared memory"""

    def __init__(
        self, folder_path: str, shm_path: str = "", extensions: Set[str] = {".jpg"}
    ):
        if not shm_path:
            shm_path = "/dev/shm/video_data" + folder_path.replace("/", "_")
        self.shm_path = Path(shm_path)
        self.extensions = extensions
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create shared memory directory
        self.shm_path.mkdir(parents=True, exist_ok=True)
        self.folder_path = Path(folder_path)
        self._video_paths = []
        self.video_paths_to_shm_path = {}

    def get_shm_path(self, video_path: str) -> str:
        rel_path = str(
            Path(os.path.realpath(video_path)).relative_to(
                os.path.realpath(self.folder_path)
            )
        )
        return self.shm_path / f"{rel_path}"

    # def get_meta_data(self):
    #     for ext in self.extensions:
    #         for video_path in self.folder_path.rglob(f"*{ext}"):
    #             rel_path = str(video_path.relative_to(self.folder_path))
    #             shm_file = self.shm_path / f"{rel_path}"
    #             self._video_paths.append(shm_file)
    #             self.video_paths_to_shm_path[os.path.realpath(video_path)] = shm_file

    def copy_videos(self) -> None:
        """
        Copy video files in the specified folder to shared memory in parallel
        """
        video_info = {}

        def _copy_file(src_path, dst_path):
            """Copy a single file using a large buffer"""
            try:
                if dst_path.exists():
                    src_size = src_path.stat().st_size
                    dst_size = dst_path.stat().st_size
                    if src_size == dst_size:
                        return

                dst_path.parent.mkdir(parents=True, exist_ok=True)
                buffer_size = 1024 * 1024  # 1MB buffer
                with open(src_path, "rb") as fsrc, open(dst_path, "wb") as fdst:
                    while True:
                        buf = fsrc.read(buffer_size)
                        if not buf:
                            break
                        fdst.write(buf)
                self.logger.info(f"already copied: {src_path.name}")
            except Exception as e:
                self.logger.error(f"copy {src_path} failed: {str(e)}")
                raise

        try:
            # Collect all files to be copied
            copy_tasks = []
            for ext in self.extensions:
                for video_path in self.folder_path.rglob(f"*{ext}"):
                    rel_path = str(video_path.relative_to(self.folder_path))
                    shm_file = self.shm_path / f"{rel_path}"
                    self._video_paths.append(shm_file)
                    copy_tasks.append((video_path, shm_file))

            total_files = len(copy_tasks)
            self.logger.info(f"Found {total_files} files to process")

            # Use thread pool to copy files in parallel
            with ThreadPoolExecutor(
                max_workers=min(os.cpu_count() * 2, 16)
            ) as executor:
                futures = [
                    executor.submit(_copy_file, src, dst) for src, dst in copy_tasks
                ]

                # Wait for all copy tasks to complete
                for future in futures:
                    future.result()  # This will raise any exceptions during the copy process

            self.logger.info(f"completed {total_files} files")

        except Exception as e:
            self.logger.error(f"copy files failed: {str(e)}")
            raise

    def clear_shm(self) -> None:
        """Clean shared memory"""
        try:
            if self.shm_path.exists():
                shutil.rmtree(self.shm_path)
            self.logger.info("clean jpg manager: clean shared memory")
        except Exception as e:
            self.logger.error(f"clean jpg manager: clean shared memory failed: {str(e)}")
            raise
