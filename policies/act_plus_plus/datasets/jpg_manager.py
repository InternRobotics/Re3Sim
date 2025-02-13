try:
    import av
except:
    print("av not installed")
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
    """单例模式的元类，支持跨进程共享"""

    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        # 使用folder_path作为key
        key = args[0] if args else kwargs.get("folder_path", None)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[key] = instance
                    # 只在第一次创建实例时打印日志
                    instance._log_initialized = True
        return cls._instances[key]


class JpgToShm:
    """将视频文件拷贝到共享内存的工具类"""

    def __init__(
        self, folder_path: str, shm_path: str = "", extensions: Set[str] = {".jpg"}
    ):
        if not shm_path:
            shm_path = "/dev/shm/video_data" + folder_path.replace("/", "_")
        self.shm_path = Path(shm_path)
        self.extensions = extensions
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 创建共享内存目录
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
        将指定文件夹中的视频文件并行拷贝到共享内存
        """
        video_info = {}

        def _copy_file(src_path, dst_path):
            """使用大缓冲区复制单个文件"""
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
                self.logger.info(f"已复制: {src_path.name}")
            except Exception as e:
                self.logger.error(f"复制 {src_path} 失败: {str(e)}")
                raise

        try:
            # 收集所有需要复制的文件
            copy_tasks = []
            for ext in self.extensions:
                for video_path in self.folder_path.rglob(f"*{ext}"):
                    rel_path = str(video_path.relative_to(self.folder_path))
                    shm_file = self.shm_path / f"{rel_path}"
                    self._video_paths.append(shm_file)
                    copy_tasks.append((video_path, shm_file))

            total_files = len(copy_tasks)
            self.logger.info(f"共发现 {total_files} 个文件需要处理")

            # 使用线程池并行复制
            with ThreadPoolExecutor(
                max_workers=min(os.cpu_count() * 2, 16)
            ) as executor:
                futures = [
                    executor.submit(_copy_file, src, dst) for src, dst in copy_tasks
                ]

                # 等待所有复制任务完成
                for future in futures:
                    future.result()  # 这会抛出任何复制过程中的异常

            self.logger.info(f"已完成 {total_files} 个文件的处理")

        except Exception as e:
            self.logger.error(f"拷贝文件失败: {str(e)}")
            raise

    def clear_shm(self) -> None:
        """清理共享内存"""
        try:
            if self.shm_path.exists():
                shutil.rmtree(self.shm_path)
            self.logger.info("已清理共享内存")
        except Exception as e:
            self.logger.error(f"清理共享内存失败: {str(e)}")
            raise


class ShmVideoReader:
    """
    基于共享内存的视频帧读取类
    从shm中读取视频数据，支持多进程共享
    """

    def __init__(
        self,
        shm_path: str = "/dev/shm/video_data",  # 共享内存基础路径
        cache_size: int = 20,  # 帧缓存大小
        gpu_id: int = 0,  # GPU设备ID
        enable_gpu: bool = True,  # 是否启用GPU解码
        prefetch_size: int = 0,  # 预加载帧数
        num_workers: int = 6,  # 解码线程数
    ):
        self.shm_path = Path(shm_path)
        self.cache_size = cache_size
        self.gpu_id = gpu_id
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.prefetch_size = prefetch_size

        # 初始化存储结构
        self.frame_cache = OrderedDict()
        self.prefetch_queues: Dict[str, queue.Queue] = {}

        # 线程池初始化
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # 锁初始化
        self.cache_lock = Lock()
        self.video_locks = {}

        # 日志配置
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 加载视频元信息
        self._load_metadata()

    def _load_metadata(self) -> None:
        """加载视频元信息"""
        try:
            metadata_path = self.shm_path / "metadata.json"
            if not metadata_path.exists():
                self.video_info = {}
                return

            with open(metadata_path, "r") as f:
                self.video_info = json.load(f)

            # 为每个视频创建锁
            for video_path in self.video_info:
                self.video_locks[video_path] = Lock()

        except Exception as e:
            self.logger.error(f"加载元信息失败: {str(e)}")
            self.video_info = {}

    def _get_video_mmap(self, video_path: str) -> mmap.mmap:
        """获取视频数据的内存映射"""
        video_shm_path = self.shm_path / f"{video_path}.bin"
        if not video_shm_path.exists():
            raise FileNotFoundError(f"共享内存中不存在视频文件: {video_path}")

        fd = os.open(str(video_shm_path), os.O_RDONLY)
        return mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)

    def _get_video_container(self, video_path: str):
        """获取视频容器对象"""
        # 从共享内存读取视频数据
        mm = self._get_video_mmap(video_path)
        video_bytes = io.BytesIO(mm.read())
        mm.close()

        # 创建容器
        container = av.open(video_bytes)
        stream = container.streams.video[0]

        # 配置解码器
        stream.thread_type = "AUTO"
        if self.enable_gpu:
            stream.codec_context.options = {
                "hwaccel": "cuda",
                "gpu_id": str(self.gpu_id),
            }

        return container

    def _decode_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """解码指定帧"""
        with self.video_locks[video_path]:
            container = self._get_video_container(video_path)
            stream = container.streams.video[0]

            # 计算目标时间戳
            target_ts = int(frame_idx * stream.duration / stream.frames)
            container.seek(target_ts, stream=stream)

            # 解码目标帧
            frame_count = 0
            for frame in container.decode(video=0):
                if frame_count == 0:
                    if self.enable_gpu:
                        with torch.cuda.device(self.gpu_id):
                            frame_tensor = torch.from_numpy(
                                frame.to_ndarray(format="rgb24")
                            ).cuda()
                            result = frame_tensor.cpu().numpy()
                            # 显式清理GPU内存
                            del frame_tensor
                            return result
                    else:
                        return frame.to_ndarray(format="rgb24")
                frame_count += 1

            raise ValueError(f"无法找到帧 {frame_idx}")

    def _prefetch_frames(self, video_path: str, start_frame: int) -> None:
        """预加载后续帧"""
        try:
            for i in range(start_frame, start_frame + self.prefetch_size):
                if i >= self.video_info[video_path]["frame_count"]:
                    break

                cache_key = (video_path, i)
                if cache_key in self.frame_cache:
                    continue

                frame = self._decode_frame(video_path, i)

                queue_key = (video_path, i)
                if queue_key not in self.prefetch_queues:
                    self.prefetch_queues[queue_key] = queue.Queue(maxsize=1)
                self.prefetch_queues[queue_key].put(frame)

        except Exception as e:
            self.logger.error(f"预加载失败: {str(e)}")

    def get_frame(
        self,
        video_path: str,
        frame_idx: int,
        max_retries: int = 100,
        retry_delay: float = 0.1,
    ) -> np.ndarray:
        """获取指定帧"""
        video_path = str(video_path)
        last_exception = None
        for attempt in range(max_retries):
            try:
                if video_path not in self.video_info:
                    raise ValueError(f"视频未加载: {video_path}")

                if frame_idx >= self.video_info[video_path]["frame_count"]:
                    raise IndexError(f"帧序号越界: {frame_idx}")

                # 检查缓存
                cache_key = (video_path, frame_idx)
                with self.cache_lock:
                    if cache_key in self.frame_cache:
                        self.frame_cache.move_to_end(cache_key)
                        return self.frame_cache[cache_key]

                # 检查预加载队列
                queue_key = (video_path, frame_idx)
                if queue_key in self.prefetch_queues:
                    try:
                        frame = self.prefetch_queues[queue_key].get_nowait()
                        self.prefetch_queues.pop(queue_key)

                        with self.cache_lock:
                            if len(self.frame_cache) >= self.cache_size:
                                self.frame_cache.popitem(last=False)
                            self.frame_cache[cache_key] = frame

                        self.executor.submit(
                            self._prefetch_frames, video_path, frame_idx + 1
                        )

                        return frame

                    except queue.Empty:
                        pass

                # 直接解码
                frame = self._decode_frame(video_path, frame_idx)

                with self.cache_lock:
                    if len(self.frame_cache) >= self.cache_size:
                        self.frame_cache.popitem(last=False)
                    self.frame_cache[cache_key] = frame

                self.executor.submit(self._prefetch_frames, video_path, frame_idx + 1)
            except Exception as e:
                last_exception = e
                import traceback

                traceback.print_exc()
                self.logger.warning(
                    f"加载{video_path}时获取帧失败,正在重试 ({attempt + 1}/{max_retries}): {str(e)}"
                )
                time.sleep(retry_delay)
                continue

            return frame

        self.logger.error(
            f"加载{video_path}时获取帧失败,重试{max_retries}次后仍然失败: {str(last_exception)}"
        )
        raise last_exception

    def get_video_info(self, video_path: str) -> Dict:
        """获取视频信息"""
        if video_path not in self.video_info:
            raise ValueError(f"视频未加载: {video_path}")
        return self.video_info[video_path].copy()

    def get_loaded_videos(self) -> List[str]:
        """获取已加载的视频列表"""
        return list(self.video_info.keys())

    def release(self) -> None:
        """释放资源"""
        self.executor.shutdown(wait=False)
        self.frame_cache.clear()
        self.prefetch_queues.clear()
        self.logger.info("已释放所有资源")

    def __del__(self):
        """析构函数"""
        self.release()
