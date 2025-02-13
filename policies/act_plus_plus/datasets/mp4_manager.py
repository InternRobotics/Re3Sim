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


class FastVideoReader(metaclass=SingletonMeta):
    """
    高效的视频帧读取类，支持多DataLoader共享
    使用单例模式确保相同folder_path只创建一个实例
    """

    def __init__(
        self,
        folder_path: str,
        cache_size: int = 20,  # 缓存帧数
        max_memory_ratio: float = 0.9,  # 最大内存使用比例
        video_extensions: Set[str] = {".mp4", ".MP4", ".avi", ".mov"},
        gpu_id: int = 0,  # GPU设备ID
        enable_gpu: bool = True,  # 是否启用GPU解码
        prefetch_size: int = 4,  # 预加载帧数
        num_workers: int = 6,  # 解码线程数
        auto_load: bool = True,  # 是否自动加载视频
    ):
        # 避免重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self._log_initialized = False  # 用于控制日志只打印一次
        self.folder_path = Path(os.path.realpath(folder_path))
        self.cache_size = cache_size
        self.max_memory_ratio = max_memory_ratio
        self.video_extensions = video_extensions
        self.gpu_id = gpu_id
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.prefetch_size = prefetch_size

        # 存储结构初始化
        self._init_storage()

        # 线程池初始化
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # 日志配置
        self._setup_logging()

        # 延迟初始化锁
        self._locks_initialized = False

        self.auto_load = auto_load

    def _init_storage(self):
        """初始化存储结构"""
        self.video_data: Dict[str, bytes] = {}  # 存储视频文件的字节数据
        self.video_info: Dict[str, Dict] = {}  # 存储视频元信息
        self.video_priorities: Dict[str, float] = {}  # 存储视频优先级
        self.frame_cache = OrderedDict()  # LRU帧缓存
        self.prefetch_queues: Dict[str, queue.Queue] = {}  # 预加载队列

    def _init_locks(self):
        """延迟初始化锁"""
        if not self._locks_initialized:
            self.locks = {}
            self.cache_lock = Lock()
            self.load_lock = Lock()
            self._locks_initialized = True
            # 自动加载视频
            if self.auto_load:
                self._auto_load_videos()

    def _ensure_locks(self):
        """确保锁已初始化"""
        if not self._locks_initialized:
            self._init_locks()

    def _setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _get_available_memory(self) -> int:
        """
        获取可用内存大小（字节）
        """
        memory = psutil.virtual_memory()
        return int(memory.total * self.max_memory_ratio - memory.used)

    def _get_video_files(self) -> List[Path]:
        """
        扫描文件夹获取所有视频文件
        """
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(self.folder_path.rglob(f"*{ext}"))
        return video_files

    def _calculate_video_priority(self, video_path: Path) -> float:
        """
        计算视频的加载优先级
        优先级基于:
        1. 文件大小（较小文件优先）
        2. 文件位置（根目录优先）
        3. 最近访问时间
        """
        stats = video_path.stat()

        # 标准化各个因素
        size_score = 1.0 / (1 + stats.st_size / 1024 / 1024)  # MB
        depth_score = 1.0 / (1 + len(video_path.relative_to(self.folder_path).parts))
        time_score = stats.st_atime / psutil.boot_time()

        # 加权计算
        priority = 0.4 * size_score + 0.3 * depth_score + 0.3 * time_score

        return priority

    def _auto_load_videos(self) -> None:
        """
        自动加载视频文件
        1. 扫描所有视频文件
        2. 计算优先级
        3. 按优先级加载，直到达到内存限制
        """
        try:
            # 获取所有视频文件
            video_files = self._get_video_files()
            if not video_files:
                self.logger.warning(f"未在 {self.folder_path} 找到视频文件")
                return

            # 计算优先级
            video_priorities = []
            for video_path in video_files:
                rel_path = str(video_path.relative_to(self.folder_path))
                priority = self._calculate_video_priority(video_path)
                video_priorities.append((-priority, rel_path))  # 负优先级用于最大堆

            # 按优先级排序
            heapq.heapify(video_priorities)

            # 逐个加载视频
            available_memory = self._get_available_memory()
            loaded_size = 0

            while video_priorities and loaded_size < available_memory:
                _, video_path = heapq.heappop(video_priorities)
                try:
                    file_size = (self.folder_path / video_path).stat().st_size
                    if loaded_size + file_size > available_memory:
                        continue

                    self.load_video(video_path)
                    loaded_size += file_size

                except Exception as e:
                    self.logger.error(f"加载视频失败: {video_path}, 错误: {str(e)}")
                    continue

            # 只在第一次初始化时打印日志
            if not hasattr(self, "_log_initialized") or not self._log_initialized:
                self.logger.info(
                    f"已加载 {len(self.video_data)} 个视频文件，"
                    f"总大小: {loaded_size/1024/1024:.2f}MB"
                )

        except Exception as e:
            self.logger.error(f"自动加载视频失败: {str(e)}")
            raise

    def load_video(self, video_path: str) -> None:
        """
        加载视频文件到内存
        如果内存不足，只清理帧缓存；如果仍然不足，则抛出异常
        """
        self._ensure_locks()
        with self.load_lock:
            try:
                video_path = str(Path(video_path))
                abs_path = self.folder_path / video_path

                if not abs_path.exists():
                    raise FileNotFoundError(f"视频文件不存在: {abs_path}")

                # 检查内存限制并尝试清理帧缓存
                file_size = abs_path.stat().st_size
                if file_size > self._get_available_memory():
                    self._free_memory(needed_size=file_size)

                # 读取和处理视频文件
                with open(abs_path, "rb") as f:
                    video_bytes = f.read()

                # 获取视频信息
                container = av.open(io.BytesIO(video_bytes))
                stream = container.streams.video[0]

                video_info = {
                    "frame_count": stream.frames,
                    "fps": float(stream.average_rate),
                    "width": stream.width,
                    "height": stream.height,
                    "duration": float(stream.duration * stream.time_base),
                    "size": file_size,
                }

                # 更新存储
                self.video_data[video_path] = video_bytes
                self.video_info[video_path] = video_info
                self.locks[video_path] = Lock()
                self.video_priorities[video_path] = self._calculate_video_priority(
                    abs_path
                )

                # self.logger.info(f"已加载视频: {video_path}, 大小: {file_size/1024/1024:.2f}MB")

            except Exception as e:
                self.logger.error(f"加载视频失败: {video_path}, 错误: {str(e)}")
                raise

    def _free_memory(self, needed_size: int) -> None:
        """
        释放内存以容纳新的视频文件

        策略:
        只清理帧缓存，保持视频文件始终在内存中
        """
        self._ensure_locks()
        with self.cache_lock:
            # 只清理帧缓存
            self.frame_cache.clear()
            self.prefetch_queues.clear()

            # 检查是否有足够的内存
            if needed_size > self._get_available_memory():
                raise MemoryError(
                    f"内存不足，无法加载新视频。需要 {needed_size/1024/1024:.2f}MB，"
                    f"当前可用 {self._get_available_memory()/1024/1024:.2f}MB"
                )
            if self.enable_gpu:
                # 检查GPU内存使用率，只在使用率高时清理
                allocated = torch.cuda.memory_allocated(self.gpu_id)
                total = torch.cuda.get_device_properties(self.gpu_id).total_memory
                if allocated / total > 0.8:  # 比如使用率超过80%时
                    torch.cuda.empty_cache()

    def get_loaded_videos(self) -> List[str]:
        """获取已加载的视频列表"""
        return list(self.video_data.keys())

    def get_video_info(self, video_path: str) -> Dict:
        """获取视频信息"""
        if video_path not in self.video_info:
            raise ValueError(f"视频未加载: {video_path}")
        return self.video_info[video_path].copy()

    def _get_video_container(self, video_path: str):
        """
        获取视频容器对象，支持GPU加速
        """
        video_bytes = io.BytesIO(self.video_data[video_path])
        options = {}

        if self.enable_gpu:
            # 设置CUDA硬件加速
            options = {
                "hwaccel": "cuda",
                "hwaccel_device": str(self.gpu_id),
                "hwaccel_output_format": "cuda",
            }

        container = av.open(video_bytes, options=options)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        return container

    def _decode_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """
        解码指定帧，仅在必要时清理GPU资源
        """
        self._ensure_locks()
        container = None
        with self.locks[video_path]:
            container = self._get_video_container(video_path)
            stream = container.streams.video[0]

            target_ts = int(frame_idx * stream.duration / stream.frames)
            container.seek(target_ts, stream=stream)

            frame_count = 0
            for frame in container.decode(video=0):
                if frame_count == 0:
                    if self.enable_gpu:
                        with torch.cuda.device(self.gpu_id):
                            result = frame.to_ndarray(format="rgb24")
                            del frame  # 只删除frame对象，让CUDA自己管理内存
                            return result
                    else:
                        return frame.to_ndarray(format="rgb24")
                frame_count += 1

    def _prefetch_frames(self, video_path: str, start_frame: int) -> None:
        """
        预加载后续帧
        """
        try:
            for i in range(start_frame, start_frame + self.prefetch_size):
                if i >= self.video_info[video_path]["frame_count"]:
                    break

                # 检查是否已在缓存中
                cache_key = (video_path, i)
                if cache_key in self.frame_cache:
                    continue

                # 解码并加入预加载队列
                frame = self._decode_frame(video_path, i)

                queue_key = (video_path, i)
                if queue_key not in self.prefetch_queues:
                    self.prefetch_queues[queue_key] = queue.Queue(maxsize=1)
                self.prefetch_queues[queue_key].put(frame)

        except Exception as e:
            self.logger.error(f"预加载失败: {str(e)}")

    def get_frame(
        self, video_path: str, frame_idx: int, max_retries: int = 100
    ) -> np.ndarray:
        """
        获取指定帧，支持缓存和预加载

        Args:
            video_path: 视频路径
            frame_idx: 帧序号
            max_retries: 最大重试次数
        """
        self._ensure_locks()
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # 路径处理
                video_path = str(Path(video_path))
                if video_path not in self.video_data:
                    self.load_video(video_path)

                # 检查帧号范围
                if frame_idx >= self.video_info[video_path]["frame_count"]:
                    raise IndexError(f"帧序号越界: {frame_idx}")

                # 检查缓存
                cache_key = (video_path, frame_idx)
                try:
                    with self.cache_lock:
                        if cache_key in self.frame_cache:
                            self.frame_cache.move_to_end(cache_key)
                            return self.frame_cache[cache_key]
                except OSError:
                    time.sleep(0.1)  # 短暂等待后重试
                    continue

                # 检查预加载队列
                queue_key = (video_path, frame_idx)
                if queue_key in self.prefetch_queues:
                    try:
                        frame = self.prefetch_queues[queue_key].get_nowait()
                        self.prefetch_queues.pop(queue_key)

                        # 更新缓存
                        try:
                            with self.cache_lock:
                                if len(self.frame_cache) >= self.cache_size:
                                    self.frame_cache.popitem(last=False)
                                self.frame_cache[cache_key] = frame
                        except OSError:
                            pass  # 缓存更新失败不影响返回结果

                        # 提交新的预加载任务
                        try:
                            self.executor.submit(
                                self._prefetch_frames, video_path, frame_idx + 1
                            )
                        except Exception:
                            pass  # 预加载失败不影响返回结果

                        return frame

                    except (queue.Empty, OSError):
                        pass

                # 直接解码
                frame = self._decode_frame(video_path, frame_idx)

                # 更新缓存
                try:
                    with self.cache_lock:
                        if len(self.frame_cache) >= self.cache_size:
                            self.frame_cache.popitem(last=False)
                        self.frame_cache[cache_key] = frame
                except OSError:
                    pass  # 缓存更新失败不影响返回结果

                # 提交预加载任务
                try:
                    self.executor.submit(
                        self._prefetch_frames, video_path, frame_idx + 1
                    )
                except Exception:
                    pass  # 预加载失败不影响返回结果

                return frame

            except OSError as e:
                last_error = e
                retry_count += 1
                time.sleep(0.1)  # 短暂等待后重试
                continue
            except Exception as e:
                self.logger.error(f"获取帧失败: {str(e)}")
                raise

        # 如果所有重试都失败
        self.logger.error(f"在{max_retries}次尝试后获取帧失败: {str(last_error)}")
        raise last_error

    def release(self) -> None:
        """释放所有资源"""
        self.executor.shutdown(wait=False)
        self.video_data.clear()
        self.video_info.clear()
        self.frame_cache.clear()
        self.prefetch_queues.clear()
        self.locks.clear()
        self.logger.info("已释放所有资源")

    def __del__(self):
        """析构函数"""
        self.release()


class VideoToShm:
    """将视频文件拷贝到共享内存的工具类"""

    def __init__(
        self,
        folder_path: str,
        shm_path: str = "",
        video_extensions: Set[str] = {".mp4", ".MP4", ".avi", ".mov"},
    ):
        if not shm_path:
            shm_path = "/dev/shm/video_data" + folder_path.replace("/", "_")
        self.shm_path = Path(shm_path)
        self.video_extensions = video_extensions
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 创建共享内存目录
        self.shm_path.mkdir(parents=True, exist_ok=True)
        self.folder_path = Path(folder_path)
        self._video_paths = []
        self.video_paths_to_shm_path = {}

    def get_meta_data(self):
        for ext in self.video_extensions:
            for video_path in self.folder_path.rglob(f"*{ext}"):
                rel_path = str(video_path.relative_to(self.folder_path))
                shm_file = self.shm_path / f"{rel_path}"
                self._video_paths.append(shm_file)
                self.video_paths_to_shm_path[os.path.realpath(video_path)] = shm_file

    def copy_videos(self) -> None:
        """
        将指定文件夹中的视频文件拷贝到共享内存
        如果目标文件已存在且大小相同，则跳过复制
        """
        video_info = {}

        try:
            # 扫描视频文件
            for ext in self.video_extensions:
                for video_path in self.folder_path.rglob(f"*{ext}"):
                    rel_path = str(video_path.relative_to(self.folder_path))
                    shm_file = self.shm_path / f"{rel_path}"
                    self._video_paths.append(shm_file)
                    self.video_paths_to_shm_path[os.path.realpath(video_path)] = (
                        shm_file
                    )

                    # 检查标文件是否存在且大小相同
                    if shm_file.exists():
                        src_size = video_path.stat().st_size
                        dst_size = shm_file.stat().st_size
                        if src_size == dst_size:
                            # self.logger.info(f"跳过复制(文件已存在): {rel_path}")
                            # 仍然需要获取视频信息
                            container = av.open(str(video_path))
                            stream = container.streams.video[0]
                            video_info[rel_path] = {
                                "frame_count": stream.frames,
                                "fps": float(stream.average_rate),
                                "width": stream.width,
                                "height": stream.height,
                                "duration": float(stream.duration * stream.time_base),
                                "size": src_size,
                            }
                            continue

                    # 创建目标目录并复制文件
                    shm_file.parent.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"正在拷贝: {rel_path}")
                    shutil.copy2(video_path, shm_file)
                    # 获取视频信息
                    container = av.open(str(video_path))
                    stream = container.streams.video[0]
                    video_info[rel_path] = {
                        "frame_count": stream.frames,
                        "fps": float(stream.average_rate),
                        "width": stream.width,
                        "height": stream.height,
                        "duration": float(stream.duration * stream.time_base),
                        "size": video_path.stat().st_size,
                    }

            # 保存元信息
            with open(self.shm_path / "metadata.json", "w") as f:
                json.dump(video_info, f)

            self.logger.info(f"已完成 {len(video_info)} 个视频文件的处理")

        except Exception as e:
            self.logger.error(f"拷贝视频失败: {str(e)}")
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
