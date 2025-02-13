from . import BaseLogger, get_sub_log_path
import lmdb
import numpy as np
import pickle
import cv2
import json
from loguru import logger


class LmdbLogger(BaseLogger):
    def __init__(
        self,
        log_root_path: str,
        disk_space_threshold: int = 5,
        max_size: int = 1,
        image_quality: int = 40,
        save_depth: bool = False,
        save_depth_as_png: bool = False,
    ):
        super().__init__(
            log_root_path, disk_space_threshold, save_depth, save_depth_as_png
        )
        self.max_size = int(max_size * 1024**4)
        self.image_quality = image_quality

    def close(self):
        pass

    def save(self):
        sub_log_path = get_sub_log_path(self.log_root_path)
        logger.info(f"Saving log data to: {sub_log_path}")
        sub_log_path_lmdb = sub_log_path / "lmdb"
        self.env = lmdb.open(str(sub_log_path_lmdb), map_size=self.max_size)

        txn = self.env.begin(write=True)
        # save scalar data
        for key, value in self.scalar_data_logger.items():
            txn.put(key.encode("utf-8"), pickle.dumps(value))

        # save image data jpg
        for key, value in self.image_data_logger_jpg.items():
            for i, image in enumerate(value):
                txn.put(
                    f"{key}/{i}".encode("utf-8"),
                    pickle.dumps(
                        cv2.imencode(
                            ".jpg",
                            image.astype(np.uint8),
                            [cv2.IMWRITE_JPEG_QUALITY, self.image_quality],
                        )[1]
                    ),
                )

        # save image data png
        for key, value in self.image_data_logger_png.items():
            for i, image in enumerate(value):
                txn.put(
                    f"{key}/{i}".encode("utf-8"),
                    pickle.dumps(cv2.imencode(".png", image.astype(np.uint8))[1]),
                )

        # save json data
        with open(sub_log_path / "info.json", "w") as f:
            json.dump(self.json_data_logger, f)
        txn.put("json_data".encode("utf-8"), pickle.dumps(self.json_data_logger))

        if self.save_depth:
            if self.save_depth_as_png:
                for key, value in self.depth_data_logger.items():
                    for i, depth in enumerate(value):
                        depth = np.clip(depth, 0.15, 1.5)  # Hardcoded depth range
                        depth_normalize = (
                            (depth - np.min(depth))
                            / (np.max(depth) - np.min(depth))
                            * 255
                        )
                        depth_uint8 = depth_normalize.astype(np.uint8)
                        txn.put(
                            f"{key}/{i}".encode("utf-8"),
                            pickle.dumps(cv2.imencode(".png", depth_uint8)[1]),
                        )
            else:
                # save depth data
                for key, value in self.depth_data_logger.items():
                    for i, depth in enumerate(value):
                        txn.put(
                            f"{key}/{i}".encode("utf-8"),
                            pickle.dumps(
                                cv2.imencode(".exr", depth.astype(np.float32))[1]
                            ),
                        )

        txn.commit()
        self.env.close()

        if not self.check_disk_space():
            return False
        return True
