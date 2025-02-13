import os
from typing import Optional

import toml
from pydantic import BaseModel
from pathlib import Path
import time
import re
import random
import logging


def get_log_folder(log_root: str):
    log_folder = Path(log_root) / (
        time.strftime("%Y-%m-%d_%H-%M-%S_")
        + f"{int(time.time() * 1000000) % 1000000:06d}"
    )
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def get_json_log_path(log_folder: Path):
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
    new_filename = f"traj.json"
    return dir_path / new_filename


class Logger(object):
    """global logger

    Args:
        filename (str, optional): log file name. Defaults to None.
        level (str, optional): log level( debug info warning error critical ). Defaults to 'info'.
        fmt (str, optional): log format. Defaults to '[%(asctime)s][%(levelname)s] %(message)s'.
    PS:
        more format details at : https://docs.python.org/zh-cn/3/library/logging.html
    """

    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # '[%(asctime)s][%(levelname)s] %(pathname)s[line:%(lineno)d] -: %(message)s'
    def __init__(
        self,
        filename: str = None,
        level: str = "info",
        fmt: str = "[%(asctime)s][%(levelname)s] %(message)s",
    ):
        if filename == "None":
            filename = None
        self.log = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.log.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        self.log.addHandler(sh)
        # Logging file
        if filename is not None:
            th = logging.FileHandler(filename=filename, encoding="utf-8")
            th.setFormatter(format_str)
            self.log.addHandler(th)


class LogConfig(BaseModel):
    filename: Optional[str] = None
    level: Optional[str] = "info"
    fmt: Optional[str] = (
        "[%(asctime)s][%(levelname)s] %(pathname)s[line:%(lineno)d] -: %(message)s"
    )


with open(
    os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.ini"), "r"
) as f:
    config = LogConfig(**(toml.loads(f.read())["log"]))

# Use this rather than `Logger`
log = Logger(
    filename=config.filename,
    level=config.level,
    fmt=config.fmt,
).log
