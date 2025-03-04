import os
import argparse
import lmdb
from pathlib import Path
import pickle
import tqdm
from multiprocessing import Manager, Pool, Lock
import shutil

mult_proc = False
IMAGE_KEYS = [
    "observations/sim_images/wrist_camera",
    "observations/sim_images/camera_0",
    "observations/render_images/wrist_camera",
]
FIXED_CAMERA_NAMES = ["camera_0"]
CAMERA_NAMES = ["wrist_camera", "camera_0"]
if mult_proc:
    GLOBAL_DICT = Manager().dict()
else:
    GLOBAL_DICT = {}

DELETE_NUM_LOCK = Lock()
GLOBAL_DICT["broken_num"] = 0
GLOBAL_DICT["delete_num"] = 0
GLOBAL_DICT["broken_lmdb_dirs"] = []


def check_mask(directory):
    try:
        largest_nums = []
        if not os.path.exists(directory / "lmdb"):
            print(f"lmdb not found in {directory}")
            return True
        with lmdb.open(str(directory / "lmdb"), lock=False) as f:
            with f.begin(write=False) as txn:
                action = pickle.loads(txn.get("action".encode("utf-8")))
                length = len(action)
                for camera_name in CAMERA_NAMES:
                    for i in range(length):
                        mask_key = f"observations/mask/{camera_name}/{i}"
                        if txn.get(mask_key.encode("utf-8")) is None:
                            print(f"Not found '{mask_key}' in {directory}")
                            return True
        return False
    except Exception as e:
        import traceback

        # traceback.print_exc()
        # print(f"Error in check_mask: {e}")
        print(f"Found a broken lmdb: {directory}")
        return True


def check_lmdb(lmdb_dir: Path, fast=False):
    to_delete = check_mask(lmdb_dir)

    if to_delete:
        print(f"mask broken in {lmdb_dir}")
        return True

    # check other things
    to_delete = False
    if not os.path.exists(lmdb_dir / "lmdb"):
        assert False, f"lmdb not found in {lmdb_dir}"
    with lmdb.open(str(lmdb_dir / "lmdb"), lock=False) as f:
        with f.begin(write=False) as txn:
            action = pickle.loads(txn.get("action".encode("utf-8")))
            length = len(action)
            if not fast:
                for base_key in IMAGE_KEYS:
                    for i in range(length):
                        key = f"{base_key}/{i}"
                        if txn.get(key.encode("utf-8")) is None:
                            print(f"Not found '{key}' in {lmdb_dir}")
                            to_delete = True

                # check fixed camera
                for camera_name in FIXED_CAMERA_NAMES:
                    if (
                        txn.get(
                            f"observations/fix_render_images/{camera_name}".encode(
                                "utf-8"
                            )
                        )
                        is None
                    ):
                        print(
                            f"Not found 'observations/fix_render_images/{camera_name}' in {lmdb_dir}"
                        )
                        to_delete = True

            # check qpos
            if txn.get("observations/qpos".encode("utf-8")) is None:
                print(f"Not found 'observations/qpos' in {lmdb_dir}")
                to_delete = True

    return to_delete


def process_lmdb(lmdb_dir: Path, fast=False):
    try:
        to_delete = check_lmdb(lmdb_dir.parent, fast)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("Broken lmdb: ", lmdb_dir)
        to_delete = True
    if to_delete:
        with DELETE_NUM_LOCK:
            GLOBAL_DICT["broken_num"] += 1
            GLOBAL_DICT["broken_lmdb_dirs"].append(str(lmdb_dir.absolute()))
    return to_delete


def check_and_delete_render_images(directory, fast=False):
    info_dirs = list(Path(directory).rglob("info.json"))
    lmdb_dirs = []
    for info_dir in info_dirs:
        lmdb_dir = info_dir.parent / "lmdb"
        lmdb_dirs.append(lmdb_dir)
    if not mult_proc:
        for lmdb_dir in tqdm.tqdm(
            lmdb_dirs, desc=f"{len(GLOBAL_DICT['broken_lmdb_dirs'])} broken lmdb"
        ):
            process_lmdb(lmdb_dir, fast)
    else:
        with Pool(processes=os.cpu_count()) as pool:
            func = lambda x: process_lmdb(x, fast)
            results = list(tqdm.tqdm(pool.imap(func, lmdb_dirs), total=len(lmdb_dirs)))
    return GLOBAL_DICT["broken_lmdb_dirs"]


def main():
    parser = argparse.ArgumentParser(
        description="Check and optionally delete render_images from HDF5 files."
    )
    parser.add_argument(
        "directory", type=str, help="Directory to search for traj.hdf5 files"
    )
    parser.add_argument(
        "--delete", action="store_true", help="Delete render_images if found"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode, only check the lmdb directory"
    )
    args = parser.parse_args()

    check_and_delete_render_images(args.directory, args.fast)
    delete_now = input(f"是否删除 {GLOBAL_DICT['broken_num']} 个文件? (y/n)")
    if delete_now == "y" or args.delete:
        for lmdb_dir in GLOBAL_DICT["broken_lmdb_dirs"]:
            lmdb_dir = Path(lmdb_dir)
            shutil.rmtree(lmdb_dir.parent)
            print(f"已删除: {lmdb_dir.parent}")


if __name__ == "__main__":
    main()
