import os
import shutil
import re
from pathlib import Path
import argparse


def move_files_and_folders(file_name, source_path, output_path):
    index = 0
    lmdb_path = Path(source_path).rglob(file_name)
    for source_file in lmdb_path:
        relative_path = os.path.relpath(source_file.parent, source_path)
        relative_name = relative_path.split("/")
        relative_name = "-".join(relative_name)
        relative_name = f"{source_path.split('/')[-1]}-{relative_name}.hdf5"
        target_file = Path(output_path) / relative_name

        # Move file
        if not target_file.exists():
            os.symlink(Path(source_file).absolute(), target_file.absolute())
            print(f"Linked file: {source_file} -> {target_file}")
        index += 1
    print(f"Processed {index} files")


parser = argparse.ArgumentParser(description="Process dataset files")
parser.add_argument(
    "--source-path",
    type=str,
    default="lmdb_data/pick_one_item",
    help="Source data path",
)
parser.add_argument(
    "--output-path", type=str, default="data/pick_one", help="Output path"
)
parser.add_argument(
    "--file-name", type=str, default="info.json", help="a unique file at the same level as the lmdb folder"
)

args = parser.parse_args()
source_path = args.source_path
output_path = args.output_path
file_name = args.file_name
os.makedirs(output_path, exist_ok=True)
move_files_and_folders(file_name, source_path, output_path)

print(os.environ["HOME"])
