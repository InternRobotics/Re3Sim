import os
import re
import argparse
from collections import defaultdict


def process_directory(directory, min_step, keep_count, dry_run=False):
    pattern = re.compile(r"policy_step_(\d+)_seed.*\.ckpt")
    files_by_dir = defaultdict(list)

    # 遍历目录，收集文件信息
    for root, _, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                step = int(match.group(1))
                full_path = os.path.join(root, file)
                files_by_dir[root].append((step, full_path))

    # 处理每个目录
    for dir_path, files in files_by_dir.items():
        files.sort(reverse=True)  # 按 step 值降序排序
        files_to_keep = []
        files_to_delete = []

        for step, file_path in files:
            if step >= min_step:
                if len(files_to_keep) < keep_count:
                    files_to_keep.append(file_path)
                else:
                    files_to_delete.append(file_path)
            else:
                files_to_delete.append(file_path)

        # 删除文件
        for file_path in files_to_delete:
            if dry_run:
                print(f"Would delete: {file_path}")
            else:
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        # 打印保留的文件
        for file_path in files_to_keep:
            print(f"Kept: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up policy step files in directories."
    )
    parser.add_argument("directory", help="The root directory to process")
    parser.add_argument(
        "--min-step",
        type=int,
        default=50000,
        help="Minimum step number to keep (default: 50000)",
    )
    parser.add_argument(
        "--keep-count",
        type=int,
        default=5,
        help="Number of files to keep per directory (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually deleting files",
    )
    args = parser.parse_args()

    process_directory(args.directory, args.min_step, args.keep_count, args.dry_run)


if __name__ == "__main__":
    main()
