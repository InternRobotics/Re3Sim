## The script supposes that the robot is Franka FR3 or Franka PANDA

import argparse
import subprocess
import pycolmap
import os
import json
import requests
from pathlib import Path
import shutil


def extract_features(image_dir, database_path, output_dir, camera_model="PINHOLE"):
    """use pinhole model to extract features"""
    print("start extract features...")

    os.makedirs(output_dir, exist_ok=True)

    options = {
        "SiftExtraction": {},
        "ImageReaderOptions": {
            "camera_model": camera_model,
        },
    }

    sift_options = pycolmap.SiftExtractionOptions(**options["SiftExtraction"])
    image_reader = pycolmap.ImageReaderOptions(**options["ImageReaderOptions"])

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_model=camera_model,
        reader_options=image_reader,
        sift_options=sift_options,
    )

    return sift_options


def download_vocab_tree(output_dir, image_num):
    if image_num < 1000:
        download_url = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words32K.bin"
    elif image_num < 10000:
        download_url = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words256K.bin"
    else:
        download_url = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words1M.bin"

    vocab_tree_path = os.path.join(output_dir, "vocab_tree.bin")
    # Download vocabulary tree file
    print(f"Downloading vocabulary tree from {download_url}...")
    response = requests.get(download_url)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save file locally
    with open(vocab_tree_path, "wb") as f:
        f.write(response.content)

    print(f"Vocabulary tree downloaded to {vocab_tree_path}")
    return vocab_tree_path


def count_images(image_dir, exts=[".jpg", ".png", ".jpeg"]):
    image_num = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                image_num += 1
    return image_num


def feature_matching(database_path, vocab_tree_path, model="sequential"):
    if model == "sequential":
        match_options = pycolmap.SequentialMatchingOptions(
            vocab_tree_path=vocab_tree_path,
        )
        pycolmap.match_sequential(
            database_path=database_path, matching_options=match_options
        )
    elif model == "exhaustive":
        match_options = pycolmap.ExhaustiveMatchingOptions(
            block_size=500,
        )
        pycolmap.match_exhaustive(
            database_path=database_path, matching_options=match_options
        )
    else:
        raise NotImplementedError(f"Matching model {model} not implemented")


def run_colmap(image_dir, output_dir, exts=[".jpg", ".png", ".jpeg"]):
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join(output_dir, "database.db")
    sift_options = extract_features(image_dir, database_path, output_dir)
    image_num = count_images(image_dir, exts)
    vocab_tree_path = download_vocab_tree(output_dir, image_num)
    feature_matching(database_path, vocab_tree_path)
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=os.path.join(output_dir, "sparse"),
    )
    txt_output_dir = os.path.join(output_dir, "sparse", "text")
    os.makedirs(txt_output_dir, exist_ok=True)
    for idx, reconstruction in maps.items():
        reconstruction.write_text(txt_output_dir)
        print(
            f"save the {idx+1} reconstruction result to txt format in: {txt_output_dir}"
        )


def main():
    args = get_args()
    input_dir = args.input_dir
    image_dir = os.path.join(input_dir, "images")
    progress_file = os.path.join(input_dir, "progress.json")
    assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist"
    # colmap
    colmap_dir = os.path.join(input_dir, "colmap")
    jump_colmap = False
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
            if progress.get("colmap", False):
                print("colmap has done")
                jump_colmap = True
    if not jump_colmap:
        run_colmap(image_dir, colmap_dir)
        with open(progress_file, "w") as f:
            f.write(json.dumps({"colmap": True}))
    # gaussian
    jump_gaussian = False
    with open(progress_file, "r") as f:
        process = json.load(f)
        if process.get("gaussian", False):
            print("gaussian has done")
            jump_gaussian = True
    if not jump_gaussian:
        gaussian_dir = Path(__file__).parent.parent / "gaussian_splatting"
        subprocess.run(
            [
                "python",
                "train.py",
                "-s",
                os.path.abspath(colmap_dir),
                "-i",
                os.path.abspath(image_dir),
                "-m",
                os.path.join(os.path.abspath(input_dir), "gs/0"),
                "--random_background",
            ],
            cwd=gaussian_dir,
        )
        with open(progress_file, "w") as f:
            f.write(json.dumps({"colmap": True, "gaussian": True}))
    gs_path = os.path.join(input_dir, "gs/0/")
    # mvs
    jump_mvs_colmap_dense = False
    with open(progress_file, "r") as f:
        process = json.load(f)
        if process.get("mvs_colmap_dense", False):
            print("mvs_colmap_dense has done")
            jump_mvs_colmap_dense = True
    dense_dir = os.path.abspath(os.path.join(colmap_dir, "dense"))
    if not jump_mvs_colmap_dense:
        subprocess.run(
            [
                "colmap",
                "image_undistorter",
                "--image_path",
                image_dir,
                "--input_path",
                os.path.join(colmap_dir, "sparse", "0"),
                "--output_path",
                dense_dir,
                "--output_type",
                "COLMAP",
            ]
        )
        with open(progress_file, "w") as f:
            f.write(
                json.dumps({"colmap": True, "gaussian": True, "mvs_colmap_dense": True})
            )
    # mvs
    mvs_dir = os.path.abspath(os.path.join(input_dir, "mvs"))
    jump_mvs_colmap_dense_mvs = False
    with open(progress_file, "r") as f:
        process = json.load(f)
        if process.get("mvs_colmap_dense_mvs", False):
            print("mvs_colmap_dense_mvs has done")
            jump_mvs_colmap_dense_mvs = True
    if not jump_mvs_colmap_dense_mvs:
        # print(mvs_dir)
        os.makedirs(mvs_dir, exist_ok=True)
        print(os.path.join(dense_dir, "images"))
        subprocess.run(
            [
                "InterfaceCOLMAP",
                "-i",
                dense_dir,
                "-o",
                os.path.join(mvs_dir, "scene.mvs"),
                "--image-folder",
                os.path.join(dense_dir, "images"),
            ],
            cwd=mvs_dir,
        )
        subprocess.run(
            ["DensifyPointCloud", os.path.join(mvs_dir, "scene.mvs")], cwd=mvs_dir
        )
        subprocess.run(
            ["ReconstructMesh", "scene_dense.mvs", "-p", "scene_dense.ply"], cwd=mvs_dir
        )
        subprocess.run(
            [
                "RefineMesh",
                "scene_dense.mvs",
                "-m",
                "scene_dense_mesh.ply",
                "-o",
                "scene_dense_mesh_refine.mvs",
                "--scales",
                "1",
                "--max-face-area",
                "16",
            ],
            cwd=mvs_dir,
        )
        if args.texture:
            subprocess.run(
                [
                    "TextureMesh",
                    "scene_dense.mvs",
                    "-m",
                    "scene_dense_mesh_refine.ply",
                    "-o",
                    "scene_dense_mesh_refine_texture.mvs",
                ],
                cwd=mvs_dir,
            )
        with open(progress_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "colmap": True,
                        "gaussian": True,
                        "mvs_colmap_dense": True,
                        "mvs_colmap_dense_mvs": True,
                    }
                )
            )
        print("************************")
        print("************************")
        print()
        print("mvs_colmap_dense_mvs has done")
        print(
            "the textured result is in: ",
            os.path.join(mvs_dir, "scene_dense_mesh_refine_texture.ply"),
        )
        print(
            "the mesh result is in: ",
            os.path.join(mvs_dir, "scene_dense_mesh_refine.ply"),
        )
        print()
        print("************************")
        print("************************")

    mvs_path = os.path.join(mvs_dir, "scene_dense_mesh_refine.ply")
    if args.texture:
        mvs_path = os.path.join(mvs_dir, "scene_dense_mesh_refine_texture.ply")

    r2s_root_path = Path(__file__).parent.parent

    if not os.path.exists(os.path.join(gs_path, "gs_to_marker.npy")):
        # align
        subprocess.run(
            [
                "python",
                "../real-deployment/utils/compute_transform_to_marker_aruco.py",
                "--data_type",
                "gaussian",
                "--data_folder",
                gs_path,
                "--headless",
            ],
            cwd=r2s_root_path,
        )
    else:
        print("gs_to_marker.npy already exists")
    if not os.path.exists(os.path.join(mvs_dir, "mesh_to_marker.npy")):
        try:
            os.symlink(
                os.path.abspath(os.path.join(input_dir, "colmap/sparse")),
                os.path.join(mvs_dir, "sparse"),
            )
        except FileExistsError:
            pass
        try:
            os.symlink(
                os.path.abspath(os.path.join(input_dir, "images")),
                os.path.join(mvs_dir, "images"),
            )
        except FileExistsError:
            pass
        subprocess.run(
            [
                "python",
                "../real-deployment/utils/compute_transform_to_marker_aruco.py",
                "--data_type",
                "openmvs",
                "--data_folder",
                mvs_dir,
                "--headless",
            ],
            cwd=r2s_root_path,
        )
        shutil.copy(
            os.path.join(colmap_dir, "sparse/0/colmap_to_marker.npy"),
            os.path.join(mvs_dir, "mesh_to_marker.npy"),
        )

    # obj to usd
    subprocess.run(
        [
            "python",
            "utils/usd/obj_to_usd.py",
            "--obj_path",
            mvs_path,
            "--usd_dir",
            mvs_dir,
            "--collision_approximation",
            "meshSimplification",
        ],
        cwd=r2s_root_path,
    )
    # marker to base


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-c", "--calibration_image", type=str, default=None)
    parser.add_argument("-d", "--depth_image", type=str, default=None)
    parser.add_argument("-t", "--texture", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
