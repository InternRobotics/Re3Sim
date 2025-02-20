import os
import logging
import shutil
from argparse import ArgumentParser
from pycolmap import SceneManager, FeatureExtractor, ExhaustiveMatcher, Mapper, ImageUndistorter

parser = ArgumentParser("Colmap converter using pycolmap")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--resize", action="store_true")
args = parser.parse_args()
use_gpu = not args.no_gpu

scene_manager = SceneManager(database_path=os.path.join(args.source_path, "distorted", "database.db"),
                             image_path=os.path.join(args.source_path, "input"))

if not args.skip_matching:
    os.makedirs(os.path.join(args.source_path, "distorted", "sparse"), exist_ok=True)

    ## Feature extraction
    feature_extractor = FeatureExtractor(
        scene_manager,
        single_camera=True,
        camera_model=args.camera,
        use_gpu=use_gpu
    )
    feature_extractor.run()

    ## Feature matching
    matcher = ExhaustiveMatcher(scene_manager, use_gpu=use_gpu)
    matcher.run()

    ### Bundle adjustment
    mapper = Mapper(
        scene_manager,
        output_path=os.path.join(args.source_path, "distorted", "sparse"),
        ba_global_function_tolerance=1e-6
    )
    mapper.run()

### Image undistortion
undistorter = ImageUndistorter(
    scene_manager,
    input_path=os.path.join(args.source_path, "distorted", "sparse", "0"),
    output_path=args.source_path,
    output_type="COLMAP"
)
undistorter.run()

files = os.listdir(os.path.join(args.source_path, "sparse"))
os.makedirs(os.path.join(args.source_path, "sparse", "0"), exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if args.resize:
    from PIL import Image

    print("Copying and resizing...")

    # Resize images.
    os.makedirs(os.path.join(args.source_path, "images_2"), exist_ok=True)
    os.makedirs(os.path.join(args.source_path, "images_4"), exist_ok=True)
    os.makedirs(os.path.join(args.source_path, "images_8"), exist_ok=True)

    # Get the list of files in the source directory
    files = os.listdir(os.path.join(args.source_path, "images"))
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)
        for factor, folder in zip([2, 4, 8], ["images_2", "images_4", "images_8"]):
            destination_file = os.path.join(args.source_path, folder, file)
            img = Image.open(source_file)
            img = img.resize((img.width // factor, img.height // factor))
            img.save(destination_file)

print("Done.")
