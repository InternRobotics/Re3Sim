import h5py
import cv2
import numpy as np
import imageio
import time


def show_videos_from_hdf5(file_path, fps, keys="observations/images"):
    # Open HDF5 file
    log_video = True
    with h5py.File(file_path, "r") as f:
        # Get all camera names
        camera_names = list(f[keys].keys())

        # Create dictionary to store image data for each camera
        camera_images = {}
        for camera_name in camera_names:
            camera_images[camera_name] = f[f"{keys}/{camera_name}"]

        # Calculate delay between frames
        delay = int(1000 / fps)

        # Assume all cameras have same image size, get dimensions from first camera's first frame
        first_camera = camera_names[0]
        frame_height, frame_width = camera_images[first_camera][0].shape[:2]

        # Get total number of frames (assume all cameras have same number of frames)
        num_frames = camera_images[first_camera].shape[0]

        if log_video:
            # Define video codec and output files
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writers = {}
            for camera_name in camera_names:
                video_writers[camera_name] = cv2.VideoWriter(
                    f"{camera_name}_video.mp4", fourcc, fps, (frame_width, frame_height)
                )

        # Loop through frames
        paused = False
        i = 0
        while i < num_frames:
            if not paused:
                for camera_name in camera_names:
                    # Read current frame from each camera
                    frame = camera_images[camera_name][i]
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = np.array(frame, dtype=np.uint8)

                    # Display image from each camera
                    cv2.imshow(camera_name, frame)

                    if log_video:
                        video_writers[camera_name].write(frame)

                i += 1
            else:
                print(f"paused at frame {i}")
            time.sleep(delay / 1000)
            # Wait for specified time until next frame
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord(" "):
                paused = not paused

        # Release all windows
        cv2.destroyAllWindows()

        if log_video:
            for writer in video_writers.values():
                writer.release()

import argparse

keys = "observations/images"
parser = argparse.ArgumentParser(description="Display videos from HDF5 file")
parser.add_argument("file_path", type=str, help="Path to HDF5 file")
parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
parser.add_argument("--log_video", action="store_true", help="Whether to save video")
parser.add_argument("--keys", type=str, default=keys, help="Keys for video data")
args = parser.parse_args()

file_path = args.file_path
fps = args.fps
log_video = args.log_video
print(f"keys: {args.keys}")
show_videos_from_hdf5(file_path, fps, keys=args.keys)
