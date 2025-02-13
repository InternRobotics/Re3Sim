import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from src.frankapy.src.utils.arcuo_marker import estimate_pose


def calibrate_camera_to_base(intrinsic_matrix, image_path, output_path):
    print(intrinsic_matrix)
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, charuco_dict)

    image = cv2.imread(str(image_path))
    cam_2_marker = estimate_pose(
        image, charuco_dict, intrinsic_matrix, np.zeros(5), board
    )
    if cam_2_marker is None:
        print("Failed to detect marker in image")
        return

    marker_2_base_path = Path(
        "/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/assets/marker_2_base_1017.npy"
    )
    marker_2_base = np.load(marker_2_base_path)
    cam_2_base = marker_2_base @ cam_2_marker

    print("intrinsic_matrix:")
    print(intrinsic_matrix)

    print("Camera to robot base transformation matrix:")
    print(cam_2_base)

    rotation = R.from_matrix(cam_2_base[:3, :3])
    translation = cam_2_base[:3, 3]

    print("\nRotation (quaternion x, y, z, w):")
    print(rotation.as_quat())
    print("\nTranslation (x, y, z):")
    print(translation)

    # Save results
    np.savez(
        output_path,
        transform=cam_2_base,
        rotation_quat=rotation.as_quat(),
        translation=translation,
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    data_root = "/isaac-sim/src/frankapy/logs/2024-10-30_16-18-06"
    data_root = Path(data_root)

    intrinsic_zip = np.load(data_root / "intrinsics.npz", allow_pickle=True)["arr_0"]
    print(intrinsic_zip)
    for i, color_root in enumerate(data_root.glob("color_*")):
        print(color_root)
        idd = int(color_root.stem.split("_")[-1])
        fx, fy, cx, cy = (
            intrinsic_zip[idd]["fx"],
            intrinsic_zip[idd]["fy"],
            intrinsic_zip[idd]["ppx"],
            intrinsic_zip[idd]["ppy"],
        )
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        image_paths = list(color_root.glob("*.png"))
        image_paths.sort(key=lambda x: int(x.stem), reverse=True)

        image_path = image_paths[0]
        calibrate_camera_to_base(
            intrinsic_matrix, image_path, data_root / f"camera_to_base_{idd}.npz"
        )
