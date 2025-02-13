import cv2
import numpy as np


def estimate_pose(image, charuco_dict, intrinsics_matrix, dist_coeffs, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_dict)

    if len(corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )  # can not pass
        if charuco_ids is not None and len(charuco_corners) > 3:
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                board,
                intrinsics_matrix,
                dist_coeffs,
                None,
                None,
            )
            if valid:
                R_target2cam = cv2.Rodrigues(rvec)[0]
                t_target2cam = tvec.reshape(3, 1)
                target2cam = np.eye(4)
                target2cam[:3, :3] = R_target2cam
                target2cam[:3, 3] = t_target2cam.reshape(-1)
                return np.linalg.inv(target2cam)
    return None


if __name__ == "__main__":
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, charuco_dict)
