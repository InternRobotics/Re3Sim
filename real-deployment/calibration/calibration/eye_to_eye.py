import cv2
import numpy as np


class EyetoEyeCalibrator:
    def __init__(
        self,
        intrinsic_matrix1,
        dist_coeffs1,
        intrinsic_matrix2,
        dist_coeffs2,
        charuco_dict,
        board,
    ):
        self.intrinsic_matrix1 = intrinsic_matrix1
        self.dist_coeffs1 = dist_coeffs1
        self.intrinsic_matrix2 = intrinsic_matrix2
        self.dist_coeffs2 = dist_coeffs2
        self.charuco_dict = charuco_dict
        self.board = board

    def estimate_pose(self, image, intrinsic_matrix, dist_coeffs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.charuco_dict)

        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )
            if charuco_ids is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    self.board,
                    intrinsic_matrix,
                    dist_coeffs,
                    None,
                    None,
                )
                if valid:
                    return rvec, tvec
        return None, None

    def calculate_relative_pose(self, rvec1, tvec1, rvec2, tvec2):
        R_world_to_cam1, _ = cv2.Rodrigues(rvec1)
        R_world_to_cam2, _ = cv2.Rodrigues(rvec2)

        R_cam1_to_cam2 = np.dot(R_world_to_cam2, R_world_to_cam1.T)
        t_cam1_to_cam2 = tvec2 - np.dot(R_cam1_to_cam2, tvec1)

        return R_cam1_to_cam2, t_cam1_to_cam2

    def perform(self, rgb_list1, rgb_list2):
        relative_poses = []

        for image1, image2 in zip(rgb_list1, rgb_list2):
            rvec1, tvec1 = self.estimate_pose(
                image1, self.intrinsic_matrix1, self.dist_coeffs1
            )
            rvec2, tvec2 = self.estimate_pose(
                image2, self.intrinsic_matrix2, self.dist_coeffs2
            )

            if rvec1 is not None and rvec2 is not None:
                relative_pose = self.calculate_relative_pose(rvec1, tvec1, rvec2, tvec2)
                relative_poses.append(relative_pose)

        # Calculate average relative pose
        R_avg = np.zeros((3, 3))
        t_avg = np.zeros((3, 1))
        for R, t in relative_poses:
            R_avg += R
            t_avg += t

        # make sure R_avg is a proper rotation matrix
        R_avg /= len(relative_poses)
        U, S, Vt = np.linalg.svd(R_avg)
        R_avg = np.dot(U, Vt)

        t_avg /= len(relative_poses)

        print("Average relative rotation:")
        print(R_avg)
        print("Average relative translation:")
        print(t_avg)

        tot_error = 0
        for R, t in relative_poses:
            error = np.linalg.norm(R - R_avg) + np.linalg.norm(t - t_avg)
            tot_error += error
        mean_error = tot_error / len(relative_poses)
        print("Mean error:", mean_error)

        return R_avg, t_avg
