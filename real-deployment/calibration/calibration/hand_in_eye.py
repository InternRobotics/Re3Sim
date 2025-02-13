import cv2
import numpy as np


class HandinEyeCalibrator:
    def __init__(self, intrinsics_matrix, dist_coeffs, charuco_dict, board):
        self.intrinsics_matrix = intrinsics_matrix
        self.dist_coeffs = dist_coeffs
        self.charuco_dict = charuco_dict
        self.board = board

    def estimate_pose(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.charuco_dict)

        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )  # can not pass
            if charuco_ids is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    self.board,
                    self.intrinsics_matrix,
                    self.dist_coeffs,
                    None,
                    None,
                )
                if valid:
                    return rvec, tvec
        return None, None

    def perform(self, rgb_list, pose_list):
        # Initialize lists to store data
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []

        # Loop over each data pair
        for rgb, pose in zip(rgb_list, pose_list):
            # Load pose data
            R_gripper2base = pose[0:3, 0:3]
            t_gripper2base = pose[0:3, 3]

            # Estimate pose from RGB image
            rvec, tvec = self.estimate_pose(rgb)
            if rvec.any() == None:
                return None, None

            R_target2cam = cv2.Rodrigues(rvec)[0]
            t_target2cam = tvec.reshape(3, 1)

            # Append data to lists
            R_gripper2base_list.append(R_gripper2base)
            t_gripper2base_list.append(t_gripper2base)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(t_target2cam)

        # Convert lists to arrays
        R_gripper2base_array = np.array(R_gripper2base_list)
        t_gripper2base_array = np.array(t_gripper2base_list)
        R_target2cam_array = np.array(R_target2cam_list)
        t_target2cam_array = np.array(t_target2cam_list)

        # Optional initial guess for camera-to-gripper transformation
        R_cam2gripper_guess = np.eye(3)
        t_cam2gripper_guess = np.zeros((3, 1))

        # Perform hand-eye calibration
        R_cam2gripper_avg, t_cam2gripper_avg = cv2.calibrateHandEye(
            R_gripper2base_array,
            t_gripper2base_array,
            R_target2cam_array,
            t_target2cam_array,
            R_cam2gripper_guess,
            t_cam2gripper_guess,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        # Return the average results
        return R_cam2gripper_avg, t_cam2gripper_avg
