import numpy as np
import cv2


class IntrinsicsCalibrator:
    def __init__(self, aruco_dict, board):
        self.aruco_dict = aruco_dict
        self.board = board

    def calibrate_camera(self, images, camera_matrix_init=None):
        print("POSE ESTIMATION STARTS:")
        all_corners = []
        all_ids = []
        decimator = 0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        imsize = None

        for frame in images:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
                gray, self.aruco_dict
            )

            if len(corners) > 0:
                for corner in corners:
                    cv2.cornerSubPix(
                        gray,
                        corner,
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=criteria,
                    )
                res2 = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
                if (
                    res2[1] is not None
                    and res2[2] is not None
                    and len(res2[1]) > 3
                    and decimator % 1 == 0
                ):
                    all_corners.append(res2[1])
                    all_ids.append(res2[2])
                    imsize = gray.shape

            decimator += 1

        if imsize is None:
            raise ValueError("No corners detected in the provided images.")

        dist_coeffs_init = np.zeros((5, 1))
        flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO

        if camera_matrix_init is not None:
            flags += cv2.CALIB_USE_INTRINSIC_GUESS

        (
            ret,
            camera_matrix,
            distortion_coefficients0,
            rotation_vectors,
            translation_vectors,
            std_deviations_intrinsics,
            std_deviations_extrinsics,
            per_view_errors,
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=self.board,
            imageSize=imsize,
            cameraMatrix=camera_matrix_init,
            distCoeffs=dist_coeffs_init,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
        )

        return (
            ret,
            camera_matrix,
            distortion_coefficients0,
            rotation_vectors,
            translation_vectors,
        )
