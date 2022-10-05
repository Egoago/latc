import cv2
import numpy as np
import glob
from tqdm import tqdm

from latc import utils


def calibrate(img_dir, save_path):
    CHECKERBOARD = (6, 9)
    square_size = 23.95  # in millimeters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)

    threedpoints = []  # Vector for 3D points
    twodpoints = []  # Vector for 2D points
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

    images = glob.glob(img_dir+'/*.jpg')

    for filename in tqdm(images, unit="image", desc="Detecting corners"):
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        success, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
        if success:
            threedpoints.append(objectp3d)
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
            twodpoints.append(corners2)
            # image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
        else:
            print(f"Corners could not be detected for {filename}")
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    cam_params = utils.CameraParameters(matrix, distortion)
    cam_params.save(save_path)
