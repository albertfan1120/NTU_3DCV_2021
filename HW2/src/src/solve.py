from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
from .pnp import PnP


def pnpsolver(query, model, cameraMatrix, distCoeffs, method):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.85 * n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    rvec, tvec = PnP(cameraMatrix, distCoeffs).solvePnPRansac(points3D, points2D, method)

    return R.from_matrix(rvec).as_quat(), tvec