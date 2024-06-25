from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import linalg as LA
import pandas as pd
import cv2


def undistPts(pts2D, cameraMatrix, distCoeffs):
    return cv2.undistortPoints(pts2D, cameraMatrix, distCoeffs).squeeze()


def read_pkl(root):
    images_df = pd.read_pickle(root + "data/images.pkl")
    train_df = pd.read_pickle(root + "data/train.pkl")
    points3D_df = pd.read_pickle(root + "data/points3D.pkl")
    point_desc_df = pd.read_pickle(root + "data/point_desc.pkl")
    
    return images_df, train_df, points3D_df, point_desc_df


def avg_inliers(x):
    max_std = 3
    dim = x.shape[1] # 128 
    mask = np.sum(abs(x - np.mean(x, axis = 0)) <= max_std * np.std(x, axis = 0), axis = 1) > \
           0.8 * dim
    x_mean = list(np.mean(x[mask], axis = 0))
    
    return x_mean


def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(avg_inliers)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc


def load_query(point_desc_df, idx):
    points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
    kp_query = np.array(points["XY"].to_list())
    desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
    
    return kp_query, desc_query


def get_gt(images_df, idx):
    ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values
    
    return rotq_gt, tvec_gt


def save_pose(path, trans, rotate):
    np.save(path + '/rotation.npy', np.array(rotate))
    np.save(path + '/trans.npy', np.array(trans))


def WorldToPIxel(rotate, trans, cameraMatrix, pts):
    # u = K(RX + T)
    pts = cameraMatrix.dot((rotate.dot(pts.T).T + trans).T).T
    pts /= (np.expand_dims(pts[:, -1], axis = -1) + 1e-12)
    
    return pts[:, :-1]


def WorldToCamera(pts3D, R, T):
    pts = R.dot(pts3D.T).T + T
    pts /= (np.expand_dims(pts[:, -1], axis = -1) + 1e-12)
    return pts[:, :-1]


def compute_error(r, r_gt, t, t_gt):
    error_t = LA.norm(t - t_gt)

    relative_rvec = (R.from_quat(r_gt) * R.from_quat(r).inv()).as_rotvec().flatten()
    error_r = LA.norm(relative_rvec)
    
    return error_r, error_t