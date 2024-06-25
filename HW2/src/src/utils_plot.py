from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import linalg as LA
import open3d as o3d
import cv2
from .utils import undistPts, WorldToPIxel


def load_img(images_df, idx):
    fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
    img = cv2.imread("data/frames/" + fname)
    
    return img
    

def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd


def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat 


def get_PtsLines(pose_R, pose_T, shape, K, distCoeffs):
    # u = K(RX + T),   X = inv(R) * (inv(K) * u - T)
    H, W = shape
    origin = np.array([0, 0, 0])
    corner = np.array([[0, 0], [H-1, 0], [H-1, W-1], [0, W-1]], dtype = 'float')
    corner = np.hstack((undistPts(corner, K, distCoeffs) , np.ones((corner.shape[0], 1))))

    points = np.empty((0, 3))
    lines = np.empty((0, 2))
    colors = np.empty((0, 3))
    
    for i, (R_m, T) in enumerate(zip(pose_R, pose_T)):
        inv_R = LA.inv(R.from_quat(R_m).as_matrix())
        
        c_origin_in_w = inv_R.dot(origin - T)
        corners_in_w = inv_R.dot((corner - T).T).T

        points = np.vstack((points, np.vstack((c_origin_in_w, corners_in_w))))
        lines = np.vstack((lines, np.array([[i*5+0, i*5+1], [i*5+0, i*5+2], 
                                            [i*5+0, i*5+3], [i*5+0, i*5+4],
                                            [i*5+1, i*5+2], [i*5+2, i*5+3], 
                                            [i*5+3, i*5+4], [i*5+4, i*5+1]])))
        colors = np.vstack((colors, np.array([[0, 0, 0], [0, 0, 0], 
                                              [0, 0, 0], [0, 0, 0], 
                                              [0, 0, 1], [0, 0, 1], 
                                              [0, 0, 1], [0, 0, 1]])))
        if i > 0: # for trajectory line
            lines = np.vstack((lines, np.array([[i*5, (i-1)*5]])))
            colors = np.vstack((colors, np.array([[1, 0, 0]])))
    
    return points, lines, colors


def getSurfacePts(corners, color):
    num = 12 
    dx = (corners[1] - corners[0]) / (num - 1)
    dy = (corners[2] - corners[1]) / (num - 1)
    pts_in_w = np.array([corners[0] + dx*i + dy*j for j in range(num) for i in range(num)])
    color = np.repeat(np.array([color]), pts_in_w.shape[0], axis = 0)

    return np.hstack((pts_in_w, color))


def getCube(vertices, ver_idx, color):
    cube_pts = np.empty((0, 6))
    for idx, color in zip(ver_idx, color):
        cube_pts = np.vstack((cube_pts, getSurfacePts(vertices[idx], color)))
    
    return cube_pts


def getVideo(images_df, cube_pts, R_m, T, K, shape, val_idx):
    N, H, W = shape
    video = np.zeros((N, H, W, 3))
    for i, img_idx in enumerate(val_idx):
        r, t = R.from_quat(R_m[i]).as_matrix(), T[i]
        img = load_img(images_df, img_idx)
        
        cube_in_world, cube_colors = cube_pts[:, :3], cube_pts[:, 3:]
        cube_in_c = WorldToPIxel(r, t, K, cube_in_world)
        
        for (h, w), color in zip(cube_in_c, cube_colors):
            h, w = round(h), round(w)
            if 0 <= h < H and 0 <= w < W: 
                cv2.circle(img, (h, w), 4, color, -1)
        video[i] = img
        
    return video.astype("uint8")