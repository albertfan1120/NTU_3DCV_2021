import numpy as np
import pandas as pd
from src.utils import *
from src.utils_plot import *
import cv2
import imageio


if __name__ == '__main__':
    shape = (1920, 1080)
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])  
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    valid_idx = np.arange(164, 293+1)
    
    # load pkl
    images_df = pd.read_pickle("data/images.pkl")
    
    # load npy
    vertices = np.load("npy/cube_vertices.npy")
    tar = np.load('npy/idx.npy')
    pose_trans, pose_rotate = np.load('npy/trans.npy'), np.load('npy/rotation.npy')
    
    # cibe setting
    color = [[255,   0,   0], #back
             [255, 255,   0], #top
             [255,   0, 255], #left
             [  0, 225, 255], #right
             [  0,   0, 255], #bottom
             [  0, 125,   0]] #front
    ver_idx = [[2, 3, 7, 6],   #back
               [2, 3, 1, 0],   #top
               [2, 6, 4, 0],   #left
               [3, 7, 5, 1],   #right
               [6, 7, 5, 4],   #bottom
               [0, 1, 5, 4]]   #front
    
    # corordinate in first 3 col, color for last 3 col
    cube_pts = getCube(vertices, ver_idx, color)
    
    # sort from far to close
    z_col = 2
    cube_pts = cube_pts[cube_pts[:, z_col].argsort()][::-1]
     
    # draw cube on image and save gif
    shape_v = tuple([valid_idx.shape[0]]) + shape
    video = getVideo(images_df, cube_pts, pose_rotate, pose_trans, 
                     cameraMatrix, shape_v, valid_idx)[tar.argsort()]
    
    new_video = []
    for frame in video:
        new_video.append(cv2.resize(frame, (shape[1]//3, shape[0]//3), interpolation=cv2.INTER_AREA))
    video = np.array(new_video)
    imageio.mimsave('output/cube.gif', video[:, :, :, ::-1], duration = 0.01)