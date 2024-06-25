import numpy as np
import open3d as o3d
import pandas as pd
from src.utils import *
from src.utils_plot import *


if __name__ == '__main__':
    shape = (1920, 1080)
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])  
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    valid_idx = np.arange(164, 293+1)
    
    # load pose
    tar = np.load('npy/idx.npy')
    pose_trans, pose_rotate = np.load('npy/trans.npy'), np.load('npy/rotation.npy')
    pose_trans, pose_rotate = pose_trans[tar.argsort()], pose_rotate[tar.argsort()]

    points, lines, colors = get_PtsLines(pose_rotate, pose_trans, shape, 
                                         cameraMatrix, distCoeffs)
    
    # use open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    pcd = load_point_cloud(pd.read_pickle("data/points3D.pkl"))
    vis.add_geometry(pcd)
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), 
                                    np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)
    
    vis.run()
    vis.destroy_window()