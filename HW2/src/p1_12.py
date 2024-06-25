import numpy as np
import time
import argparse
from src.utils import *
from src.solve import pnpsolver


if __name__ == '__main__':
    # parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default = "P3P",)
    args = parser.parse_args()
    method = args.method
    assert method in ["P3P", 'DLT'], 'Your method should be P3P or DLT!!'
    
    images_df, train_df, points3D_df, point_desc_df = read_pkl(root = "")
    
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])  
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    valid_idx = np.arange(164, 293+1)
    
    # fix seed
    np.random.seed(320)
    
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df) 
    kp_model = np.array(desc_df["XYZ"].to_list()) 
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32) 
    
    trans_error = np.zeros_like(valid_idx, dtype = 'float')
    rotate_error = np.zeros_like(valid_idx, dtype = 'float')
    pose_trans = np.zeros((valid_idx.shape[0], 3), dtype = 'float')
    pose_rotate = np.zeros((valid_idx.shape[0], 4), dtype = 'float')
    t_start = time.time() 
    for i, idx in enumerate(valid_idx):
        kp_query, desc_query = load_query(point_desc_df, idx)
        rotq, tvec = pnpsolver((kp_query, desc_query),(kp_model, desc_model), 
                                cameraMatrix, distCoeffs, method)
        
        pose_rotate[i], pose_trans[i] = rotq, tvec
        rotq_gt, tvec_gt = get_gt(images_df, idx)
        rotate_error[i], trans_error[i] = compute_error(rotq, rotq_gt, tvec, tvec_gt)
        
        print("Process NO.{} image".format(idx))
        print("Trans error = {:.4f}".format(trans_error[i]))
        print("Rotate error = {:.4f}\n".format(rotate_error[i]))
    t_end = time.time()  
    
    print("-------------------- Report -----------------------")
    print("Your method is {}".format(method))
    print("Translation error (median/mean/max): {:.4f}/{:.4f}/{:.4f}"
          .format(np.median(trans_error), np.mean(trans_error), np.max(trans_error)))
    print("Rotation error (median/mean/max): {:.4f}/{:.4f}/{:.4f}"
          .format(np.median(rotate_error), np.mean(rotate_error), np.max(rotate_error)))
    print("Total computation time: {:.0f}s".format(t_end - t_start))
    print("---------------------------------------------------\n")
    
    print("Save pose data......")
    save_pose("npy", pose_trans, pose_rotate)
    print("Save pose data sucessfully!!")