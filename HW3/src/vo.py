import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
from numpy.linalg import inv


class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']

        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))
        self.H, self.W = self.get_HW()


    def get_HW(self):
        img = cv.imread(self.frame_paths[0])
        return img.shape[0], img.shape[1]
    
    
    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:

                    origin = np.array([0,0,0]).reshape(3,1)
                    c0 = np.array([0,0,1]).reshape(3,1)
                    c1 = np.array([self.H,0,1]).reshape(3,1)
                    c2 = np.array([self.H,self.W,1]).reshape(3,1)
                    c3 = np.array([0,self.W,1]).reshape(3,1)
            
                    c = np.concatenate((c0, c1, c2, c3), axis=1)

                    ct = np.concatenate((t,t,t,t),axis=1)
                   
                    w = inv(R).dot((inv(self.K).dot(c))-ct)

                    w0 = w[:,0].reshape(3,1)
                    w1 = w[:,1].reshape(3,1)
                    w2 = w[:,2].reshape(3,1)
                    w3 = w[:,3].reshape(3,1)


                    origin_p = inv(R).dot(origin-t)
                    
                    points = [
                        origin_p,
                        w0,
                        w1,
                        w2,
                        w3
                    ]
                    
                    lines = [[i,j] for i in range(len(points)) for j in range(i)]
                    
                    colors = [[1, 0, 0] for _ in range(len(lines))]
 
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    vis.add_geometry(line_set)
                    pass
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        # first image 
        img1 = cv.imread(self.frame_paths[0])
        
        # Initiate ORB detector
        orb = cv.ORB_create()

        kp1, des1 = orb.detectAndCompute(img1,None)

        
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        

        for idx, frame_path in enumerate(self.frame_paths[1:]):
            img2 = cv.imread(frame_path)
            #TODO: compute camera pose here
            
            kp2, des2 = orb.detectAndCompute(img2,None)

            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            kp1_list = np.array([kp1[ele.queryIdx].pt for ele in matches])
            kp2_list = np.array([kp2[ele.trainIdx].pt for ele in matches])

            E, mask = cv.findEssentialMat(kp1_list, kp2_list, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)                         
            retr, R, t, mask = cv.recoverPose(E, kp1_list, kp2_list, self.K) 

            
            P = np.concatenate((R,t), axis=1)
            
            row_ = np.array([[0,0,0,1]])
            P = np.concatenate((P,row_), axis=0)
            if idx == 0:
                R = P[:3,:3]
                t = P[0:3,2].reshape(3,1)
                P_pre = np.copy(P)
            else:
                P = np.dot(P, P_pre)
                R = P[:3,:3]
                t = P[0:3,3].reshape(3,1)
                P_pre = np.copy(P)
            
            queue.put((R, t))
            
            
            img_show = None
            kp = [kp2[ele.trainIdx] for ele in matches ]
            img_show = cv.drawKeypoints(img2, keypoints=kp, outImage=np.array([]) ,color=(0,0,255), flags=0)
            
            cv.imshow('frame', img_show)
            img1 = np.copy(img2)
            des1, kp1 = np.copy(des2), np.copy(kp2)
            if cv.waitKey(30) == 27: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='npy/camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()
    vo = SimpleVO(args)
    vo.run()