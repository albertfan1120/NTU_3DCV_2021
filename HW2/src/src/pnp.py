import numpy as np
from .p3p import P3P
from .dlt import DLT
from .utils import WorldToCamera, undistPts


class PnP():
    def __init__(self, cameraMatrix, distCoeffs):
        self.K = cameraMatrix
        self.distCoeffs = distCoeffs
        
        
    def solvePnPRansac(self, points3D, points2D, method, n_iters = 1200, threshold = 0.001):
        if method == "P3P":
            n_samples = 4 # 4 pts for p3p
            solver = P3P(self.K, self.distCoeffs)
        elif method == "DLT": 
            n_samples = 6 # 6 pts for DLT
            solver = DLT(self.K, self.distCoeffs)
        
        N = points3D.shape[0]
        index = np.arange(N)
        
        n_inliers = 0
        best_RT = None
        # RANSAC iteration start
        for _ in range(n_iters):
            # 1. Select random set of matches
            np.random.shuffle(index)
            samples_index = index[:n_samples]
            samples_3D = points3D[samples_index]
            samples_2D = points2D[samples_index]
            
            # if same points in this set, contunue
            if np.unique(samples_3D, axis = 0).shape[0] < n_samples : continue
            
            # 2. Compute pose
            R_m, T = solver.solve(samples_2D, samples_3D)
            
            # 3. Compute inliers via Euclidean distance
            pts3D_c = WorldToCamera(samples_3D, R_m, T)
            pts2D_c = undistPts(samples_2D, self.K, self.distCoeffs)
            
            error = ((pts3D_c - pts2D_c) ** 2).sum(axis = 1) ** 0.5
            inliers = error < threshold
            
            # 4. Keep the largest set of inliers
            if np.sum(inliers) > n_inliers: 
                n_inliers = np.sum(inliers)
                best_RT = (R_m, T)

        return best_RT