import numpy as np 
from .utils import undistPts


class DLT():
    def __init__(self, cameraMatrix, distCoeffs):
        self.K = cameraMatrix
        self.distCoeffs = distCoeffs
        
    
    def solve(self, pts2D, pts3D):
        pts2D = undistPts(pts2D, self.K, self.distCoeffs)
        return self.getRT(self.getP(pts2D, pts3D), pts3D[0])
    
    
    def getP(self, pts2D, pts3D):
        A = []
        for i in range(pts3D.shape[0]):
            x, y, z = pts3D[i, 0], pts3D[i, 1], pts3D[i, 2]
            u, v = pts2D[i, 0], pts2D[i, 1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v])
            
        U, S, Vh = np.linalg.svd(np.array(A))
        P = Vh[-1, :].reshape((3, 4))

        return P
    
    
    def getRT(self, P, pts3D):
        R, T = P[:, :3], P[:, -1]
    
        U, S, Vh = np.linalg.svd(R)
        
        beta = 1/(S.sum()/3)
        if np.dot(P[-1, :], np.concatenate((pts3D, np.array([1])))) <= 0: beta *= -1
            
        R, T = beta*R, beta*T
        
        return R, T