import numpy as np
from numpy import linalg as LA
from .utils import undistPts


class P3P():
    def __init__(self, cameraMatrix, distCoeffs):
        self.K = cameraMatrix
        self.distCoeffs = distCoeffs
        
        
    def solve(self, pts2D, pts3D):
        pts2D = undistPts(pts2D, self.K, self.distCoeffs)
        
        # in camera corordinate
        v1, v2, v3 = np.hstack((pts2D[0], 1)),np.hstack((pts2D[1], 1)), np.hstack((pts2D[2], 1)) 
        
        # in world corordinate
        X1, X2, X3 = pts3D[0], pts3D[1], pts3D[2] 
        
        C_ab, C_bc, C_ac = self.Cosine(v1, v2), self.Cosine(v2, v3), self.Cosine(v1, v3)
        R_ab, R_bc, R_ac = LA.norm(X1 - X2), LA.norm(X2 - X3), LA.norm(X1 - X3)
        
        K_1, K_2 = (R_bc / R_ac)**2, (R_bc / R_ab)**2
        roots = np.roots(self.getCoefficient(C_ab, C_bc, C_ac, K_1, K_2))
        real_root = np.array([root.real for root in roots if np.isreal(root)]) 
        
        bestR, bestT = np.zeros((3, 3)), np.zeros(3)
        if real_root.size:
            a, b, c = self.getLength(real_root, R_ab, C_ab, C_bc, C_ac, K_1, K_2)
            centers = self.trilateration(X1, X2, X3, a, b, c)
            if centers.size:
                all_R, all_T  = self.getRT(centers, pts3D[:3], np.hstack((pts2D[:3], np.ones((3, 1)))))
                bestR, bestT = self.reproject(pts2D[-1],  pts3D[-1], all_R, all_T)
                bestT = - bestR.dot(bestT)
            
        return bestR, bestT
    

    def Cosine(self, x, y):
        return x.dot(y) / (LA.norm(x) * LA.norm(y)) 
    
    
    def getCoefficient(self, C_ab, C_bc, C_ac, K_1, K_2):
        G4 = (K_1*K_2 - K_1 - K_2)**2 - 4*K_1*K_2*(C_bc**2)
        G3 = 4*(K_1*K_2 - K_1 - K_2)*K_2*(1 - K_1)*C_ab \
            + 4*K_1*C_bc*((K_1*K_2 - K_1 + K_2) * C_ac + 2*K_2*C_ab*C_bc)
        G2 = (2*K_2*(1-K_1)*C_ab)**2 \
            + 2*(K_1*K_2 - K_1 - K_2)*(K_1*K_2 + K_1 - K_2) \
            + 4*K_1*((K_1 - K_2)*(C_bc**2) + K_1*(1-K_2)*(C_ac**2) - 2*(1+K_1)*K_2*C_ab*C_ac*C_bc)
        G1 = 4*(K_1*K_2 + K_1 - K_2)*K_2*(1-K_1)*C_ab \
            + 4*K_1*((K_1*K_2 - K_1 + K_2)*C_ac*C_bc + 2*K_1*K_2*C_ab*(C_ac**2))
        G0 = (K_1*K_2 + K_1 - K_2)**2 \
            - 4*(K_1**2)*K_2*(C_ac**2)
        
        return np.array([G4, G3, G2, G1, G0])
    
    
    def getLength(self, x, R_ab, C_ab, C_bc, C_ac, K_1, K_2):
        m, p, q = 1 - K_1, 2*(K_1 * C_ac - x*C_bc), x**2 - K_1
        m_, p_, q_ = 1, 2*(-x*C_bc), x**2*(1-K_2) + 2*x*K_2*C_ab - K_2
                
        y = -(m_*q - m*q_) / (p*m_ - p_*m)
        
        a = np.sqrt((R_ab**2) / (1 + x**2 - 2*x*C_ab))
        b = x*a
        c = y*a
        
        return a, b, c
    
        
    def trilateration(self, p1, p2, p3, a, b, c):
        centers = []
        for a_, b_, c_ in zip(a, b, c):
            v_1, v_2 = (p2 - p1) / (LA.norm(p2 - p1)), (p3 - p1) / (LA.norm(p3 - p1))

            ix = v_1
            iz = np.cross(v_1, v_2) / LA.norm(np.cross(v_1, v_2))
            iy = np.cross(ix, iz) / LA.norm(np.cross(ix, iz))

            x2 = LA.norm(p2 - p1)
            x3, y3 = (p3-p1).dot(ix), (p3-p1).dot(iy)

            x_length = (a_**2 - b_**2 + x2**2) / (2*x2)
            y_length = (a_**2 - c_**2 + x3**2 + y3**2 - (2*x3*x_length)) / (2*y3)
            
            l = a_**2 - x_length**2 - y_length**2
            if l > 0: 
                z_length = l ** 0.5
                direction = x_length*ix + y_length*iy + z_length*iz
                direction_minus = x_length*ix + y_length*iy - z_length*iz

                centers.append(p1+direction_minus)
                centers.append(p1+direction)
            
        return np.array(centers)
    
    
    def getRT(self, centers, X, v):
        all_R, all_T = [], []
        for center in centers:
            for sign in [1, -1]:
                lada = LA.norm((X - center), axis=1) / LA.norm(v, axis=1) * sign
                R = (v.T * lada).dot(LA.inv(X.T - center.reshape(3,1)))
                all_R.append(R)
                all_T.append(center)
        return all_R, all_T
    
    
    def reproject(self, v_4, point3D_4, all_R, all_T):
        lowesrError = 100000
        finalR, finalT = all_R[0], all_T[0]
        for R, T in zip(all_R, all_T):
            point = R.dot(point3D_4 - T)
            error = LA.norm(v_4 - (point / point[-1])[:2])
            if(error < lowesrError):
                finalR, finalT = R, T
                lowesrError = error

        return finalR, finalT