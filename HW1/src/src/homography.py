import numpy as np
import numpy.linalg as LA
import cv2 as cv


class Homography():
    def __init__(self):
        self.H = None
        
        
    def get_correspondences(self, anc_img, tar_img, detector = 'SIFT'):
        '''
        Input:
            img1: numpy array of the first image
            img2: numpy array of the second image

        Return:
            points1: numpy array [N, 2], N is the number of correspondences
            points2: numpy array [N, 2], N is the number of correspondences
        '''
        if detector == 'SIFT': detector = cv.SIFT_create()
        elif detector == 'ORB': detector = cv.ORB_create()
        elif detector == 'AKAZE': detector = cv.AKAZE_create()
 
        kp1, des1 = detector.detectAndCompute(anc_img, None)
        kp2, des2 = detector.detectAndCompute(tar_img, None)

        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k = 2)
        good_matches = []
        
        for m, n in matches:
            if m.distance < 0.68 * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

        return points1, points2


    def DLT(self, anchor_pts, target_pts, normalize = False):        
        if normalize:
            anchor_pts, anc_T = self.normalize(anchor_pts)
            target_pts, tar_T = self.normalize(target_pts)
        
        A = []
        num_pts = anchor_pts.shape[0]
        for i in range(num_pts):
            x, y = anchor_pts[i, 0], anchor_pts[i, 1]
            u, v = target_pts[i, 0], target_pts[i, 1]
            A.append([0, 0, 0, -x, -y, -1,  v*x,  v*y,  v])
            A.append([x, y, 1,  0,  0,  0, -u*x, -u*y, -u])
            
        U, S, Vh = LA.svd(np.array(A))
        H = Vh[-1, :].reshape((3, 3))

        if normalize: 
            H = LA.inv(tar_T).dot(H).dot(anc_T)
        
        self.H = H


    def RANSAC(self, keypoints1, keypoints2, k, norm, n_iters = 1500, threshold = 0.5):
        N = keypoints1.shape[0]
        n_samples = int(0.2 * N) if k == "all" else k 
        index = np.arange(N)

        max_inliers = np.zeros(N, dtype = bool)
        n_inliers = 0
        best_pts = None
        # # RANSAC iteration start
        for _ in range(n_iters):
            # 1. Select random set of matches
            np.random.shuffle(index)
            samples_index = index[:n_samples]
            anchor_pts = keypoints1[samples_index]
            target_pts = keypoints2[samples_index]
            
            # 2. Compute homography transformation matrix
            self.DLT(anchor_pts, target_pts, norm)
            
            # 3. Compute inliers via Euclidean distance
            trans = self.transform(keypoints1)
            
            error = ((trans - keypoints2) ** 2).sum(axis = 1) ** 0.5
            inliers = error < threshold
            
            # 4. Keep the largest set of inliers
            if np.sum(inliers) > n_inliers: 
                n_inliers = np.sum(inliers)
                max_inliers = inliers
                
                if k == "all": 
                    best_pts = keypoints1[max_inliers], keypoints2[max_inliers]
                else:
                    best_pts = anchor_pts, target_pts

        return best_pts


    def transform(self, anchor_pts):
        N = anchor_pts.shape[0]
        
        anchor = np.concatenate((anchor_pts, np.ones((N, 1))), axis = 1) 
        trans = anchor.dot(self.H.T)
        trans /= (np.expand_dims(trans[:, -1], axis = -1) + 1e-12)
        
        return trans[:, :-1]
    
    
    def compute_error(self, anchor_pts, target_pts):
        N = anchor_pts.shape[0]
        trans = self.transform(anchor_pts)
        error = (((trans - target_pts) ** 2).sum(axis = 1) ** 0.5).sum() / N
        
        return error
        

    def normalize(self, pts):
        N = pts.shape[0]
        mean = pts.mean(axis = 0)
        s = (((pts - mean) ** 2).sum() / (2 * N)) ** 0.5
        
        T = np.array([[1/s,     0,  -mean[0]/s],
                      [  0,   1/s,  -mean[1]/s],
                      [  0,     0,           1]])
        
        pts = np.concatenate((pts, np.ones((N, 1))), axis = 1) 
        nor_pts = pts.dot(T.T)
        
        return nor_pts[:, :-1], T
    

    def slope_detect(self, points1, points2):
        points2_t = points2.copy()
        offset = 2000
        threshold = 2.0
        points2_t [:, 1] += offset
        
        m = (points2_t [:, 1] - points1[:, 1]) / (points2_t[:, 0] - points1[:, 0])
        m_inlier = abs(m - m.mean()) < threshold
        
        return points1[m_inlier], points2[m_inlier]