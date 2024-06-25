import sys
import numpy as np
import cv2 as cv
from src.homography import *   


if __name__ == '__main__':
    anc_img = cv.imread(sys.argv[1])
    tar_img = cv.imread(sys.argv[2])
    
    gt_correspondences = np.load(sys.argv[3])
    gt_anchor = gt_correspondences[0]
    gt_target = gt_correspondences[1]
    
    
    hom = Homography()
    points1, points2 = hom.get_correspondences(anc_img, tar_img)
    
    
    K = [4, 8, 20]
    iter = 1
    print("Direct Linear Transform without normalization")
    for k in K:
        error = 0
        for _ in range(iter):
            new_points1, new_points2 = hom.RANSAC(points1, points2, k, norm = False)
            hom.DLT(new_points1, new_points2, normalize = False)
            error += hom.compute_error(gt_anchor, gt_target)
        
        print("k = {:>3d}, average error = {:2.3f}".format(k, error / iter))
    new_points1, new_points2 = hom.RANSAC(points1, points2, "all", norm = False)
    hom.DLT(new_points1, new_points2, normalize = False)
    error = hom.compute_error(gt_anchor, gt_target)
    print("k = all inliers, error = {:2.3f}".format(error))
    
    
    print("\nDirect Linear Transform with normalization")
    for k in K:
        error = 0
        for _ in range(iter):
            new_points1, new_points2 = hom.RANSAC(points1, points2, k, norm = True)
            hom.DLT(new_points1, new_points2, normalize = True)
            error += hom.compute_error(gt_anchor, gt_target)
        
        print("k = {:>2d}, average error = {:2.3f}".format(k, error / iter))
    new_points1, new_points2 = hom.RANSAC(points1, points2, "all", norm = True)
    hom.DLT(new_points1, new_points2, normalize = True)
    error = hom.compute_error(gt_anchor, gt_target)
    print("k = all inliers, error = {:2.3f}".format(error))
    
    
    print("\nDifferent types of feature, k = 4")
    for type in ["SIFT", "ORB", "AKAZE"]:
        points1, points2 = hom.get_correspondences(anc_img, tar_img, detector = type)
        error = 0
        for _ in range(iter):
            new_points1, new_points2 = hom.RANSAC(points1, points2, 4, norm = True)
            hom.DLT(new_points1, new_points2, normalize = True)
            error += hom.compute_error(gt_anchor, gt_target)

        print("Type is {:>5}, average error = {:2.3f}".format(type, error / iter))