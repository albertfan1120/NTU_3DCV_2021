import sys
import cv2 as cv
import numpy as np
from src.utils import *


if __name__ == '__main__':
    incline_img = cv.imread(sys.argv[1])
    
    corners = np.array([[502, 599], [642, 1398], [1529, 70], [1890, 1241]])
    rec_img = wrapping(incline_img, corners)
    
    cv.imwrite('images/rectified.jpg', rec_img)
    cv.imshow('rec_img', cv.resize(rec_img, (rec_img.shape[1]//3, rec_img.shape[0]//3)))
    cv.waitKey(0)