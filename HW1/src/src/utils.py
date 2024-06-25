import numpy as np
from .homography import *


def wrapping(anc_img, anc_corners):
    '''
        backward warping
        anc_corners, shape = [4, 2], in upleft, up right, botleft, botright order
    '''
    rec_img = np.zeros_like(anc_img)
    H, W, _ = rec_img.shape
    rec_corners = np.array([[0, 0], [0, W-1], [H-1, 0], [H-1, W-1]])
    
    homography = Homography()
    homography.DLT(rec_corners, anc_corners, normalize = True)
    
    index_pts = np.array([[h, w] for h in range(H) for w in range(W)])
    trans_pt = homography.transform(index_pts)
    rec_img = interpolation(anc_img, trans_pt)
            
    return rec_img
    
    
def interpolation(anc_img, pt):
    h, w = pt[:, 0], pt[:, 1]
    h0, w0 = np.floor(h).astype(int), np.floor(w).astype(int)
    h1, w1 = h0 + 1, w0 + 1
    
    a, b = anc_img[h0, w0], anc_img[h0, w1]
    c, d = anc_img[h1, w1], anc_img[h1, w0]
    
    h1_h, h_h0 = h1 - h, h - h0
    w1_w, w_w0 = w1 - w, w - w0
    wa = np.expand_dims((h1_h * w1_w), axis = -1)
    wb = np.expand_dims((h1_h * w_w0), axis = -1)
    wc = np.expand_dims((h_h0 * w_w0), axis = -1)
    wd = np.expand_dims((h_h0 * w1_w), axis = -1)
    
    x = (wa*a + wb*b + wc*c + wd*d).astype(np.uint8).reshape(anc_img.shape)
    
    return x