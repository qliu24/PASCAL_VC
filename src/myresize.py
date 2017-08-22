import numpy as np
import cv2
import sys

def myresize(img, dim, tp):
    H, W = img.shape[0:2]
    if tp == 'short':
        if H <= W:
            ratio = dim/float(H)
        else:
            ratio = dim/float(W)
        
    elif tp == 'long':
        if H <= W:
            ratio = dim/float(W)
        else:
            ratio = dim/float(H)
    else:
        sys.exit('myresize: wrong type!')
            
    return cv2.resize(img, (0,0), fx=ratio, fy=ratio)