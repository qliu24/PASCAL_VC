import pickle
import numpy as np
import cv2
import os
from copy import *

class Data_loader:
    def __init__(self, filelist, scale_size, img_dir):
        self.scale_size = scale_size
        self.img_dir = img_dir
        # self.img_mean = np.float32([[[104., 117., 124.]]])
        
        with open(filelist, 'r') as fh:
            contents = fh.readlines()
            
        self.img_list = [cc.strip().split()[0] for cc in contents]
        self.label_list = [cc.strip().split()[1] for cc in contents]
        self.totalN = len(self.img_list)
        print('total number of images: {0}'.format(self.totalN))
        
        permuted = np.random.permutation(self.totalN)
        self.eval_idx = permuted[0:int(self.totalN/5)]  # %20 for evaluation
        self.train_idx = permuted[int(self.totalN/5):]  # %80 for training
        
    def get_batch(self, batch_size=128, set_type='train'):
        if set_type=='train':
            idx_s = np.random.choice(self.train_idx, size=[batch_size,], replace=False)
        else:
            idx_s = np.random.choice(self.eval_idx, size=[batch_size,], replace=False)
            
        img_s = np.zeros((batch_size, self.scale_size, self.scale_size, 3))
        label_s = np.ones((batch_size, ))*-1
        for ib,ii in enumerate(idx_s):
            fname = self.img_list[ii]
            assert(os.path.isfile(fname))
            img = cv2.imread(fname)
            # check scale
            assert(np.min(img.shape[0:2]) == self.scale_size)
            
            # center crop
            img = img.astype(np.float32)
            height,width=img.shape[0:2]
            if height>width:
                start_h=(height-self.scale_size)//2
                img=img[start_h:start_h+self.scale_size, :, :]
            else:
                start_w=(width-self.scale_size)//2
                img=img[:, start_w:start_w+self.scale_size, :]
                
            img_s[ib] = deepcopy(img)
            label_s[ib] = self.label_list[ii]
            
        return img_s, label_s