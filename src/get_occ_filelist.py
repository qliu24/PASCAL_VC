import scipy.io as sio
import h5py
from config_PASCAL_VC import *

file_root = '/mnt/1TB_SSD/dataset/PASCAL3D+_occ/occ_img_cropped'

for category in ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']:
    save_file = os.path.join(file_root, category+'_imagenet_occ.txt')
    
    file_dir = os.path.join(file_root, category+'LEVELONE')
    file_list = [ff.split('.')[0] for ff in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, ff))]
    with open(save_file,'w') as fh:
        for item in file_list:
            fh.write(item+'\n')

print('done')