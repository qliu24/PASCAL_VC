import scipy.io as sio
import h5py
from config_PASCAL_VC import *

file_root = '/mnt/1TB_SSD/dataset/PASCAL3D+_occ/occ'
save_root = '/mnt/1TB_SSD/dataset/PASCAL3D+_occ/occ_img_cropped'
level = 'LEVELNINE'
for category in ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']:
        
    file_dir = os.path.join(file_root, category+level)
    file_list = [ff for ff in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, ff))]
    save_dir = os.path.join(save_root, category+level)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for testfile in file_list:
        f = h5py.File(os.path.join(file_dir, testfile))
        img = np.array(f.get('record').get('img')).T
        hh,ww=img.shape[0:2]

        bbox_ref = f.get('record').get('annotation').get('objects').get('bbox')
        if bbox_ref.dtype=='float64':
            bbox = np.array(bbox_ref).squeeze().astype(int)
        else:
            bbox = np.array(f[bbox_ref[int(testfile[-5])-1][0]]).squeeze().astype(int)

        col1, row1, col2, row2 = bbox
        col1 = max(col1, 1)
        row1 = max(row1, 1)
        col2 = min(col2, ww)
        row2 = min(row2, hh)

        patch=img[row1-1:row2, col1-1:col2]

        save_file = os.path.join(save_dir, testfile.replace('.mat','.JPEG'))
        cv2.imwrite(save_file, patch[:,:,[2,1,0]])

print('done')