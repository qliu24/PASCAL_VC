from scipy.spatial.distance import cdist
import scipy.io as sio
from FeatureExtractor import *
from config_PASCAL_VC import *
import h5py

def extractLayerFeat_whole(category, extractor, set_type='train'):
    # img_dir = Dataset['occ_img_dir'].format(category,'NINE')
    if set_type[0:3] == 'occ':
        set_type,occlevel = set_type.split('_')
        img_dir = Dataset['occ_img_dir'].format(category, occlevel)
        anno_dir = Dataset['anno_dir'].format(category)
        filelist = Dataset['{}_list'.format(set_type)].format(category)
        with open(filelist, 'r') as fh:
            contents = fh.readlines()
            
        img_list = [cc.strip()[0:-2] for cc in contents if cc != '\n']
        idx_list = [cc.strip()[-1] for cc in contents if cc != '\n']
        N = len(img_list)
        print('Total image number for {} set of {}: {}'.format(set_type, category, N))
        
    else:
        img_dir = Dataset['img_dir_org'].format(category)
        anno_dir = Dataset['anno_dir'].format(category)
        filelist = Dataset['{}_list'.format(set_type)].format(category)
        with open(filelist, 'r') as fh:
            contents = fh.readlines()

        img_list = [cc.strip() for cc in contents]
        # idx_list = [1 for cc in contents]
        
        N = len(img_list)
        print('Total image number for {} set of {}: {}'.format(set_type, category, N))
    
    feat_set = [None for nn in range(N)]
    resize_ratio_ls = np.zeros(N)
    for nn in range(N):
        if nn%100==0:
            print(nn, end=' ', flush=True)
            
        if set_type == 'occ':
            matfile = os.path.join(img_dir, '{}_{}.mat'.format(img_list[nn], idx_list[nn]))
            f = h5py.File(matfile)
            img = np.array(f['record']['img']).T
            img = img[:,:,::-1]  # RGB to BGR
            
        else:
            img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
            try:
                assert(os.path.exists(img_file))
            except:
                print('file not exist: {}'.format(img_file))
                continue

            img = cv2.imread(img_file)
            
        img_h, img_w = img.shape[0:2]
        
        anno_file = os.path.join(anno_dir, '{}.mat'.format(img_list[nn]))
        try:
            assert(os.path.exists(anno_file))
        except:
            print('file not exist: {}'.format(anno_file))
            continue

        matcontent = sio.loadmat(anno_file)
        
        if set_type[0:3] == 'occ':
            bbox_value = matcontent['record']['objects'][0,0][0,int(idx_list[nn])-1]['bbox'][0]
            bbox_value = [max(math.ceil(bbox_value[0]), 1), max(math.ceil(bbox_value[1]), 1), \
                        min(math.floor(bbox_value[2]), img_w), min(math.floor(bbox_value[3]), img_h)]
        else:
            bbox_area = -np.inf
            bbox_value = None
            for bbi in range(matcontent['record']['objects'][0,0].shape[1]):
                bv_curr = matcontent['record']['objects'][0,0][0,bbi]['bbox'][0]
                bv_curr = [max(math.ceil(bv_curr[0]), 1), max(math.ceil(bv_curr[1]), 1), \
                        min(math.floor(bv_curr[2]), img_w), min(math.floor(bv_curr[3]), img_h)]
                bv_area_curr = (bv_curr[2]-bv_curr[0]+1)*(bv_curr[3]-bv_curr[1]+1)
                if bv_area_curr > bbox_area:
                    bbox_area = bv_area_curr
                    bbox_value = bv_curr
            
        bbox_height = bbox_value[3]-bbox_value[1]+1
        bbox_width = bbox_value[2]-bbox_value[0]+1
        resize_ratio = scale_size/np.min((bbox_height,bbox_width))
        resize_ratio_ls[nn] = resize_ratio
        img_resized = cv2.resize(img,None,fx=resize_ratio, fy=resize_ratio)
        
        layer_feature = extractor.extract_feature_image(img_resized)[0]
        assert(featDim == layer_feature.shape[2])
        feat_set[nn] = layer_feature
        
    print('\n')
    
    file_cache_feat = os.path.join(Feat['cache_dir'], 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
    if set_type == 'occ':
        file_cache_feat = os.path.join(Feat['cache_dir'], \
                                       'feat_{}_{}_{}.pickle'.format(category, '{}_{}'.format(set_type,occlevel), VC['layer']))
        
    with open(file_cache_feat, 'wb') as fh:
        pickle.dump(feat_set, fh)
        
    file_cache_rr = os.path.join(Feat['cache_dir'], 'feat_{}_{}_rr.pickle'.format(category, set_type))
    with open(file_cache_rr, 'wb') as fh:
        pickle.dump(resize_ratio_ls, fh)
        
            
if __name__=='__main__':
    extractor = FeatureExtractor(cache_folder=model_cache_folder_f, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
    for category in all_categories:
        extractLayerFeat_whole(category, extractor, set_type='train')