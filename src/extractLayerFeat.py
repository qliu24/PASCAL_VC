from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_PASCAL_VC import *

def extractLayerFeat(category, extractor, scale_size=224, set_type='train', center_crop=False):
    # img_dir = Dataset['occ_img_dir'].format(category,'NINE')
    img_dir = Dataset['img_dir'].format(category)
    
    filelist = Dataset['{}_list'.format(set_type)].format(category)
    with open(filelist, 'r') as fh:
        contents = fh.readlines()
        
    img_list = [cc.strip() for cc in contents]
    N = len(img_list)
    print('Total image number for {} set of {}: {}'.format(set_type, category, N))
    
    if category == 'motorbike' and set_type=='train':
        del(img_list[19])
        N = len(img_list)
        print('Total image number for {} set of {}: {}'.format(set_type, category, N))
    
    feat_set = [None for nn in range(N)]
    for nn in range(N):
        if nn%100==0:
            print(nn, end=' ', flush=True)

        img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
        img = cv2.imread(img_file)
        img = myresize(img, scale_size, 'short')
        assert(np.min(img.shape[0:2]) == scale_size)
        if center_crop:
            img = img.astype(np.float32)
            height,width=img.shape[0:2]
            if height>scale_size:
                start_h=(height-scale_size)//2
                img=img[start_h:start_h+scale_size, :, :]
            else:
                start_w=(width-scale_size)//2
                img=img[:, start_w:start_w+scale_size, :]
            
        layer_feature = extractor.extract_feature_image(img)[0]
        assert(featDim == layer_feature.shape[2])
        feat_set[nn] = layer_feature
        
    print('\n')
        
    file_cache_feat = os.path.join(Feat['cache_dir'], 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
    with open(file_cache_feat, 'wb') as fh:
        pickle.dump(feat_set, fh)
        
            
if __name__=='__main__':
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
    for category in all_categories:
        extractLayerFeat(category, extractor, set_type='test')