from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_PASCAL_VC import *

def extractLayerFeat_adv(category, extractor, scale_size=224):
    adv_file = os.path.join(Adv_dir2, 'adv_img_{}.pickle'.format(category))
    
    with open(adv_file, 'rb') as fh:
        _, im_fool_ls = pickle.load(fh)
        
    N = len(im_fool_ls)
    print('Total image number for {}: {}'.format(category, N))
    
    feat_set = [None for nn in range(N)]
    for nn in range(N):
        if nn%100==0:
            print(nn, end=' ', flush=True)

        
        img = im_fool_ls[nn]
        assert(np.min(img.shape[0:2]) == scale_size)
        layer_feature = extractor.extract_feature_image(img)[0]
        assert(featDim == layer_feature.shape[2])
        feat_set[nn] = layer_feature
        
    print('\n')
    
    file_cache_feat = os.path.join(Feat['cache_dir'], 'feat_{}_test_adv2.pickle'.format(category))
    with open(file_cache_feat, 'wb') as fh:
        pickle.dump(feat_set, fh)
        
            
if __name__=='__main__':
    extractor = FeatureExtractor(cache_folder=model_cache_folder_f, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
    for category in all_categories:
        extractLayerFeat_adv(category, extractor)