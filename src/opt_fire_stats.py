from scipy.spatial.distance import cdist
from config_PASCAL_VC import *

def opt_fire_stats(category, centers, set_type):
    magic_thh_ls = [0.25]
    step = 0.05
    print('optimizing magic threshold for category {}, set_type {}, layer {}'.format(category, set_type, VC['layer']))
    if category in all_categories:
        filename = os.path.join(Feat['cache_dir'], 'cropped', 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
        
        with open(filename, 'rb') as fh:
            layer_feature = pickle.load(fh)
    
    elif category=='all':
        layer_feature = []
        for cc in all_categories:
            filename = os.path.join(Feat['cache_dir'], 'feat_{}_{}.pickle'.format(cc, set_type))
            with open(filename, 'rb') as fh:
                layer_feature += pickle.load(fh)
            
    else:
        sys.exit('error: unknown category {}'.format(category))
        
    
    N = len(layer_feature)
    # print('{0}: total number of instances {1}'.format(category, N))
    # print(layer_feature[0].shape)
    
    layer_feature_dist = []
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        layer_feature_dist.append(cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1))
        
    while step > 0.0005:
        magic_thh = magic_thh_ls[-1]
        layer_feature_b = [None for nn in range(N)]

        for nn in range(N):
            layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int)

        vc_fire_cnt = [None for nn in range(N)]
        vc_fire_empty = [None for nn in range(N)]
        for nn in range(N):
            vc_fire_cnt[nn] = np.mean(np.sum(layer_feature_b[nn], axis=2))
            vc_fire_empty[nn] = np.sum(np.sum(layer_feature_b[nn], axis=2)==0)/np.prod(layer_feature_b[nn].shape[0:2])


        print('Magic threshold {2}: {0}, {1}'.format(np.mean(vc_fire_cnt), np.mean(vc_fire_empty), magic_thh))
            
        if np.mean(vc_fire_empty) > 0.2:
            magic_thh_ls.append(np.around(magic_thh+step, decimals=3))
        else:
            del(magic_thh_ls[-1])
            step /= 2.0
            magic_thh_ls.append(np.around(magic_thh_ls[-1]+step, decimals=3))
            
            
    magic_thh = magic_thh_ls[-1]
    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int)

    vc_fire_cnt = [None for nn in range(N)]
    vc_fire_empty = [None for nn in range(N)]
    for nn in range(N):
        vc_fire_cnt[nn] = np.mean(np.sum(layer_feature_b[nn], axis=2))
        vc_fire_empty[nn] = np.sum(np.sum(layer_feature_b[nn], axis=2)==0)/np.prod(layer_feature_b[nn].shape[0:2])
        
    print('final rst: {}, {}, {}'.format(magic_thh, np.mean(vc_fire_cnt), np.mean(vc_fire_empty)))
    return(magic_thh)



if __name__=='__main__':
    with open(Dict['Dictionary'], 'rb') as fh:
        centers=pickle.load(fh)
        
        if len(centers)==3:
            centers = centers[1]
    
    set_type= 'train'
    per_category = True
    
    if per_category:
        rst_ls = []
        for category in all_categories:
            rst_ls.append(opt_fire_stats(category, centers, set_type))
            
    else:
        rst_ls = opt_fire_stats('all', centers, set_type) # scalar
        
    save_file = os.path.join(Model_dir,'magic_thh_{}_{}.pickle'.format(set_type, VC['layer']))
    if not per_category:
        save_file = os.path.join(Model_dir,'magic_thh_{}_all.pickle'.format(set_type))
        
    with open(save_file, 'wb') as fh:
        pickle.dump(rst_ls, fh)

