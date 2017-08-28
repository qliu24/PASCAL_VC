from config_PASCAL_VC import *
from scipy.spatial.distance import cdist
from cls_factorizable_funcs import *


def eval_VCclassifier(category):
    magic_thh=0.45
    K = 4
    total_models = len(all_categories)

    print('############################')
    print('class : {}'.format(category))

    # load test feat
    test_feat = os.path.join(Feat['cache_dir'], 'feat_{}_test.pickle'.format(category))
    with open(test_feat,'rb') as fh:
        layer_feature = pickle.load(fh)

    N = len(layer_feature)
    print('Total number of test samples: {}'.format(N))

    # convert to 0-1 VC encoding

    with open(Dict['Dictionary'], 'rb') as fh:
        _, centers, _ = pickle.load(fh)

    r_set = [None for nn in range(N)]
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)

    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int)

    fire_total = [np.sum(lfb) for lfb in layer_feature_b]

    rst_scores = np.zeros((N, total_models))
    rst_scores_norm = np.zeros((N, total_models))
    for model_idx in range(total_models):
        print('model {}:'.format(model_idx), end=' ', flush=True)
        model_file = os.path.join(Model_dir, 'mmodel_{}_K{}_notrain.pickle'.format(all_categories[model_idx], K))
        assert(os.path.exists(model_file))
        with open(model_file,'rb') as fh:
            all_weights, _ = pickle.load(fh)
            
        assert(len(all_weights)==K)
        all_logZs = []
        for kk in range(K):
            # logZs.append(np.sum(np.log(1+np.exp(weights[kk]))))
            all_logZs.append(np.log(1+np.exp(all_weights[kk])))

        for nn in range(N):
            if nn%100==0:
                print(nn,end=' ', flush=True)
                
            rst_scores[nn,model_idx] = comptScoresM(layer_feature_b[nn], all_weights, all_logZs)
            rst_scores_norm[nn,model_idx] = rst_scores[nn,model_idx]/fire_total[nn]

        print('')


    rst_file = os.path.join(Result_dir, 'scores_{}.pickle'.format(category))
    with open(rst_file, 'wb') as fh:
        pickle.dump([rst_scores, rst_scores_norm], fh)

    rst = np.argmax(rst_scores_norm, axis=1)
    accu = np.sum(rst == all_categories.index(category))
    accu = accu/N
    print('accuracy is: %4.4f'% accu)
    
    
if __name__=='__main__':
    for category in all_categories:
        eval_VCclassifier(category)