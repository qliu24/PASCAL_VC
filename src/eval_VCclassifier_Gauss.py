from config_PASCAL_VC import *
from scipy.spatial.distance import cdist
from cls_factorizable_funcs import *


def eval_VCclassifier_Gauss(category, kappa=5):
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

    # transfer from raw feature space to distance space
    with open(Dict['Dictionary'], 'rb') as fh:
        _, centers, _ = pickle.load(fh)

    r_set = [None for nn in range(N)]
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)

    # transfer from distance space to firing rate space
    layer_feature_fr = [None for nn in range(N)]
    for nn in range(N):
        hnn,wnn = r_set[nn].shape[0:2]
        if hnn>14:
            marg = (hnn-14)//2
            r_set[nn] = r_set[nn][marg:marg+14, :, :]
        elif wnn>14:
            marg = (wnn-14)//2
            r_set[nn] = r_set[nn][:, marg:marg+14, :]
            
        layer_feature_fr[nn] = np.exp(-kappa*r_set[nn])
        

    rst_scores = np.zeros((N, total_models))
    for model_idx in range(total_models):
        print('model {}:'.format(model_idx), end=' ', flush=True)
        model_file = os.path.join(Model_dir, 'mmodel_{}_K{}_notrain_gauss.pickle'.format(all_categories[model_idx], K))
        assert(os.path.exists(model_file))
        with open(model_file,'rb') as fh:
            all_mus, all_sigmas, _ = pickle.load(fh)
            
        assert(len(all_mus)==K)
        term1s = [np.sum(-np.log(sigma)) for sigma in all_sigmas]

        for nn in range(N):
            if nn%100==0:
                print(nn,end=' ', flush=True)
                
            rst_scores[nn,model_idx] = comptGaussScoresM(layer_feature_fr[nn], all_mus, all_sigmas, term1s)

        print('')


    # rst_file = os.path.join(Result_dir, 'scores_{}_occ.pickle'.format(category))
    # with open(rst_file, 'wb') as fh:
    #     pickle.dump([rst_scores, rst_scores_norm], fh)

    rst = np.argmax(rst_scores, axis=1)
    accu = np.sum(rst == all_categories.index(category))
    accu = accu/N
    print('accuracy is: %4.4f'% accu)
    
    
if __name__=='__main__':
    for category in all_categories:
        eval_VCclassifier_Gauss(category)