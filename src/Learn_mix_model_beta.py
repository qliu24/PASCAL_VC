from config_PASCAL_VC import *
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from scipy.stats import beta

def learn_mix_model_beta(categroy, K=4, kappa=5):
    
    with open(Dict['Dictionary'], 'rb') as fh:
        _, centers, _ = pickle.load(fh)
    
    sim_fname = os.path.join(Feat['cache_dir'],'simmat','simmat_mthrh045_{}.pickle'.format(category))
    feat_fname = os.path.join(Feat['cache_dir'], 'feat_{}_train.pickle'.format(category))
    savename = os.path.join(root_dir,'mix_model','mmodel_{}_K{}_notrain_beta.pickle'.format(category, K))

    # Spectral clustering based on the similarity matrix
    with open(sim_fname, 'rb') as fh:
        mat_dis1, _ = pickle.load(fh)

    mat_dis = mat_dis1
    N = mat_dis.shape[0]
    print('total number of instances for obj {}: {}'.format(categroy, N))

    mat_full = mat_dis + mat_dis.T - np.ones((N,N))
    np.fill_diagonal(mat_full, 0)
    
    W_mat = 1. - mat_full
    print('W_mat stats: {}, {}'.format(np.mean(W_mat), np.std(W_mat)))

    K1 = 2
    cls_solver = SpectralClustering(n_clusters=K1,affinity='precomputed', random_state=666)
    lb = cls_solver.fit_predict(W_mat)

    K2=2
    idx2 = []
    W_mat2 = []
    lb2 = []
    for k in range(K1):
        idx2.append(np.where(lb==k)[0])
        W_mat2.append(W_mat[np.ix_(idx2[k],idx2[k])])
        print('W_mat_i stats: {}, {}'.format(np.mean(W_mat2[k]), np.std(W_mat2[k])))

        cls_solver = SpectralClustering(n_clusters=K2,affinity='precomputed', random_state=666)
        lb2.append(cls_solver.fit_predict(W_mat2[k]))

    rst_lbs1 = np.ones(len(idx2[0]))*-1
    rst_lbs1[np.where(lb2[0]==0)[0]] = 0
    rst_lbs1[np.where(lb2[0]==1)[0]] = 1
    rst_lbs2 = np.ones(len(idx2[1]))*-1
    rst_lbs2[np.where(lb2[1]==0)[0]] = 2
    rst_lbs2[np.where(lb2[1]==1)[0]] = 3
    rst_lbs = np.ones(N)*-1
    rst_lbs[idx2[0]] = rst_lbs1
    rst_lbs[idx2[1]] = rst_lbs2
    rst_lbs = rst_lbs.astype('int')
    
    del(mat_dis)
    
    for kk in range(K):
        print('cluster {} has {} samples'.format(kk, np.sum(rst_lbs==kk)))

    # Load the feature vector and compute VC encoding
    with open(feat_fname, 'rb') as fh:
        layer_feature = pickle.load(fh)

    assert(N == len(layer_feature))

    r_set = [None for nn in range(N)]
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)

    # transfer from distance space to firing rate space, center crop
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

    del(layer_feature)
    del(r_set)

    all_train = [[] for kk in range(K)]
    for nn in range(N):
        if nn%100==0:
            print(nn, end=' ', flush=True)

        all_train[rst_lbs[nn]].append(layer_feature_fr[nn].ravel())

    print('')

    
    all_alphas = [None for kk in range(K)]
    all_betas = [None for kk in range(K)]
    all_N = [0 for kk in range(K)]
    for kk in range(K):
        data_kk = np.array(all_train[kk])
        all_alphas[kk] = np.zeros(data_kk.shape[1])
        all_betas[kk] = np.zeros(data_kk.shape[1])
        for dd in range(data_kk.shape[1]):
            all_alphas[kk][dd], all_betas[kk][dd],_,_ = beta.fit(data_kk[:,dd])
            
        all_N[kk] = data_kk.shape[0]
        
    assert(N == np.sum(all_N))
    all_priors = np.array(all_N)/N

    with open(savename, 'wb') as fh:
        pickle.dump([all_alphas, all_betas, all_priors], fh)
            

if __name__=='__main__':
    for category in all_categories:
        learn_mix_model_beta(category)
