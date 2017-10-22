from config_PASCAL_VC import *
from vMFMM import *

cluster_num = VC['num']
    
file_num = 12

feat_set = np.zeros((featDim, 0))
# loc_set = np.zeros((5, 0), dtype='int')
for ii in range(file_num):
    print('loading file {0}'.format(ii))
    fname = Dict['cache_path']+'{}.pickle'.format(ii)
    with open(fname, 'rb') as fh:
        res, _ , _= pickle.load(fh)
        feat_set = np.column_stack((feat_set, res))
        # loc_set = np.column_stack((loc_set, iloc.astype('int')))

print('all feat_set')
feat_set = feat_set.T
print(feat_set.shape)

model = vMFMM(cluster_num, 'k++', tmp_dir='/home/qing/tmp/vMFMM/PASCAL3D+')
model.fit(feat_set, 30, max_it=150)
    
with open(Dict['Dictionary'], 'wb') as fh:
    pickle.dump(model.mu, fh)

bins = 4
per_bin = cluster_num//bins+1
for bb in range(bins):
    with open(Dict['Dictionary'].replace('.pickle','_p{}.pickle'.format(bb)), 'wb') as fh:
        pickle.dump(model.p[:,bb*per_bin:(bb+1)*per_bin], fh)

############## save examples ###################
# with open(Dict['file_list'], 'r') as fh:
#     image_path = [ff.strip() for ff in fh.readlines()]

# num = 50
# print('save top {0} images for each cluster'.format(num))
# example = [None for vc_i in range(cluster_num)]
# for vc_i in range(cluster_num):
#     patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
#     sort_idx = np.argsort(-model.p[:,vc_i])[0:num]
#     for idx in range(num):
#         iloc = loc_set[:,sort_idx[idx]]
#         img = cv2.imread(os.path.join(Dict['file_dir'], image_path[iloc[0]]))
#         img = myresize(img, scale_size, 'short')
        
#         patch = img[iloc[1]:iloc[3], iloc[2]:iloc[4], :]
#         patch_set[:,idx] = patch.flatten()
        
#     example[vc_i] = np.copy(patch_set)
#     if vc_i%10 == 0:
#         print(vc_i)
        
# save_path2 = Dict['Dictionary'].replace('.pickle','_example.pickle')
# with open(save_path2, 'wb') as fh:
#     pickle.dump(example, fh)
