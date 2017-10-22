from config_PASCAL_VC import *
from vMFMM import *

cluster_num = VC['num']
file_num = 12

loc_set = np.zeros((5, 0), dtype='int')
for ii in range(file_num):
    print('loading file {0}'.format(ii))
    fname = Dict['cache_path']+'{}.pickle'.format(ii)
    with open(fname, 'rb') as fh:
        _, iloc, image_path = pickle.load(fh)
        loc_set = np.column_stack((loc_set, iloc.astype('int')))
        
print(loc_set.shape)

bins = 2
per_bin = cluster_num//bins+1

model_p = np.zeros((loc_set.shape[1], 0))
for bb in range(bins):
    with open(Dict['Dictionary'].replace('.pickle','_p{}.pickle'.format(bb)), 'rb') as fh:
        p_curr = pickle.load(fh)
        model_p = np.column_stack((model_p, p_curr))
        
print(model_p.shape)

############## save examples ###################

num = 50
print('save top {0} images for each cluster'.format(num))
example = [None for vc_i in range(cluster_num)]
for vc_i in range(cluster_num):
    patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
    sort_idx = np.argsort(-model_p[:,vc_i])[0:num]
    for idx in range(num):
        iloc = loc_set[:,sort_idx[idx]]
        img = cv2.imread(image_path[iloc[0]])
        assert(np.min(img.shape[0:2])==224)
        # img = myresize(img, scale_size, 'short')
        
        patch = img[iloc[1]:iloc[3], iloc[2]:iloc[4], :]
        patch_set[:,idx] = patch.flatten().astype('uint8')
        
    example[vc_i] = np.copy(patch_set)
    if vc_i%10 == 0:
        print(vc_i)
        
save_path2 = Dict['Dictionary'].replace('.pickle','_example.pickle')
with open(save_path2, 'wb') as fh:
    pickle.dump(example, fh)

