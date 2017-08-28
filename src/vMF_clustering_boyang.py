from config_PASCAL_VC import *
from vMFMM import *

cluster_num = 200
fname = '/export/home/bdeng4/features_untrained.npy'
feat_set = np.load(fname)

print('all feat_set')
print(feat_set.shape)

model = vMFMM(cluster_num, 'k++')
model.fit(feat_set, 300, max_it=1000)

savefile = '/export/home/qliu24/tmp/vMFMM/Dictionary_PASCAL3D+_pool4_all_5shots.pickle'
with open(savefile, 'wb') as fh:
    pickle.dump([model.p, model.mu, model.pi], fh, protocol=2)

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

