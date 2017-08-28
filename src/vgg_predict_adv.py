from config_PASCAL_VC import *
from VGG_classifier import *
from scipy.misc import logsumexp

category = sys.argv[1]

print(category)

######### config #############
adv_file = os.path.join(adv_dir, 'adv_img_{}.pickle'.format(category))

with open(adv_file, 'rb') as fh:
    _, im_fool_ls = pickle.load(fh)

img_num = len(im_fool_ls)
print('Total image number for {}: {}'.format(category, img_num))

save_dir = os.path.join(root_dir, 'result_vgg')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_name = os.path.join(save_dir, 'VGG_predict_{}_adv.pickle'.format(category))

classifier = VGG_classifier(model_cache_folder)

##################### load images


pred_rst = [None for nn in range(img_num)]
for nn in range(img_num):
    if nn%100==0:
        print(nn, end=' ', flush=True)
    
    im = im_fool_ls[nn]
    
    pred_rst[nn] = classifier.predict_image(im)[0]
    
print()

pred_rst = np.array(pred_rst)
print(pred_rst.shape)

with open(save_name, 'wb') as fh:
    pickle.dump(pred_rst, fh)
    