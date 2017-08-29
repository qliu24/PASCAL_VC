from config_PASCAL_VC import *
from VGG_classifier import *
from scipy.misc import logsumexp

category = sys.argv[1]

print(category)

######### config #############
with_noise=True

img_dir = Dataset['img_dir'].format(category)
file_list = Dataset['test_list'].format(category)

save_dir = os.path.join(root_dir, 'result_vgg')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_name = os.path.join(save_dir, 'VGG_predict_{}_unvr.pickle'.format(category))

classifier = VGG_classifier(model_cache_folder)

noise_r_file = '/mnt/1TB_SSD/qing/PASCAL_adv/universal_r.pickle'
if with_noise:
    with open(noise_r_file, 'rb') as fh:
        unv_r=pickle.load(fh)
        
    assert(scale_size==unv_r.shape[0])
    assert(scale_size==unv_r.shape[1])
    assert(unv_r.shape[2]==3)

##################### load images
with open(file_list, 'r') as fh:
    content = fh.readlines()
    
img_list = [cc.strip() for cc in content]
img_num = len(img_list)
print('total number of images for {}: {}'.format(category, img_num))

pred_rst = [None for nn in range(img_num)]
for nn in range(img_num):
    if nn%100==0:
        print(nn, end=' ', flush=True)
        
    file_img = os.path.join(img_dir, '{0}.JPEG'.format(img_list[nn]))
    assert(os.path.isfile(file_img))
    im = cv2.imread(file_img)
    im = im.astype(np.float32)
    assert(scale_size == np.min(im.shape[0:2]))
    if with_noise:
        height,width=im.shape[0:2]
        if height>scale_size:
            start_h=(height-scale_size)//2
            im=im[start_h:start_h+scale_size, :, :]
        else:
            start_w=(width-scale_size)//2
            im=im[:, start_w:start_w+scale_size, :]
            
        im += unv_r
        
    
    pred_rst[nn] = classifier.predict_image(im)[0]
    
print('')
pred_rst = np.array(pred_rst)
print(pred_rst.shape)
with open(save_name, 'wb') as fh:
    pickle.dump(pred_rst, fh)
    