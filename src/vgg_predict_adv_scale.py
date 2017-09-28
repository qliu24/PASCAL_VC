from config_PASCAL_VC import *
from VGG_classifier import *
from scipy.misc import logsumexp
    
def vgg_predict(category, classifier, scale):
    print(category, scale)

    ######### config #############
    adv_file = os.path.join(Adv_dir, 'adv_img_{}.pickle'.format(category))

    with open(adv_file, 'rb') as fh:
        _, im_fool_ls = pickle.load(fh)

    img_num = len(im_fool_ls)
    print('Total image number for {}: {}'.format(category, img_num))

    save_dir = os.path.join(root_dir, 'result_vgg')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name = os.path.join(save_dir, 'VGG_predict_{}_adv_sc{}.pickle'.format(category, scale))

    ##################### load images


    pred_rst = [None for nn in range(img_num)]
    for nn in range(img_num):
        if nn%100==0:
            print(nn, end=' ', flush=True)

        im = im_fool_ls[nn]
        hh1,ww1 = im.shape[0:2]
        im = myresize(im, scale, 'short')
        hh2,ww2 = im.shape[0:2]
        if scale>scale_size:
            diff_h = hh2-hh1
            diff_w = ww2-ww1
            shift_h = np.random.choice(diff_h)
            shift_w = np.random.choice(diff_w)
            im = im[shift_h:shift_h+hh1, shift_w:shift_w+ww1]
        else:
            diff_h = hh1-hh2
            diff_w = ww1-ww2
            shift_h = np.random.choice(diff_h)
            shift_w = np.random.choice(diff_w)
            im = np.pad(im, ((shift_h, diff_h-shift_h),(shift_w, diff_w-shift_w),(0,0)), 'constant', constant_values=0)

        pred_rst[nn] = classifier.predict_image(im)[0]

    print('')

    pred_rst = np.array(pred_rst)
    print(pred_rst.shape)

    with open(save_name, 'wb') as fh:
        pickle.dump(pred_rst, fh)
    
    
if __name__=='__main__':
    classifier = VGG_classifier(model_cache_folder_f, num_classes=len(all_categories))
    for category in all_categories:
        for scale in [scale_size-64, scale_size-32, scale_size+32, scale_size+64]:
            vgg_predict(category, classifier, scale)