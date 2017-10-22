from config_PASCAL_VC import *
from VGG_classifier import *
from scipy.misc import logsumexp

# category = sys.argv[1]
classifier = VGG_classifier(model_cache_folder_f, num_classes=len(all_categories))

for category in ['car']:
    print(category)

    ######### config #############
    # adv_file = os.path.join(Adv_dir3, 'adv_img_{}.pickle'.format(category))
    adv_file = '/mnt/1TB_SSD/qing/VC_adv/adv/adv_img_car.pickle'

    with open(adv_file, 'rb') as fh:
        _, im_fool_ls = pickle.load(fh)

    img_num = len(im_fool_ls)
    print('Total image number for {}: {}'.format(category, img_num))

    save_dir = os.path.join(root_dir, 'result_vgg_B')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save_name = os.path.join(save_dir, 'VGG_predict_{}_adv3.pickle'.format(category))
    save_name = os.path.join('/mnt/1TB_SSD/qing/VC_adv/result_vgg', 'VGG_predict_{}_adv.pickle'.format(category))
    ##################### load images


    pred_rst = [None for nn in range(img_num)]
    for nn in range(img_num):
        if nn%100==0:
            print(nn, end=' ', flush=True)

        im = im_fool_ls[nn]

        pred_rst[nn] = classifier.predict_image(im)[0]

    print('')

    pred_rst = np.array(pred_rst)
    print(pred_rst.shape)

    with open(save_name, 'wb') as fh:
        pickle.dump(pred_rst, fh)
    