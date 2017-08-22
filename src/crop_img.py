from config_PASCAL_VC import *
import scipy.io as sio

for category in all_categories:
    print(category)
    # if category != 'car':
    #     continue
    
    filelist = Dataset['test_list'].format(category)

    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    img_list = [cc.strip() for cc in contents]

    dir_img = Dataset['img_dir_org'].format(category)
    dir_anno = Dataset['anno_dir'].format(category)
    dir_save = Dataset['img_dir'].format(category)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    N = len(img_list)
    for nn in range(N):
        if nn%100==0:
            print(nn)

        img_file = os.path.join(dir_img, '{}.JPEG'.format(img_list[nn]))
        # print(img_file)
        img=cv2.imread(img_file)
        # plt.imshow(img[:,:,[2,1,0]])
        # plt.show()

        height, width = img.shape[0:2]

        anno_file = os.path.join(dir_anno, '{}.mat'.format(img_list[nn]))
        assert(os.path.isfile(anno_file))
        mat_contents = sio.loadmat(anno_file)
        record = mat_contents['record']
        objects = record['objects']
        bbox = objects[0,0]['bbox'][0,0][0]
        bbox = [max(math.ceil(bbox[0]), 1), max(math.ceil(bbox[1]), 1), \
                min(math.floor(bbox[2]), width), min(math.floor(bbox[3]), height)]
        patch = img[bbox[1]-1: bbox[3], bbox[0]-1: bbox[2], :]
        # patch = cv2.resize(patch, (scale_size, scale_size))
        try:
            patch = myresize(patch, scale_size, 'short')
        except:
            print(nn)
            continue
        # plt.imshow(patch[:,:,[2,1,0]])
        # plt.show()
        save_file = os.path.join(dir_save, '{}.JPEG'.format(img_list[nn]))
        cv2.imwrite(save_file, patch)