import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from datetime import datetime
import network as vgg
from config_PASCAL_VC import *
from scipy.misc import logsumexp

# category = sys.argv[1]
######### config #############
step_size = 5.0

img_mean = np.array([104., 117., 124.]).reshape(1,1,3)

save_dir = os.path.join(root_dir, 'adv_fgsm')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

##################### init VGG
def get_init_restorer():
    variables_to_restore = []
    # for var in slim.get_model_variables():
    for var in tf.global_variables():
        variables_to_restore.append(var)

    return tf.train.Saver(variables_to_restore)

# checkpoints_dir = os.path.join(model_cache_folder, 'checkpoints_vgg')
checkpoints_dir = os.path.join(model_cache_folder_f, 'checkpoints')
tf.logging.set_verbosity(tf.logging.INFO)
with tf.device('/cpu:0'):
    input_images = tf.placeholder(tf.float32, [1, None, None, 3])

vgg_var_scope = 'vgg_16'
with tf.variable_scope(vgg_var_scope, reuse=False):
    with slim.arg_scope(vgg.vgg_arg_scope(bn=False, is_training=False)):
        _, end_points = vgg.vgg_16(input_images, num_classes=len(all_categories), is_training=False)

restorer = get_init_restorer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init_op = tf.global_variables_initializer()
sess = tf.Session(config=config)
print(str(datetime.now()) + ': Start Init')
# restorer.restore(sess, os.path.join(checkpoints_dir, 'vgg_16.ckpt'))
restorer.restore(sess, os.path.join(checkpoints_dir, 'fine_tuned-8000'))
print(str(datetime.now()) + ': Finish Init')

grad_ts_ls = []
for ii in range(end_points['vgg_16/fc8/reduced'].get_shape().as_list()[1]):
    grad_ts_ls.append(tf.gradients(end_points['vgg_16/fc8/reduced'][0,ii], input_images)[0])
    

for category in all_categories:

    # cat_to_idx = dict()
    # cat_to_idx['aeroplane'] = 404
    # cat_to_idx['bicycle'] = 444
    # cat_to_idx['boat'] = 628
    # cat_to_idx['bottle'] = 440
    # cat_to_idx['bus'] = 874
    # cat_to_idx['car'] = 436
    # cat_to_idx['chair'] = 559
    # cat_to_idx['diningtable'] = 532
    # cat_to_idx['motorbike'] = 665
    # cat_to_idx['sofa'] = 831
    # cat_to_idx['train'] = 466
    # cat_to_idx['tvmonitor'] = 664
    target_ls = np.array(all_categories)[np.arange(len(all_categories))!= all_categories.index(category)]

    img_dir = Dataset['img_dir'].format(category)
    file_list = Dataset['test_list'].format(category)

    ##################### load images
    with open(file_list, 'r') as fh:
        content = fh.readlines()

    img_list = [cc.strip() for cc in content]
    img_num = len(img_list)
    print('total number of images for {}: {}'.format(category, img_num))

    r_ls = []
    im_fool_ls = []
    for nn in range(img_num):
        target = np.random.choice(target_ls)
        # target_idx = cat_to_idx[target]
        target_idx = all_categories.index(target)
        target_grad_ts = grad_ts_ls[target_idx]

        print('Fooling image {} to {}:'.format(nn, target))
        file_img = os.path.join(img_dir, '{0}.JPEG'.format(img_list[nn]))
        assert(os.path.isfile(file_img))
        im = cv2.imread(file_img)
        im = im.astype(np.float32)
        assert(scale_size == np.min(im.shape[0:2]))

        im_ori = np.copy(im)
        im -= img_mean
        im = im.reshape(np.concatenate([[1],im.shape]))

        out = sess.run(end_points['vgg_16/fc8/reduced'], feed_dict={input_images: im})[0]
        out_prob = np.exp(out-logsumexp(out))

        target_score = out[target_idx]
        target_prob = out_prob[target_idx]

        pred_score = np.max(out)
        pred_idx = np.argmax(out)
        pred_prob = out_prob[pred_idx]

        print("Before perturbation: predicted index {0}({1:.4f}), targeted index {2}({3:.4f})".format(pred_idx, pred_prob, target_idx, target_prob))

        sort_idx = np.argsort(-out)
        if sort_idx[0] == target_idx:
            pred_idx = sort_idx[1]
        else:
            pred_idx = sort_idx[0]

        pred_grad_ts = grad_ts_ls[pred_idx]
        grad = sess.run(target_grad_ts-pred_grad_ts, feed_dict={input_images: im})

        r = step_size*np.sign(grad)

        out = sess.run(end_points['vgg_16/fc8/reduced'], feed_dict={input_images: im+r})[0]
        out_prob = np.exp(out-logsumexp(out))

        target_prob = out_prob[target_idx]
        pred_idx = np.argmax(out)
        pred_prob = out_prob[pred_idx]

        print("After perturbation: predicted index {0}({1:.4f}), targeted index {2}({3:.4f})".format(pred_idx, pred_prob, target_idx, target_prob))

        im_fool = im+r
        im_fool = np.squeeze(im_fool)
        im_fool += img_mean
        im_fool = np.clip(im_fool, 0, 255)

        r_ls.append(np.copy(r))
        im_fool_ls.append(np.copy(im_fool))

    fooling_image_name = os.path.join(save_dir, 'adv_img_{}.pickle'.format(category))
    with open(fooling_image_name, 'wb') as fh:
        pickle.dump([r_ls, im_fool_ls], fh)
    