import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from datetime import datetime
import network as vgg
from config_PASCAL_VC import *
from scipy.misc import logsumexp

category = sys.argv[1]

print(category)

######### config #############

img_mean = np.array([104., 117., 124.]).reshape(1,1,3)

img_dir = Dataset['img_dir'].format(category)
file_list = Dataset['test_list'].format(category)

save_dir = os.path.join(root_dir, 'result_vgg')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_name = os.path.join(save_dir, 'VGG_predict_{}.pickle'.format(category))
##################### init VGG
def get_init_restorer():
    variables_to_restore = []
    # for var in slim.get_model_variables():
    for var in tf.global_variables():
        variables_to_restore.append(var)
    
    return tf.train.Saver(variables_to_restore)


checkpoints_dir = os.path.join(model_cache_folder, 'checkpoints_vgg')
tf.logging.set_verbosity(tf.logging.INFO)
with tf.device('/cpu:0'):
    input_images = tf.placeholder(tf.float32, [1, None, None, 3])

vgg_var_scope = 'vgg_16'
with tf.variable_scope(vgg_var_scope, reuse=False):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg.vgg_16(input_images)
        
restorer = get_init_restorer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init_op = tf.global_variables_initializer()
sess = tf.Session(config=config)
print(str(datetime.now()) + ': Start Init')
restorer.restore(sess, os.path.join(checkpoints_dir, 'vgg_16.ckpt'))
print(str(datetime.now()) + ': Finish Init')

##################### load images
with open(file_list, 'r') as fh:
    content = fh.readlines()
    
img_list = [cc.strip() for cc in content]
img_num = len(img_list)
print('total number of images for {}: {}'.format(category, img_num))

pred_idx_ls=[]
for nn in range(img_num):
    if nn%100==0:
        print(nn, end=' ', flush=True)
    
    file_img = os.path.join(img_dir, '{0}.JPEG'.format(img_list[nn]))
    assert(os.path.isfile(file_img))
    im = cv2.imread(file_img)
    im = im.astype(np.float32)
    assert(scale_size == np.min(im.shape[0:2]))
    
    im -= img_mean
    im = im.reshape(np.concatenate([[1],im.shape]))
    
    out = sess.run(end_points['vgg_16/fc8/reduced'], feed_dict={input_images: im})[0]
    pred_idx = np.argmax(out)
    pred_idx_ls.append(pred_idx)
    
    
with open(save_name, 'wb') as fh:
    pickle.dump(pred_idx_ls, fh)
    