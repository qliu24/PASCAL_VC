import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.slim as slim
from datetime import datetime
import network as vgg
import os

def get_init_restorer():
    variables_to_restore = []
    # for var in slim.get_model_variables():
    for var in tf.global_variables():
        variables_to_restore.append(var)
    
    return tf.train.Saver(variables_to_restore)

class VGG_classifier:
    def __init__(self, cache_folder, num_classes=1000, batch_size=1):
        # params
        self.batch_size = batch_size
        self.scale_size = vgg.vgg_16.default_image_size
        self.img_mean = np.array([104., 117., 124.])

        # Runtime params
        with tf.device('/cpu:0'):
            self.input_images = tf.placeholder(tf.float32, [self.batch_size, None, None, 3])
        
        # checkpoints_dir = os.path.join(cache_folder, 'checkpoints_vgg')
        checkpoints_dir = os.path.join(cache_folder, 'checkpoints')
        vgg_var_scope = 'vgg_16'

        with tf.variable_scope(vgg_var_scope, reuse=False):
            with slim.arg_scope(vgg.vgg_arg_scope(bn=False, is_training=False)):
                _, end_points = vgg.vgg_16(self.input_images, num_classes=num_classes, is_training=False)
                
        self.scores = end_points['vgg_16/fc8/reduced']
        # self.scores = end_points['vgg_16/fc6/reduced']
                
        restorer = get_init_restorer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        print(str(datetime.now()) + ': Start Init')
        restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned-8000'))
        # restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned-20000'))
        # restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned-22000'))
        print(str(datetime.now()) + ': Finish Init')
        
    
    def predict_image(self, img, is_gray=False):
        assert(self.batch_size == 1)
        h, w, c = img.shape
        assert c == 3
        assert(np.min([h,w]) == self.scale_size)
        if is_gray:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            
        # img = cv2.resize(img, (self.scale_size, self.scale_size))
        
        img = img.astype(np.float32)
        img -= self.img_mean
            
        feed_dict = {self.input_images: [img]}
        out_scores = self.sess.run(self.scores, feed_dict=feed_dict)
        return out_scores

        