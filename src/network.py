import tensorflow as tf
import tensorflow.contrib.slim as slim


def my_bn(x, is_training):
    # x = slim.batch_norm(x, is_training=is_training, center=None)
    return slim.batch_norm(x, is_training=is_training)


def vgg_arg_scope(bn=False, weight_decay=0.00005, is_training=True):
    """Defines the VGG arg scope.
    Args:
      bn: Batch normalization switch on or off.
      weight_decay: The l2 regularization coefficient.
      is_training: You know what
    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        if bn:
            with slim.arg_scope([slim.conv2d], padding='SAME', normalizer_fn=lambda x: my_bn(x, is_training=is_training)) as arg_sc:
                return arg_sc
        else:
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc


def vgg_16(inputs, num_classes=1000, is_training=True, spatial_reduce=True):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    end_points_collection = tf.get_variable_scope().original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.conv2d(net, num_classes, [1, 1], scope='fc8', activation_fn=None, normalizer_fn=None)

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_reduce:
            net = tf.reduce_mean(net, [1, 2], name='fc8/reduced')
            end_points[tf.get_variable_scope().original_name_scope + 'fc8/reduced'] = net
            
        # for op in tf.get_default_graph().as_graph_def().node:
        #     print(str(op.name))
            
        return net, end_points
    

def vgg_16_12(inputs, is_training=True, spatial_reduce=True):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    end_points_collection = tf.get_variable_scope().original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.conv2d(net, 12, [1, 1], scope='fc8', activation_fn=None, normalizer_fn=None)

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_reduce:
            net = tf.reduce_mean(net, [1, 2], name='fc8/reduced')
            end_points[tf.get_variable_scope().original_name_scope + 'fc8/reduced'] = net
            
        # for op in tf.get_default_graph().as_graph_def().node:
        #     print(str(op.name))
            
        return net, end_points
    
def vgg_16_short(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_reduce=True,
           scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
    Returns:
        the last op containing the log predictions and end_points dict.
    """
    end_points_collection = tf.get_variable_scope().original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        
        net = tf.stop_gradient(net)
        
        # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # net = tf.stop_gradient(net)
            
        # Use conv2d instead of fully_connected layers.
        # vc layer
        net = slim.conv2d(net, 512, [1, 1], scope='conv5')
        # template layer
        net = slim.conv2d(net, num_classes, [14, 14], padding='VALID',activation_fn=None, normalizer_fn=None, scope='fc6')
        
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_reduce:
            net = tf.reduce_mean(net, [1, 2], name='fc6/reduced')
            end_points[tf.get_variable_scope().original_name_scope + 'fc6/reduced'] = net
            
            
        
        # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        # net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
        # Convert end_points_collection into a end_point dict.
        # end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        # if spatial_squeeze:
        #     net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        #     end_points[tf.get_variable_scope().original_name_scope + 'fc8/squeezed'] = net

        return net, end_points



def alexnet(inputs, is_training=True, spatial_squeeze=True):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    end_points_collection = tf.get_variable_scope().original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
        net = slim.conv2d(inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
        net = slim.conv2d(net, 256, [5, 5], 1, scope='conv2')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
        net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')
        # net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.conv2d(net, 384, [3, 3], 1, scope='conv4')
        # net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')  # normalizer_fn=None,
        net = slim.conv2d(net, 18392, [1, 1], scope='fc8', activation_fn=None, normalizer_fn=None)

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc7/squeezed')
            end_points[tf.get_variable_scope().original_name_scope + 'fc7/squeezed'] = net
        return net, end_points

vgg_16.default_image_size = 224