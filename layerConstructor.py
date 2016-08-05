#####################################################################################################
# functions to easily construct convolution, pooling, and fully connection layers using TensorFlow  #
# written by Zhifei Zhang, Aug., 2016                                                               #
# Details: https://github.com/ZZUTK/TensorFlow_VGG_train_test                                       #
#####################################################################################################

import tensorflow as tf


# construct a convolution layer
# input_maps is a 4-D matrix [batch, height, width, channels]
# kernel_size = [height, width]
# stride indicates the stride on each dimension of input_maps
def convolution_layer(layer_name, input_maps, num_output_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1]):
    num_input_channels = input_maps.get_shape()[-1].value
    with tf.name_scope(layer_name) as scope:
        kernel = tf.get_variable(scope+'W',
                                 shape=[kernel_size[0], kernel_size[1], num_input_channels, num_output_channels],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        convolution = tf.nn.conv2d(input_maps, kernel, stride, padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[num_output_channels], dtype=tf.float32), trainable=True, name='b')
        output = tf.nn.relu(tf.nn.bias_add(convolution, bias), name=scope)
        return output, kernel, bias


# construct a max pooling layer
def max_pooling_layer(layer_name, input_maps, kernel_size=[2, 2], stride=[1, 2, 2, 1]):
    output = tf.nn.max_pool(input_maps,
                            ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=stride,
                            padding='SAME',
                            name=layer_name)
    return output


# construct a average pooling layer
def avg_pooling_layer(layer_name, input_maps, kernel_size=[2, 2], stride=[1, 2, 2, 1]):
    output = tf.nn.avg_pool(input_maps,
                            ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=stride,
                            padding='SAME',
                            name=layer_name)
    return output


# construct a fully connection layer
def fully_connection_layer(layer_name, input_maps, num_output_nodes):
    shape = input_maps.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    with tf.name_scope(layer_name) as scope:
        kernel = tf.get_variable(scope+'W',
                                 shape=[size, num_output_nodes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[num_output_nodes], dtype=tf.float32), trainable=True, name='b')
        flat = tf.reshape(input_maps, [-1, size])
        output = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, kernel), bias))
        return output, kernel, bias