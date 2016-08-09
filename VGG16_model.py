#####################################################################################################
# network structure of VGG16 model
# written by Zhifei Zhang, Aug., 2016
# Details: https://github.com/ZZUTK/TensorFlow_VGG_train_test
#####################################################################################################

import layerConstructor as lc
import tensorflow as tf


def vgg16(input_maps, num_classes=1000, isTrain=False, keep_prob=1.0):

    # assume the input image shape is 224 x 224 x 3

    output1_1, kernel1_1, bias1_1 = lc.convolution_layer('conv1_1', input_maps, 64)
    output1_2, kernel1_2, bias1_2 = lc.convolution_layer('conv1_2', output1_1, 64)
    output1_3 = lc.max_pooling_layer('pool1', output1_2)

    # output1_3 shape 112 x 112 x 64

    output2_1, kernel2_1, bias2_1 = lc.convolution_layer('conv2_1', output1_3, 128)
    output2_2, kernel2_2, bias2_2 = lc.convolution_layer('conv2_2', output2_1, 128)
    output2_3 = lc.max_pooling_layer('pool2', output2_2)
    
    # output2_3 shape 56 x 56 x 128
    
    output3_1, kernel3_1, bias3_1 = lc.convolution_layer('conv3_1', output2_3, 256)
    output3_2, kernel3_2, bias3_2 = lc.convolution_layer('conv3_2', output3_1, 256)
    output3_3, kernel3_3, bias3_3 = lc.convolution_layer('conv3_3', output3_2, 256)
    output3_4 = lc.max_pooling_layer('pool3', output3_3)
    
    # output3_4 shape 28 x 28 x 256
    
    output4_1, kernel4_1, bias4_1 = lc.convolution_layer('conv4_1', output3_4, 512)
    output4_2, kernel4_2, bias4_2 = lc.convolution_layer('conv4_2', output4_1, 512)
    output4_3, kernel4_3, bias4_3 = lc.convolution_layer('conv4_3', output4_2, 512)
    output4_4 = lc.max_pooling_layer('pool4', output4_3)

    # output4_4 shape 14 x 14 x 512

    output5_1, kernel5_1, bias5_1 = lc.convolution_layer('conv5_1', output4_4, 512)
    output5_2, kernel5_2, bias5_2 = lc.convolution_layer('conv5_2', output5_1, 512)
    output5_3, kernel5_3, bias5_3 = lc.convolution_layer('conv5_3', output5_2, 512)
    output5_4 = lc.max_pooling_layer('pool5', output5_3)

    # drop out to avoid over fitting
    if isTrain:
        output5_4 = tf.nn.dropout(output5_4, keep_prob=keep_prob)
    
    # output5_4 shape 7 x 7 x 512

    output6_1, kernel6_1, bias6_1 = lc.fully_connection_layer('fc6_1', output5_4, 4096)
    output6_2, kernel6_2, bias6_2 = lc.fully_connection_layer('fc6_2', output6_1, 4096)
    output6_3, kernel6_3, bias6_3 = lc.fully_connection_layer('fc6_3', output6_2, num_classes)

    return output6_3
