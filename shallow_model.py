import layerConstructor as lc
import tensorflow as tf


def vgg16(input_maps, num_classes=1000, isTrain=False, keep_prob=1.0):

    # assume the input image shape is 224 x 224 x 3

    output1_1, kernel1_1, bias1_1 = lc.convolution_layer('conv1_1', input_maps, 12)
    output1_2 = lc.max_pooling_layer('pool1', output1_1)

    # output1_3 shape 112 x 112 x 64

    output2_1, kernel2_1, bias2_1 = lc.convolution_layer('conv2_1', output1_2, 24)
    output2_2 = lc.max_pooling_layer('pool2', output2_1)

    # drop out to avoid over fitting
    if isTrain:
        output2_2 = tf.nn.dropout(output2_2, keep_prob=keep_prob)

    # output5_4 shape 56 x 56 x 512

    output6_1, kernel6_1, bias6_1 = lc.fully_connection_layer('fc6_1', output2_2, num_classes)

    # output6_1 shape 1 x num_classes

    return output6_1
