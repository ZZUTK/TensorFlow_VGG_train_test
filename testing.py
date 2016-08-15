#####################################################################################################
# testing VGG16 model using a pre-trained model
# written by Zhifei Zhang, Aug., 2016
# Details: https://github.com/ZZUTK/TensorFlow_VGG_train_test
#
# The pre-trained VGG16 model parameters and imagenet_classes.py are provided by Davi Frossard
# http://www.cs.toronto.edu/~frossard/post/vgg16/
#####################################################################################################

from VGG16_model import vgg16
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import tensorflow as tf


# build the graph
graph = tf.Graph()
with graph.as_default():
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # zero mean of input
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3])
    output = vgg16(input_maps - mean, return_all=True)
    softmax = tf.nn.softmax(output[-3])
    # Finds values and indices of the k largest entries
    k = 3
    values, indices = tf.nn.top_k(softmax, k)

# read sample image
img = imread('weasel.png', mode='RGB')
img = imresize(img, [224, 224])

# run the graph
with tf.Session(graph=graph) as sess:
    # restore model parameters
    saver = tf.train.Saver()
    print('Restoring VGG16 model parameters ...')
    saver.restore(sess, 'VGG16_modelParams.tensorflow')
    # testing on the sample image
    [prob, ind, out] = sess.run([values, indices, output], feed_dict={input_maps: [img]})
    prob = prob[0]
    ind = ind[0]
    print('\nClassification Result:')
    for i in range(k):
        print('\tCategory Name: %s \n\tProbability: %.2f%%\n' % (class_names[ind[i]], prob[i]*100))
    sess.close()
