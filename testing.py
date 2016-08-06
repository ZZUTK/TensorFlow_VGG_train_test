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
    output, parameters = vgg16(input_maps)
    softmax = tf.nn.softmax(output)
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
    [prob, ind] = sess.run([values, indices], feed_dict={input_maps: [img]})
    prob = prob[0]
    ind = ind[0]
    print('\nClassification Result:')
    for i in range(k):
        print('Category Name: %s \nProbability: %.2f%%\n' % (class_names[ind[i]], prob[i]*100))
    sess.close()
