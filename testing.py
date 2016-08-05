#####################################################################################################
# testing VGG16 model using a pre-trained model
# written by Zhifei Zhang, Aug., 2016
# Details: https://github.com/ZZUTK/TensorFlow_VGG_train_test
#
# The pre-trained VGG16 model and imagenet_classes.py are provided by Davi Frossard
# http://www.cs.toronto.edu/~frossard/post/vgg16/
#####################################################################################################

from VGG16_model import vgg16
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import tensorflow as tf

# build the graph
with tf.Graph().as_default():
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
    output, parameters = vgg16(input_maps)
    softmax = tf.nn.softmax(output)
    # Finds values and indices of the k largest entries
    k = 10
    values, indices = tf.nn.top_k(softmax, k)

# read pre-trained model
params = np.load('vgg16_weights.npz')
keys = sorted(params.keys())

# read sample image
img = imread('weasel.png', mode='RGB')
img = imresize(img, (224, 224))

# run the graph
with tf.Session() as sess:
    # load weights and biases
    for ind, key in enumerate(keys):
        print ind, key, np.shape(params[key])
        sess.run(parameters[ind].assign(params[key]))
    # testing on the sample image
    [prob, ind] = sess.run([value, indices], feed_dict={input_maps: [img]})
    for i in range(k):
        print class_names[ind[i]], prob[i]
    sess.close()
