#####################################################################################################
# train the VGG16 model
# written by Zhifei Zhang, Aug., 2016
# Details: https://github.com/ZZUTK/TensorFlow_VGG_train_test
#
# This is a demo of training process using CIFAR-10 dataset
#####################################################################################################

from VGG16_model import vgg16
import tensorflow as tf
import numpy as np


# train_x is a 4-D matrix [num_images, img_height, img_width, num_channels]
# train_y is a 2-D matrix [num_images, num_classes] (using one-hot labels)
def training(train_x, train_y, valid_x=None, valid_y=None,
             batch_size=10, learn_rate=0.01, num_epochs=1, save_model=False):
    assert len(train_x.shape) == 4
    [num_images, img_height, img_width, num_channels] = train_x.shape

    # build the graph and define objective function
    graph = tf.Graph()
    with graph.as_default():
        # build graph
        train_maps = tf.placeholder(tf.float32, [batch_size, img_height, img_width, num_channels])
        train_labels = tf.placeholder(tf.int32, [batch_size, num_labels])
        logits, parameters = vgg16(train_maps)
        # loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, train_labels)
        loss = tf.reduce_mean(cross_entropy)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
        # prediction for the training data
        train_prediction = tf.nn.softmax(logits)
        # prediction for the validation data
        if valid_x is not None:
            valid_maps = tf.constant(valid_x)
            valid_logits, _ = vgg16(valid_maps)
            valid_prediction = tf.nn.softmax(valid_logits)

    # train the graph
    with tf.Session(graph=graph) as sess:
        # saver to save the trained model
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        for epoch in range(num_epochs):
            num_steps = int(np.ceil(num_images / float(batch_size)))
            for step in range(num_steps):
                offset = (step * batch_size) % (num_images - batch_size)
                batch_data = train_x[offset:(offset + batch_size), :, :, :]
                batch_labels = train_y[offset:(offset + batch_size), :]
                feed_dict = {train_maps: batch_data, train_labels: batch_labels}
                _, l, predictions, params = session.run(
                    [optimizer, loss, train_prediction, parameters], feed_dict=feed_dict)
                if step % int(np.ceil(num_steps/20.0)) == 0:
                    print('Batch loss at step %d: %.2f' % (step, l))
                    print('Batch accuracy: %.2f%%' % accuracy(predictions, batch_labels))
                    if valid_x is not None:
                        print('Validation accuracy: %.2f%%' % accuracy(valid_prediction.eval(), valid_labels))
        # Save the variables to disk
        if save_model:
            save_path = saver.save(sess, 'model.tensorflow')
            print('The model has been saved to ' + save_path)
        sess.close()


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


if __name__ == '__main__':
    # load data
    data = np.load('CIFAR-10.dataset.npz')
    tr_x = data['train_x']
    tr_y = data['train_y']
    te_x = data['test_x']
    te_y = data['test_y']
    labels = data['labels']

    # one-hot label
    tr_y_onehot = np.zeros([len(tr_y), len(labels)])
    te_y_onehot = np.zeros([len(te_y), len(labels)])
    for i in range(0, len(labels)):
        tr_y_onehot[i][tr_y[i]] = 1
        te_y_onehot[i][te_y[i]] = 1

    # training
    training(tr_x, tr_y_onehot, te_x, te_y)
