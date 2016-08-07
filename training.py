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
# valid_x is a 4-D matrix like train_x
# valid_y is a 2-D matrix like train_y
def training(train_x, train_y, valid_x=None, valid_y=None,
             batch_size=10, learn_rate=0.01, num_epochs=1, save_model=False):
    
    assert len(train_x.shape) == 4
    [num_images, img_height, img_width, num_channels] = train_x.shape
    num_classes = train_y.shape[-1]

    # build the graph and define objective function
    graph = tf.Graph()
    with graph.as_default():
        # build graph
        train_maps_raw = tf.placeholder(tf.float32, [batch_size, img_height, img_width, num_channels])
        train_maps = tf.image.resize_images(train_maps_raw, 224, 224)
        train_labels = tf.placeholder(tf.int32, [batch_size, num_classes])
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
            valid_maps_raw = tf.constant(valid_x)
            valid_maps = tf.image.resize_images(valid_maps_raw, 224, 224)
            valid_logits, _ = vgg16(valid_maps)
            valid_prediction = tf.nn.softmax(valid_logits)

    # train the graph
    with tf.Session(graph=graph) as session:
        # saver to save the trained model
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        for epoch in range(num_epochs):
            num_steps = int(np.ceil(num_images / float(batch_size)))
            for step in range(num_steps):
                offset = (step * batch_size) % (num_images - batch_size)
                batch_data = train_x[offset:(offset + batch_size), :, :, :]
                batch_labels = train_y[offset:(offset + batch_size), :]
                feed_dict = {train_maps: batch_data, train_labels: batch_labels}
                _, l, predictions, _ = session.run(
                    [optimizer, loss, train_prediction, parameters], feed_dict=feed_dict)
                if step % int(np.ceil(num_steps/20.0)) == 0:
                    print('Batch loss at step %d: %.2f' % (step, l))
                    print('Batch accuracy: %.2f%%' % accuracy(predictions, batch_labels))
                    if valid_x is not None:
                        print('Validation accuracy: %.2f%%' % accuracy(valid_prediction.eval(), valid_y))
        # Save the variables to disk
        if save_model:
            save_path = saver.save(session, 'model.tensorflow')
            print('The model has been saved to ' + save_path)
        session.close()


# predictions is a 2-D matrix [num_images, num_classes]
# labels is a 2-D matrix like predictions
def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


if __name__ == '__main__':
    # load data
    print('Loading dataset ...')
    data = np.load('CIFAR-10.dataset.npz')
    tr_x = data['train_x']
    tr_y = data['train_y']
    te_x = data['test_x']
    te_y = data['test_y']
    Labels = data['labels']

    # centralize images
    mu = np.mean(tr_x, axis=(0, 1, 2))
    print('Color center of images: %s' % mu )
    tr_x[:, :, :, :] -= mu.reshape(1, 1, 1, 3)
    te_x[:, :, :, :] -= mu.reshape(1, 1, 1, 3)

    # one-hot labels
    tr_y_onehot = np.zeros([len(tr_y), len(Labels)])
    te_y_onehot = np.zeros([len(te_y), len(Labels)])
    for i in range(0, len(Labels)):
        tr_y_onehot[i][tr_y[i]] = 1
        te_y_onehot[i][te_y[i]] = 1

    # training
    print('Training ...')
    training(tr_x, tr_y_onehot, te_x, te_y)
