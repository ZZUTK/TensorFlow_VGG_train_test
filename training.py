#####################################################################################################
# train the VGG16 model
# written by Zhifei Zhang, Aug., 2016
# Details: https://github.com/ZZUTK/TensorFlow_VGG_train_test
#
# This is a demo of training process on CIFAR-10 dataset.
# Because VGG16 model is too large to quickly get some result, we use a shallow model
# (2 convolution + 1 fully connection), which is also named vgg16. but it is NOT VGG16 model.
#####################################################################################################

from shallow_model import vgg16
# from VGG16_model import vgg16
import tensorflow as tf
import numpy as np


# train_x is a 4-D matrix [num_images, img_height, img_width, num_channels]
# train_y is a 2-D matrix [num_images, num_classes] (using one-hot labels)
# valid_x is a 4-D matrix like train_x
# valid_y is a 2-D matrix like train_y
def training(train_x, train_y, valid_x=None, valid_y=None, format_size=[224, 224],
             batch_size=10, learn_rate=0.01, num_epochs=1, save_model=False, debug=False):
    
    assert len(train_x.shape) == 4
    [num_images, img_height, img_width, num_channels] = train_x.shape
    num_classes = train_y.shape[-1]
    num_steps = int(np.ceil(num_images / float(batch_size)))

    # build the graph and define objective function
    graph = tf.Graph()
    with graph.as_default():
        # build graph
        train_maps_raw = tf.placeholder(tf.float32, [None, img_height, img_width, num_channels])
        train_maps = tf.image.resize_images(train_maps_raw, format_size[0], format_size[1])
        train_labels = tf.placeholder(tf.float32, [None, num_classes])
        # logits, parameters = vgg16(train_maps, num_classes)
        logits = vgg16(train_maps, num_classes, isTrain=True, keep_prob=0.6)

        # loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, train_labels)
        loss = tf.reduce_mean(cross_entropy)

        # optimizer with decayed learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learn_rate, global_step, num_steps*num_epochs, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # prediction for the training data
        train_prediction = tf.nn.softmax(logits)
    
    # train the graph
    with tf.Session(graph=graph) as session:
        # saver to save the trained model
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())

        for epoch in range(num_epochs):
            for step in range(num_steps):
                offset = (step * batch_size) % (num_images - batch_size)
                batch_data = train_x[offset:(offset + batch_size), :, :, :]
                batch_labels = train_y[offset:(offset + batch_size), :]
                feed_dict = {train_maps_raw: batch_data, train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                if debug:
                    if step % int(np.ceil(num_steps/2.0)) == 0:
                        print('Epoch %2d/%2d step %2d/%2d: ' % (epoch+1, num_epochs, step, num_steps))
                        print('\tBatch Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, batch_labels)))
                        if valid_x is not None:
                            feed_dict = {train_maps_raw: valid_x, train_labels: valid_y}
                            l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
                            print('\tValid Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, valid_y)))

            print ('Epoch %2d/%2d:\n\tTrain Loss = %.2f\t Accuracy = %.2f%%' %
                   (epoch+1, num_epochs, l, accuracy(predictions, batch_labels)))
            if valid_x is not None and valid_y is not None:
                feed_dict = {train_maps_raw: valid_x, train_labels: valid_y}
                l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
                print('\tValid Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, valid_y)))

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

    # centralize images [num_images, img_height, img_width, num_channels]
    mu = np.mean(tr_x, axis=(0, 1, 2))
    # std = np.std(tr_x, axis=(0, 1, 2))
    print('Color center of images: %s' % mu )
    tr_x[:, :, :, :] -= mu.reshape(1, 1, 1, 3)
    te_x[:, :, :, :] -= mu.reshape(1, 1, 1, 3)

    # one-hot labels [num_images, num_classes]
    tr_y_onehot = np.zeros([len(tr_y), len(Labels)])
    te_y_onehot = np.zeros([len(te_y), len(Labels)])
    for i in range(len(tr_y)):
        tr_y_onehot[i][tr_y[i]] = 1.0
    for i in range(len(te_y)):
        te_y_onehot[i][te_y[i]] = 1.0

    # training on a subset to get a quick result
    print('Training ...')
    training(tr_x[::10, :, :, :], tr_y_onehot[::10, :],
             te_x[::10, :, :, :], te_y_onehot[::10, :],
             format_size=[32, 32], batch_size=50, learn_rate=0.1, num_epochs=10)

