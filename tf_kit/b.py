# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j, axis=0), labels.take(j)

def gen_data():
    """
    generate original data.

    Parameters:
        void - this is the first param

    Returns:
        features - numpy type
        labels - numpy type

    Raises:
        KeyError - raises an exception
    """
    num_inputs = 2
    num_examples = 1000
    true_w = np.array([2, -3.4])
    true_b = 4.2

    features = np.random.randn(num_examples, num_inputs)
    labels = np.matmul(features, true_w) + true_b
    labels += 0.01 * np.random.randn(*labels.shape) + 0
    return features, labels

##### draw data
# print(features[0])
# print(labels[0])
##### 3D scatter
# x, y, z = features[:, 0], features[:, 1], labels
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x, y, z)
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'green'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'blue'})
# # plt.scatter(features[:, 1], labels)
# plt.show()

##### define network
X = tf.placeholder(tf.float64, shape=[None, 2])
y = tf.placeholder(tf.float64, shape=[None, ])
W = tf.get_variable('W', [2, 1], dtype=tf.float64, initializer=tf.zeros_initializer)
b = tf.get_variable('b', [1, ], dtype=tf.float64, initializer=tf.zeros_initializer)

y_hat = tf.matmul(X, W) + b
y_hat = tf.reshape(y_hat, [-1])
loss = tf.losses.mean_squared_error(labels=y, predictions=y_hat)
tf.summary.scalar('loss', loss)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./log')
writer.add_graph(tf.get_default_graph())
init = tf.global_variables_initializer()

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(init)
    num_epochs = 5
    i = 0
    features, labels = gen_data()
    for epoch in range(num_epochs):
        for X_, y_ in data_iter(10, features, labels):
            sess.run(train, feed_dict={X: X_, y: y_})
            summary, l = sess.run([merged, loss], feed_dict={X: X_, y: y_})
            writer.add_summary(summary, i)
            i += 1
        print('epochs: %d, losses: %f' % (epoch + 1, sess.run(loss, feed_dict={X: X_, y: y_})))
    writer.close()
    print(sess.run(W))
    print(sess.run(b))