'''
load data by tensorflow api.
the file can run not depended on others.
'''

# -*- coding: utf-8 -*-

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=False)

import tensorflow as tf
import math

# Parameters
learning_rate = 0.0001
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 128
num_input = 784
num_classes = 10

def inference(images, hidden1_units, hidden2_units):
    with tf.variable_scope('hidden1'):
        weights = tf.get_variable('weights', 
                                  [num_input, hidden1_units], 
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(num_input))))
        biases = tf.get_variable('biases',
                                 [hidden1_units],
                                 initializer=tf.zeros_initializer())
        # fully connected
        # hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        # none relu
        hidden1 = tf.matmul(images, weights) + biases
    with tf.variable_scope('hidden2'):
        weights = tf.get_variable('weights',
                                  [hidden1_units, hidden2_units],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(hidden1_units))))
        biases = tf.get_variable('biases',
                                 [hidden2_units],
                                 initializer=tf.zeros_initializer())
        # hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        # none relu
        hidden2 = tf.matmul(hidden1, weights) + biases
    with tf.variable_scope('softmax_linear'):
        weights = tf.get_variable('weights',
                                  [hidden2_units, num_classes],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(hidden1_units)))
        biases = tf.get_variable('biases',
                                 [num_classes],
                                 initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def training(loss, learning_rate):
    # tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.get_variable('global_step', trainable=False, initializer=tf.constant(0))
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    labels = tf.to_int64(labels)
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None])

logits = inference(X, n_hidden_1, n_hidden_2)
loss_op = loss(logits, Y)
loss_op = tf.reduce_mean(loss_op)
train_op = training(loss_op, learning_rate)
eval_correct = evaluation(logits, Y)
accuracy = eval_correct / batch_size

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("step: {:4d}".format(step) + ", loss: " + \
                  "{:.4f}".format(loss) + ", training accuracy: " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("testing accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))