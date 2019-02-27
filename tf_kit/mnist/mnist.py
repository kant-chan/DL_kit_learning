# -*- coding: utf-8 -*-

import math
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

DEBUG = True

# 初始示例
def inference_0(images):
    with tf.variable_scope('hidden1'):
        weights = tf.get_variable('weights',
                                  [IMAGE_PIXELS, NUM_CLASSES],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(IMAGE_PIXELS))))
        biases = tf.get_variable('biases',
                                 [NUM_CLASSES],
                                 initializer=tf.zeros_initializer())
        logits = tf.matmul(images, weights) + biases
        return logits

def inference(images, hidden1_units, hidden2_units):
    with tf.variable_scope('hidden1'):
        weights = tf.get_variable('weights', 
                                  [IMAGE_PIXELS, hidden1_units], 
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(IMAGE_PIXELS))))
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
                                  [hidden2_units, NUM_CLASSES],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(hidden1_units)))
        biases = tf.get_variable('biases',
                                 [NUM_CLASSES],
                                 initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss(logits, labels):
    global DEBUG
    labels = tf.to_int64(labels)
    if DEBUG:
        print(labels.get_shape(), logits.get_shape())
        DEBUG = False
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.get_variable('global_step', trainable=False, initializer=tf.constant(0))
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))