# -*- coding: utf-8 -*-
# fork from [https://github.com/ykpengba/AlexNet-A-Practical-Implementation]

import tensorflow as tf
import numpy as np


def LRN(x, R, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha, beta=beta, bias=bias, name=name)

def fc_layer(x, in_d, out_d, relu_flag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[in_d, out_d], dtype=tf.float32)
        b = tf.get_variable("b", shape=[out_d], dtype=tf.float32)
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if relu_flag:
            return tf.nn.relu(out)
        else:
            return out

def conv_layer(x, k_h, k_w, stride_x, stride_y, channel, name, padding='SAME', groups=1):
    in_channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, stride_x, stride_y, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[k_h, k_w, in_channel/groups, channel])
        b = tf.get_variable("b", shape=[channel])

        x_new = tf.split(value=x, num_or_size_splits=groups, axis=3)
        w_new = tf.split(value=w, num_or_size_splits=groups, axis=3)

        feature_map = [conv(t1, t2) for t1, t2 in zip(x_new, w_new)]
        merge_map = tf.concat(axis=3, values=feature_map)
        out = tf.nn.bias_add(merge_map, b)
        return tf.nn.relu(tf.reshape(out, merge_map.get_shape().as_list()), name=scope.name)

def maxpool_layer(x, k_h, k_w, stride_x, stride_y, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, stride_x, stride_y, 1], name=name, padding=padding)

def dropout(x, keep_prob, name=None):
    return tf.nn.dropout(x, keep_prob, name)

class AlexNet(object):
    def __init__(self, x, keep_prob, class_nums, skip, model_path='bvlc_alexnet.npy'):
        self.X = x
        self.KEEP_PROB = keep_prob
        self.CLASS_NUMS = class_nums
        self.SKIP = skip
        self.MODEL_PATH = model_path

        self.build()

    def build(self):
        conv1 = conv_layer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxpool_layer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = conv_layer(pool1, 5, 5, 1, 1, 256, "conv2", groups=2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxpool_layer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = conv_layer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = conv_layer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)

        conv5 = conv_layer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
        pool5 = maxpool_layer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fc_in = tf.reshape(pool5, [-1, 256*6*6])
        fc1 = fc_layer(fc_in, 256*6*6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEP_PROB)

        fc2 = fc_layer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEP_PROB)

        self.fc3 = fc_layer(dropout2, 4096, self.CLASS_NUMS, True, "fc8")

    def load_model(self, sess):
        w_dict = np.load(self.MODEL_PATH, encoding="bytes").item()
        for name in w_dict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse=True):
                    for p in w_dict[name]:
                        if len(p.shape) == 1:
                            sess.run(tf.get_variable("b", trainable=False).assign(p))
                        else:
                            sess.run(tf.get_variable("w", trainable=False).assign(p))