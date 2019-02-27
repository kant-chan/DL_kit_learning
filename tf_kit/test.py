# -*- coding: utf-8 -*-

import numpy as np

####
# 
####
# a = np.array([[1, 2], [3, 4]])
# b = np.array([1, 2])
# c = np.array([[1, -1],
#               [0, 1]])
# r = a * c
# print(r)
# r = a * b
# print(r)
# r = np.dot(a, b)
# print(r)
# r = np.matmul(a, b)
# print(r)
# print(*(5, 7))

# e = np.array([[1,2,3],
#               [4,5,6],
#               [9,0,6],
#               [5,7,2]])
# print(e.take([0,1,3], axis=0))

# d = np.concatenate((a, c), axis=0)
# print(d)

#############
# 
#############
# def batch():
#     for i in range(10):
#         yield i

# a = batch()

# for i in range(3):
#     print(next(a))

#############
# 
#############
# a = np.array([[[1,2],[2,3]],
#               [[1,2],[2,3]],
#               [[1,2],[2,3]]])

# b = a.reshape((3,-1))

# print(a.shape)
# print(b.shape)
# print(b)
m = np.array(12)
print(m.shape)

m = np.array([[[1,2],[4,5]],
              [[2,3],[3,6]],
              [[4,8],[0,6]]])
print(m.take([1,0], axis=0))
#############
# 
#############
# import tensorflow as tf

# tensor2 = tf.constant(-1.0, shape=[2, 3])
# tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])
# tensor1 = tf.constant(0)

# sess = tf.Session()

# print(sess.run(tensor))
# print(tensor.get_shape())
# print(sess.run(tf.shape(tensor1)))