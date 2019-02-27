#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10_read.py
# Author: muzhi <zhimu1226@gmail.com>

import cPickle

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = cPickle.load(fo)
  return dict

if __name__ == '__main__':
  print unpickle('./bin/data_batch_0')
