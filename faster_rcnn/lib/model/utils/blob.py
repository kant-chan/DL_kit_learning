# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
# from scipy.misc import imread, imresize
import cv2


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    '''mean subtract and scale an image for use in a blob
    '''
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[:2])
    im_size_max = np.max(im_shape[:2])
    im_scale = float(target_size) / float(im_size_min)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    # print('===>', np.max(im))  # 152.0199
    return im, im_scale

def im_list_to_blob(ims):
    '''convert a list images into a network input
    '''
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    
    return blob
