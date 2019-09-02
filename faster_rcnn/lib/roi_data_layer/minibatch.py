# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
from scipy.misc import imread

from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
    '''Given aa roidb, construct a minibatch sampled from it
    '''
    num_images = len(roidb)
    random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)  # [0]
    assert cfg.TRAIN.BATCH_SIZE % num_images == 0, \
        'num_iamges ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob. formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, 'Single batch only'
    assert len(roidb) == 1, 'Single batch only'

    if cfg.TRAIN.USE_ALL_GT:
        # emit background
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # for the COCO ground truth boxes
        pass

    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:,0:4] = roidb[0]['boxes'][gt_inds,:] * im_scales[0]
    gt_boxes[:,4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    # array([[h, w, im_scale]])
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    blobs['img_id'] = roidb[0]['img_id']
    
    return blobs

def _get_image_blob(roidb, scale_inds):
    '''build an input blob from the images in the roidb at the specified scales
    '''
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = imread(roidb[i]['image'])   # im.shape->(h,w,c)  type(im)->np.array  dtype->uint8

        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        # rgb -> bgr, since the original one using cv2
        im = im[:,:,::-1]
        
        if roidb[i]['flipped']:
            im = im[:,::-1,:]
        
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        # array([[[102.9801, 115.9465, 122.7717]]])  600  1000
        # prep_im_for_blob: resize the min of width and height, remain ratio same
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)

        im_scales.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)

    # print('===>', blob.shape, im_scale)  # (1, 600, 800, 3) 1.6
    return blob, im_scales

