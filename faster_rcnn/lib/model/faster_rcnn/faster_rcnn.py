import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from model.utils.config import cfg
from model.rpn.rpn import _RPN


class _fasterRCNN(nn.Module):
    '''faster rcnn'''
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.data
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_rpn(im_data)

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)