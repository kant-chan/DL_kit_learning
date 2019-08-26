from PIL import Image
import numpy as np
import random
import time
import torch
import torch.utils.data as data

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes


class RoibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT  # 600
        self.trim_width = cfg.TRAIN.TRIM_WIDTH    # 600
        self.max_num_box = cfg.MAX_NUM_GT_BOXES   # 20
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)

        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i * batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)

            if ratio_list[right_idx] < 1:
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                target_ratio = ratio_list[right_idx]
            else:
                target_ratio = 1
            self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index
        
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['img_info'])
        data_height, data_width = data.size(1), data.size(2)

        return data, im_info

    def __len__(self):
        return len(self._roidb)
