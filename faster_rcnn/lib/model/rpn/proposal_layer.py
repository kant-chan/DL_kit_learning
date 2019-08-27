import numpy as np
import torch

from model.utils.config import cfg
from .generate_anchors import generate_anchors


class _ProposalLayer(nn.Module):
    '''Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors")
    '''
    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, inpt):
        '''for each (W, H) location i
        generate an anchor boxES centered on cell i
        apply predicted bbox deltas at cell i to each of the A anchors
        clip predicted boxes to image
        '''
        # bg->[0:9] fg->[9:18]
        scores = inpt[0][:,self._num_anchors:,:,:]
        bbox_deltas = inpt[1]
        im_info = inpt[2]
        cfg_key = inpt[3]

        pre_nms_tpN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)

        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        