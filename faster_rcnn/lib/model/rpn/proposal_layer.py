import numpy as np
import torch
import torch.nn as nn

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes
from model.nms import nms


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
        scores = inpt[0][:,self._num_anchors:,:,:]  # (batch_size, 9, H, W)
        bbox_deltas = inpt[1]                       # (batch_size, 36, H, W)
        im_info = inpt[2]
        cfg_key = inpt[3]

        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N    # train: 12000, test: 6000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # train: 2000, test: 300
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH         # train: 0.7, test: 0.7
        min_size = cfg[cfg_key].RPN_MIN_SIZE             # train: 8, test: 16

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)

        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        # example: shift_x is [0, 1, 2, 3, 4]
        #          shift_y is [0, 1, 2, 3]
        # result:
        # shift_x: [[0, 1, 2, 3, 4],
        #           [0, 1, 2, 3, 4],
        #           [0, 1, 2, 3, 4],
        #           [0, 1, 2, 3, 4]]
        # shift_y: [[0, 0, 0, 0, 0],
        #           [1, 1, 1, 1, 1],
        #           [2, 2, 2, 2, 2]
        #           [3, 3, 3, 3, 3]]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # [[0, 0, 0, 0],
        #  [1, 0, 1, 0],
        #  [2, 0, 2, 0],
        #  ... so on]
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(),
                                             shift_y.ravel(),
                                             shift_x.ravel(),
                                             shift_y.ravel())).transpose())
        
        shifts = shifts.contiguous().type_as(scores).float() # torch.float32

        A = self._num_anchors  # len(scales) * len(anchors) = 9
        K = shifts.size(0)     # W_feat * H_feat

        self._anchors = self._anchors.type_as(scores)
        # (K, A, 4)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # transpose and reshape predicted bbox transformations
        # to get them into the same order as the anchors
        # original shape (batch_size, 36, H, W)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous() # (batch_size, H, W, 36)
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)          # (batch_size, H*W*9, 4)

        # same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous() # (batch_size, H, W, 9)
        scores = scores.view(batch_size, -1)             # (batch_size, H*W*9)

        # convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)


        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            proposals_single = proposals_keep[i]  # (H*W*9, 4)
            scores_single = scores_keep[i]        # (H*W*9,)
            order_single = order[i]

            # bellow line scores_keep.numel() error ?
            # if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
            if pre_nms_topN > 0 and pre_nms_topN < scores_single.numel():
                order_single = order_single[:pre_nms_topN]
            
            proposals_single = proposals_single[order_single,:]
            scores_single = scores_single[order_single].view(-1, 1)

            # - apply nms (e.g. threshhold=0.7)
            # - take after_nms_topN (e.g. 300)
            # - return the top proposals (-> RoIs top)
            keep_idx_i = nms.nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i,:]
            scores_single = scores_single[keep_idx_i,:]

            # padding 0 at the end
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single
        
        return output

    def _filter_boxes(self, boxes, min_size):
        '''remove all boxes with any side smaller than min_size
        '''
        ws = boxes[:,:,2] - boxes[:,:,0] + 1
        hs = boxes[:,:,3] - boxes[:,:,1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))
        return keep