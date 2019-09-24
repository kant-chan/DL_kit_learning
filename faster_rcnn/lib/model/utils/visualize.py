import numpy as np
import cv2
import matplotlib.pyplot as plt

import _init_path
from model.utils.config import cfg
from model.rpn.generate_anchors import generate_anchors

def display(im_data, gt_boxes, num_boxes):
    '''
    im_data: (batch_size, C, H, W)
    gt_boxes: (batch_size, 20, 5)
    num_boxes: [num]
    '''
    assert im_data.size(0) == 1

    im_data = im_data[0].numpy()
    gt_boxes = gt_boxes[0].numpy()
    num_boxes = num_boxes[0].item()
    gt_boxes = gt_boxes[:num_boxes]

    # (c, h, w) -> (h, w, c)
    im_data = im_data.transpose((1, 2, 0))
    # rgb -> bgr
    # im_data = im_data[:,:,::-1]
    # im_data = im_data * 255
    # im_data = im_data.astype(np.uint8)
    # cfg.PIXEL_MEANS
    im_data += cfg.PIXEL_MEANS
    # print(im_data.dtype, np.max(im_data), np.min(im_data))
    for i in range(num_boxes):
        box = gt_boxes[i]
        print(box)
        color=(np.random.randint(127, 255), np.random.randint(127, 255), np.random.randint(127, 255))
        im_data = cv2.rectangle(im_data, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])), color, 2)

    cv2.imwrite('./test.jpg', im_data)


def vis_gen_anchors(im_data):
    im_data = im_data[0].numpy()
    im_data = im_data.transpose((1, 2, 0))
    im_data += cfg.PIXEL_MEANS

    h, w = im_data.shape[:2]
    offset_h, offset_w = h // 2, w // 2

    anchors = generate_anchors()
    anchors += np.array([[offset_w, offset_h, offset_w, offset_h]])
    for i in range(len(anchors)):
        box = anchors[i]
        # print(box)
        color=(0, 0, 255)
        im_data = cv2.rectangle(im_data, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])), color, 2)
    cv2.imwrite('./test.jpg', im_data)
        