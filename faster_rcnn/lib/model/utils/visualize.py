import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    print(im_data.dtype, np.max(im_data), np.min(im_data))

    cv2.imwrite('./test.jpg', im_data)