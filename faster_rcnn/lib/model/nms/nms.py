import torch
import numpy as np


def nms(bboxes, scores, threshhold=0.5):
    '''Do NMS, bboxes and scores has sorted
    bboxes: (H*W*9, 4) -> (topN, 4)
    scores: (H*W*9,)   -> (topN, )
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    batch_size = bboxes.size(0)
    all_index = torch.arange(batch_size)  # dtype->torch.int64
    keep_idx = []
    areas = (x2 - x1) * (y2 - y1)  # (topN, )
    while all_index.numel() > 0:
        i = all_index[0]
        keep_idx.append(i)

        if all_index.numel() == 1:
            break
        
        ix1 = x1[all_index[1:]].clamp(min=x1[i])  # (topN-1, )
        ix2 = x2[all_index[1:]].clamp(max=x2[i])
        iy1 = y1[all_index[1:]].clamp(min=y1[i])
        iy2 = y2[all_index[1:]].clamp(max=y2[i])

        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (areas[i] + areas[all_index[1:]] - inter)  # (topN-1, )

        idx = (iou <= threshhold).nonzero().squeeze()
        if idx.numel() == 0:
            break

        all_index = all_index[idx+1]

    return torch.LongTensor(keep_idx)



def display(cordlist):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    back = np.zeros((800, 800, 3), dtype=np.uint8)
    for index, cord in enumerate(cordlist):
        # print('draw ',cord)
        color=(np.random.randint(127, 255), np.random.randint(127, 255), np.random.randint(127, 255))
        # print('color is ', color)
        cv2.rectangle(back, (int(cord[0]), int(cord[1])), (int(cord[2]),int(cord[3])), color, 1)
        # cv2.putText(back, str(cord[4]), (int(cord[0]), int(cord[1])), cv2.FONT_ITALIC, 0.5, color, 1)
    plt.imshow(back)
    plt.show()
    return back


if __name__ == '__main__':
    boxes = np.array([[12, 190, 300, 399],
                      [221, 250, 389, 500],
                      [100, 100, 150, 168],
                      [166, 70, 312, 190],
                      [28, 130, 134, 302]])
    scores = np.array([0.9, 0.79, 0.63, 0.55, 0.3])
    display(boxes)
    t_boxes = torch.from_numpy(boxes)
    t_scores = torch.from_numpy(scores)
    thresh = 0.1
    keep_ids = nms(t_boxes, t_scores, thresh)
    display(boxes[keep_ids.numpy()])