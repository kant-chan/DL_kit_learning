import numpy as np


def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:,np.newaxis]
    hs = hs[:,np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _whctrs(anchor):
    '''return w, h, x center, y center for an anchor(window)
    '''
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _ratio_enum(anchor, ratios):
    '''enumerate a set of anchors for each aspect ratio wrt an anchor
    '''
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))  # [23. 16. 11.]
    hs = np.round(ws * ratios)           # [12. 16. 22.]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    # print(anchors)
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(base_size=16,
                     ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    '''generate anchor window by enumerating aspect ratios
    X scales wrt a reference (0, 0, 15, 15) window
    '''
    base_anchors = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchors, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i,:], scales)
                            for i in range(ratio_anchors.shape[0])])
    return anchors

if __name__ == '__main__':
    a = generate_anchors()
    print(a)