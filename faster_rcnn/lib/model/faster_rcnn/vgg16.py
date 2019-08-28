import torch
import torchvision

from model.faster_rcnn.faster_rcnn import _fasterRCNN


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)
    
    def _init_modules(self):
        vgg = torchvision.models.vgg16()
        if self.pretrained:
            print('Loading pretrained weights from {}'.format(self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})