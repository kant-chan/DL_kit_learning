# coding=utf-8
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transforms = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        # left, upper, right, lower
        img_a = img.crop((0, 0, w / 2, h))
        img_b = img.crop((w / 2, 0, w, h))
        # random flip?
        if np.random.random() < 0.5:
            img_a = Image.fromarray(np.array(img_a)[:, ::-1, :], "RGB")
            img_b = Image.fromarray(np.array(img_b)[:, ::-1, :], "RGB")
        img_a = self.transforms(img_a)
        img_b = self.transforms(img_b)
        return {'img_a': img_a, 'img_b': img_b}

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    file_path = '../../data/facades/train/1.jpg'
    im = Image.open(file_path)
    w, h = im.size
    mode = im.mode
    print(w, h, mode)
    im_np = np.array(im)
    print(im_np.shape)