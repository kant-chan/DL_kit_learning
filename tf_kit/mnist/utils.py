# -*- coding: utf-8 -*-

import numpy as np
import struct
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

train_images_file = './data/train-images.idx3-ubyte'
train_labels_file = './data/train-labels.idx1-ubyte'
test_images_file = './data/t10k-images.idx3-ubyte'
test_labels_file = './data/t10k-labels.idx1-ubyte'

# to generate batch size data --> labels(batch_size, ), images(batch_size, w, h)
class Batch(object):
    def __init__(self, file_path_img, file_path_label, batch_size):
        self.file_path_img = file_path_img
        self.file_path_label = file_path_label
        self.batch_size = batch_size
        self.images = decode_idx3_ubyte(self.file_path_img, self.batch_size)
        self.labels = decode_idx1_ubyte(self.file_path_label, self.batch_size)
    def next_batch(self):
        return next(self.images), next(self.labels)
    def next_batch_shuffle(self):
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        images = next(self.images)
        labels = next(self.labels)
        return images.take(indices, axis=0), labels.take(indices)

def decode_idx3_ubyte(file_path, batch_size):
    with open(file_path, 'rb') as fo:
        bin_data = fo.read()
        # 解析文件头信息 魔数 图片数量 图片高 图片宽
        offset = 0
        fmt_header = '>iiii'
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数: %d, 图片数量: %d, 图片高: %d, 图片宽: %d' % (magic_number, num_images, num_rows, num_cols))

        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        for i in range(0, num_images, batch_size):
            images = np.empty((batch_size, num_rows, num_cols))
            for j in range(i, min(i+batch_size, num_images)):
                images[j-i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((28, 28))
                offset += struct.calcsize(fmt_image)
            yield images

def decode_idx1_ubyte(file_path, batch_size):
    with open(file_path, 'rb') as fo:
        bin_data = fo.read()
        # 解析文件头信息 魔数 标签数量
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数: %d, 标签数量: %d' % (magic_number, num_images))

        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        for i in range(0, num_images, batch_size):
            labels = np.empty(batch_size)
            for j in range(i, min(i+batch_size, num_images)):
                labels[j-i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
                offset += struct.calcsize(fmt_image)
            yield labels

def show_image():
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    images = decode_idx3_ubyte(test_images_file, 10)
    for ims in images:
        for i in range(ims.shape[0]):
            ax[i].imshow(ims[i], cmap='Greys', interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
        break

def show_label():
    labels = decode_idx1_ubyte(test_labels_file, 10)
    for lbs in labels:
        print(lbs)
        break

if __name__ == '__main__':
    # show_image()
    # show_label()

    # ims = decode_idx3_ubyte(test_images_file, 5)
    # print(next(ims).shape)
    # print(next(ims).shape)
    # lbs = decode_idx1_ubyte(test_labels_file, 5)
    # print(next(lbs))
    # print(next(lbs))

    b1 = Batch(test_images_file, test_labels_file, 10)
    for i in range(1):
        i, l = b1.next_batch()
        i_re = i.reshape((10, -1))
        print(l)
        print(i_re.shape)