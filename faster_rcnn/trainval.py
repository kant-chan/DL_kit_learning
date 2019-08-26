import _init_path
import argparse
import numpy as np
from pprint import pprint
import os

# about torch
import torch
import torchvision
import torch.nn as nn
from torch.utils.data.sampler import Sampler


# custom module
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatch_loader import RoibatchLoader
from torch.utils.data import DataLoader

###### test ######

##################


class sampler(Sampler):
    '''train_size: 10022, batch_size: 16, train_size/batch_size: 626.375
    '''
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        # self.long() is equivalent to self.to(torch.int64)
        # [[0, 1, 2, ..., batch_size-1]]
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        self.rand_num_view = self.rand_num.view(-1)
        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)
        return iter(self.rand_num_view)
    
    def __len__(self):
        return self.num_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        dest='dataset',
        help='train a dataset',
        default='pascal_voc',
        type=str)
    parser.add_argument('--net',
        dest='net',
        help='vgg16 or resnet101',
        default='vgg16',
        type=str)
    parser.add_argument('--cuda',
        dest='cuda',
        help='whether use CUDA',
        action='store_true')
    parser.add_argument('--save_dir',
        dest='save_dir',
        help='directory to save models',
        default='models',
        type=str)
    parser.add_argument('--batch_size',
        dest='batch_size',
        help='batch_size',
        default=1,
        type=int)
    parser.add_argument('--num_workers',
        help='number of worker to load data',
        default=0,
        type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # config args
    if args.dataset == 'pascal_voc':
        args.imdb_name = 'voc_2007_trainval'
        args.imdbval_name = 'voc_2007_test'
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '20']
    
    args.cfg_file = 'cfgs/{}.yml'.format('vgg16')

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("******** config ********")
    pprint(cfg)
    print('************************')

    np.random.seed(cfg.RNG_SEED)

    if torch.cuda.is_available() and not args.cuda:
        print('WARNING: CUDA is available, run with --cuda')

    cfg.USE_GPU_NMS = args.cuda

    # imdb and roidb is not same instance
    # example: imdb.num_images -> 5011
    #          len(roidb) -> 10022
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{} roidb entries'.format(train_size))

    # save models dir
    # example: models/vgg16/pascal_voc
    output_dir = args.save_dir + '/' + args.net + '/' + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = RoibatchLoader(roidb, ratio_list, ratio_index, \
        args.batch_size, imdb.num_classes, training=True)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

    for i in dataloader:
        print('fuuuuuuuuuuuuuuuuuuk')
    

    ########## test
    # sample_batch = sampler(10022, 16)
    # for i, sam in enumerate(sample_batch):
    #     if i < 2:
    #         print(sam, sam.size())
    ###############