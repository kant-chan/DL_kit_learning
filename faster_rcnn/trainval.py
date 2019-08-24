import _init_path
import argparse
import numpy as np
from pprint import pprint

# about torch
import torch
import torchvision
import torch.nn as nn

# custom module
from model.utils.config import cfg, cfg_from_file, cfg_from_list

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
        dest='cuda'
        help='whether use CUDA',
        action='store_true')

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

    pprint("******** config ********")
    pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    if torch.cuda.is_available() and not args.cuda:
        print('WARNING: CUDA is available, run with --cuda')

    cfg.USE_GPU_NMS = args.cuda

    