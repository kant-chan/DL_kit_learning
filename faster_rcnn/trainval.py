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
from torch.utils.data import DataLoader

# custom module
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatch_loader import RoibatchLoader
from model.faster_rcnn.vgg16 import vgg16
# from model.faster_rcnn.resnet import resnet
from model.utils.net_utils import clip_gradient

from model.utils.visualize import display, vis_gen_anchors, vis_proposals

###### test ######
DEBUG = True
##################


class sampler(Sampler):
    '''This is a iterator
    train_size: 10022, batch_size: 16, train_size/batch_size: 626.375
    '''
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        # self.long() is equivalent to self.to(torch.int64)
        # [[0, 1, 2, ..., batch_size-1]]
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_batch, self.batch_size) + self.range
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
    parser.add_argument('--class_agnostic',
        dest='class_agnostic',
        help='whether perform class_agnostic bbox regression',
        action='store_true')
    parser.add_argument('--lr',
        dest='lr',
        help='starting learning rate',
        default=0.001,
        type=float)
    parser.add_argument('--lr_decay_step',
        dest='lr_decay_step',
        help='step to do learning rate decay, unit is epoch',
        default=5, type=int)
    parser.add_argument('--lr_decay_gamma',
        dest='lr_decay_gamma',
        help='learning rate decay ratio',
        default=0.1,
        type=float)
    parser.add_argument('--o',
        dest='optimizer',
        help='training optimizer',
        default='sgd',
        type=str)
    parser.add_argument('--epochs',
        dest='max_epochs',
        help='number of epochs to train',
        default=20,
        type=int)
    parser.add_argument('--disp_interval',
        dest='disp_interval',
        help='number of iterations to display',
        default=100,
        type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # config args
    if args.dataset == 'pascal_voc':
        args.imdb_name = 'voc_2007_trainval'
        args.imdbval_name = 'voc_2007_test'
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5, 1, 2]',
                         'MAX_NUM_GT_BOXES', '20']
    
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    dataset = RoibatchLoader(roidb,
                             ratio_list,
                             ratio_index,
                             args.batch_size,
                             imdb.num_classes,
                             training=True)
    
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler_batch,
                            num_workers=args.num_workers)

    if args.cuda:
        cfg.CUDA = True

    if DEBUG:
        '''
        im_data:
            size --> torch.Size([batch_size, 3, 600, 904])
        im_info:
            tensor([[600.0000, 904.0000, 1.8072],
                    [600.0000, 904.0000, 1.8072],
                    ...
                    batch_size
                    ...]])
        gt_boxes:
            size --> torch.Size([batch_size, 20, 5])
        num_boxes:
            tensor([ 4,  2, 13,  1,  2,  1,  2,  4,  4,  7])
        '''
        print('==========> test')
        # print(dataset.data_size, dataset.batch_size)
        # print(dataset.ratio_list_batch[:30])
        for i, v in enumerate(dataloader):
            if i == 0:
                print("padding_data size:", v[0].size()) # padding_data
                print("im_info:", v[1])        # im_info
                print("gt_boxes:", v[2].size())        # gt_boxes
                print("num_boxes:", v[3])        # num_boxes

                ############
                print('im_data as numpy shape:', v[0][0].numpy().shape, type(v[0].size(0)))
                print('[]', type(v[3][0].item()))
                # display(v[0], v[2], v[3])
                # vis_gen_anchors(v[0])
                ############
                # print("ratio:", v[4])
                break
        # for i, sam in enumerate(sampler_batch):
        #     if i == 0:
        #         print(sam, sam.size())
        #         break
        print('==========> test end')

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    # elif args.net == 'res101':
    #     fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    # elif args.net == 'res50':
    #     fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    # elif args.net == 'res152':
    #     fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    # else:
    #     print('network is not defined')

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{
                    'params': [value],
                    'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0,
                }]
            else:
                params += [{
                    'params': [value],
                    'lr': lr,
                    'weight_decay': cfg.TRAIN.WEIGHT_DECAY
                }]

    if args.cuda:
        fasterRCNN.to(device)
    else:
        fasterRCNN.to(device)

    if args.optimizer == 'adam':
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.max_epochs + 1):

        loss_temp = 0

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)

            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            data[3] = data[3].to(device)

            fasterRCNN.zero_grad()

            if DEBUG:
                print('********* {} *********'.format(step))
                print(data[0].size())
                print(data[1])
                print(data[2].size())
                print(data[3])
            # forward pipeline:
            #   faster_rcnn --> rpn --> proposal_layer --> anchor_target_layer
            rois, rpn_loss_cls, rpn_loss_box = fasterRCNN(data[0], data[1], data[2], data[3])

            print('**************')
            print(rois.size())
            # vis_proposals(data[0], rois)

            break
        break
            # loss = rpn_loss_cls.mean() + rpn_loss_box.mean()
            # loss_temp += loss.item()

            # optimizer.zero_grad()
            # loss.backward()
            # if args.net == 'vgg16':
            #     clip_gradient(fasterRCNN, 10.0)
            # optimizer.step()

            # if step % args.disp_interval == 0:
            #     if step > 0:
            #         loss_temp /= (args.disp_interval + 1)

            #     loss_rpn_cls = rpn_loss_cls.item()
            #     loss_rpn_box = rpn_loss_box.item()

            #     print('[epoch {:2d}][iter {:4d}/{:4d}] loss: {:.4f}, lr: {:.2e}'.format(
            #         epoch, step, iters_per_epoch, loss_temp, lr))

            #     loss_temp = 0
