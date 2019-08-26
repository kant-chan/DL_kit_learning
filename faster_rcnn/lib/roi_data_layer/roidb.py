#coding=utf-8
import numpy as np
import PIL

# custom module
from model.utils.config import cfg
from datasets.factory import get_imdb


def prepare_roidb(imdb):
    '''Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    '''
    roidb = imdb.roidb
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in range(imdb.num_images)] # size -> w, h
    
    for i in range(len(imdb.image_index)):
        # add new attributes
        '''
        {
            'boxes': (num_objs, 4),
            'gt_classes': (num_objs),
            'gt_overlaps': (num_objs, num_classes),
            'seg_areas': (num_objs),
            'gt_ishards': (num_objs),


            'img_id': i,
            'image': image_path,
            'width': image original width,
            'height': image original height,
            'max_overlaps': ,
            'max_classes':
        }
        '''
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_overlaps'] = max_overlaps
        roidb[i]['max_classes'] = max_classes

        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height
    ratio_large = 2
    ratio_small = 0.5
    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)
        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box
    print('Before filtering, there are {} images...'.format(len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0:
            del(roidb[i])
            i -= 1
        i += 1
    
    print('After filtering, there are {} images...'.format(len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):
    '''combine multiple roidb
    '''

    def get_training_roidb(imdb):
        '''Return a roidb for use in training
        '''
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')
        print('Preparing training data...')

        prepare_roidb(imdb)
        print('done')
        return imdb.roidb

    def get_roidb(imdb_name):
        # imdb is the instance of pascal_voc(derive from imdb) class
        imdb = get_imdb(imdb_name)
        print('Loaded dataset {} for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # 'gt'
        print('Set proposal method: {}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidb = get_roidb(imdb_names)

    imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)
    
    return imdb, roidb, ratio_list, ratio_index
