#coding=utf-8
import numpy as np
import PIL

# custom module
from model.utils.config import cfg


def combined_roidb(imdb_names, training=True):
    '''
    combine multiple roidb
    '''
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)