from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class SaliencyMap(object):
    def __init__(self, model):
        self.model = model
        self.img_processed = model.X_processed
        relus_from_logits = []
        
    def from_logit(raw_img, logit_idx, fn='gbprop'):
        start_node = model.nnet.logits[logit_idx]
        end_node = self.img_processed
