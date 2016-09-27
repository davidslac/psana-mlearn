from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Model(object):
    def __init__(self, **kwargs):
        self.X_placeholder = kwargs.pop('X_placeholder')
        self.Y_placeholder = kwargs.pop('Y_placeholder')
        self.trainflag_placeholder = kwargs.pop('trainflag_placeholder')
        self.X_processed = kwargs.pop('X_processed')
        self.nnet        = kwargs.pop('nnet')
        self.train_ops   = kwargs.pop('train_ops')
        self.predict_op  = kwargs.pop('predict_op')
        self.sess        = kwargs.pop('sess')
        self.saver       = kwargs.pop('saver')

