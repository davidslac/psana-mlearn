from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import cPickle

from . import util

class TensorFlowTrainer(object):
    def __init__(self, args, **kwargs):
        self.args = args
        self.validation_feed_dict = kwargs.pop('validation_feed_dict')
        self.model = kwargs.pop('model')
        self.datareader = kwargs.pop('datareader')
        self.print_cmat = kwargs.pop('print_cmat')
        
        valid_samples = 0
        for arr in self.validation_feed_dict.values():
            try:
                valid_samples = max(valid_samples, arr.shape[0])
            except:
                pass
        assert valid_samples > 0, "couldn't get number of validation samples"
        # get decimal places needed to format confusion matrix
        self.fmtLen = int(math.ceil(math.log(valid_samples,10)))

        self.best_accuracy=0.0
        self.wrote_pickle = False
        
    def save_model(self):
        assert self.args.force or not os.path.exists(self.args.train_save), "model output %s exists, use force to overwrite" % self.args.train_save
        self.model.saver.save(self.model.sess, self.args.train_save)
        if not self.wrote_pickle:
            self.wrote_pickle=True
            fout = file(self.args.train_save + '.pckl','w')
            cPickle.dump(self.args, fout)
            
    def eval(self, msg, feed_dict):
        t0 = time.time()
        train_eval_ops = [self.model.predict_op, self.model.nnet.loss, self.model.nnet.opt_loss]
        train_predict, loss, opt_loss = self.model.sess.run(train_eval_ops, feed_dict=feed_dict)
        eval_predict = self.model.sess.run(self.model.predict_op, feed_dict=self.validation_feed_dict)
        train_cmat = util.get_confusion_matrix_one_hot(train_predict, feed_dict[self.model.Y_placeholder])
        valid_cmat = util.get_confusion_matrix_one_hot(eval_predict, self.validation_feed_dict[self.model.Y_placeholder])
        train_acc, train_cmat_rows = util.cmat2str(train_cmat, self.fmtLen)
        valid_acc, valid_cmat_rows = util.cmat2str(valid_cmat, self.fmtLen)

        saved = False
        if valid_acc > self.best_accuracy:
            self.best_accuracy = valid_acc
            self.save_model()
            saved = True
        tm = time.time()-t0
        msg += ' ev.tm=%.1f' % tm
        msg += ' loss=%.3e' % loss
        msg += ' opt_loss=%.3e' % opt_loss
        msg += ' tr.acc=%.2f' % train_acc
        msg += ' ev.acc=%.2f' % valid_acc
        
        if not self.print_cmat:
            if saved:
                msg += ' * saved in %s' % args.train_save
            print(msg)
            return
        
        N = len(msg)
        msg += ' train.cmat=%s' % train_cmat_rows.pop(0)
        msg += '  valid.cmat=%s' % valid_cmat_rows.pop(0)
        if saved:
            msg += ' * saved in %s' % self.args.train_save
        for tr_row, vl_row in zip(train_cmat_rows, valid_cmat_rows):
            msg += '\n'
            msg += ' '*N
            msg += '            %s' % tr_row
            msg += '             %s' % vl_row
        print(msg)
        
    def train(self):
        print("Starting training.")
        sys.stdout.flush()

        for step in range(self.args.train_steps):
            t0 = time.time()
            X, Y = self.datareader.get_next_minibatch()
            feed_dict = {self.model.X_placeholder:X,
                         self.model.Y_placeholder:Y,
                         self.model.trainflag_placeholder:True}
            self.model.sess.run(self.model.train_ops, feed_dict=feed_dict)
            tm = time.time()-t0
            msg = 'step=%3d tm=%.2f' % (step, tm)
            if step % self.args.eval_steps==0 and step > 0:
                self.eval(msg, feed_dict)
            else:
                print(msg)
            sys.stdout.flush()


