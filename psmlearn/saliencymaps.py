from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# http://stackoverflow.com/questions/38340791/guided-back-propagation-in-tensor-flow
def guided_backprop_op(fn, relus, X):
    assert len(relus)>0, "no relus"
    assert len(fn.get_shape())==0
    oplist = [X] + [op for op in relus]
    next_relu = oplist.pop()
    Dafter = tf.gradients(fn, next_relu)[0][0]
    zeros = tf.zeros(Dafter.get_shape())
    Dafter_thresh= tf.select(Dafter < 0.0, zeros, Dafter)

    while len(oplist):
        last_relu = next_relu
        next_relu = oplist.pop()
        Dafter = tf.gradients(last_relu[0], next_relu, grad_ys=Dafter_thresh)[0]
        zeros = tf.zeros(Dafter.get_shape())
        Dafter_thresh = tf.select(Dafter < 0.0, zeros, Dafter)
    return Dafter

class SaliencyMap(object):
    def __init__(self, model):
        self.model = model
        self.sess = model.sess
        self.img_processed = model.X_processed
        self.img_raw = model.X_placeholder
        self.logit2bprop = {}
        self.logit2gbprop = {}
        for logit in range(model.nnet.logits.get_shape()[1]):
            print(logit)
#            import IPython
#            IPython.embed()
            self.logit2bprop[logit] = tf.gradients(model.nnet.logits[:,logit][0], self.img_processed)[0]
            self.logit2gbprop[logit] = guided_backprop_op(Y=model.nnet.logits[:,logit][0],
                                                          relus=model.nnet.after_relus,
                                                          X=self.img_processed)
    def calc(self, raw_img, logit, fn='gbprop'):
        assert len(raw_img.shape)==4
        assert raw_img.shape[0]==1, "only do batch size of 1"
        assert fn in ['bprop', 'gbprop'], "fn must be one of bprop or gpprop, but it is %s" % fn
        ops = [self.img_processed]
        feed_dict = {self.img_raw:raw_img}
        if fn == 'bprop':
            ops.append(self.logit2bprop[logit])
            img_processed, dimg = self.sess.run(ops, feed_dict=feed_dict)
        elif fn == 'gbprop':
            assert False
        assert len(img_processed.shape)==4
        assert img_processed.shape[0]==1
        img_processed = img_processed[0,:,:,:]
        
        assert isinstance(dimg, list)
        assert len(dimg)==1
        dimg = dimg[0]

        assert len(dimg.shape)==4
        assert dimg.shape[0]==1
        dimg = dimg[0,:,:,:]

        return img_processed, dimg
    
