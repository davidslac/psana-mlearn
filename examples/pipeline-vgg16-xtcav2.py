from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import random
import tensorflow as tf
from scipy.misc import imresize
from psmlearn.pipeline import Pipeline
import psmlearn.h5util as h5util
import psmlearn.util as util
from psmlearn import tsne
import psmlearn
import h5py
from h5minibatch import H5BatchReader

class XtcavVgg16(object):
    def __init__(self):
        '''Step implementation for pipeline, doing xtcav analysis using transfer learning with 
        vgg16. Don't do any initialization in __init__, wait until command line args are parsed
        and entire config can be created - use init
        '''
        pass

    def add_arguments(self, parser):
        # we'll get everything from the config file
        pass

    def init(self, config):
        '''pipeline will have set the random seeds for python and numpy, 
        '''
        dset = psmlearn.get_dataset('xtcav', X='img', Y='enPeak', verbose=True)
        dset.split(train=90, validation=5, test=5, seed=config.seed)
        self.dset = dset
        
    def prepare_for_vgg16(self, img, config, channel_mean=None):
        img = img.astype(np.float32)
        thresh = config.thresh
        util.logTrace('prepare_for_vgg16', 'thresh=%.1f' % thresh)
        util.replaceWithLogIfAbove(img, thresh)
        img = imresize(img,(224,224), interp='lanczos', mode='F')
        if channel_mean: img -= channel_mean
        rgb = util.replicate(img, numChannels=3, dtype=np.float32)
        return rgb

    
    def tsne_cws(self, config, pipeline, step2h5list, output_files):
        '''run tsne on cws for the images that we run before 
        '''
        initial_dims = config.initial_dims
        perplexity = config.perplexity
        meta_ky2data = h5util.read_meta(step2h5list['tsne_imgs'][0])
        h5py.File(step2h5list['tsne_imgs'][0],'r')['labels'][:] fc2 =
        h5util.match_meta(meta_ky2data, 'fc2',
        step2h5list['model_layers'][0]) util.logTrace(hdr='tsne_cws',
        msg="about to run tsne on data.shape=%s perplexity=%.1f
        initial_dims=%d using first %d comp of fc2" % \ (fc2.shape,
        perplexity, initial_dims, config['tsnecomp'])) Y =
        tsne(X=fc2[:,0:config['tsnecomp']], no_dims=2,
        initial_dims=initial_dims, perplexity=perplexity) res =
        {'tsne_cws':Y, 'labels':labels}
        h5util.write_to_h5(output_files[0], datadict=res,
        meta=h5util.convert_meta(meta_ky2data))

def plot_tsne_cws(config, pipeline, plot, plotFigH, step2h5list):
    cws = h5py.File(step2h5list['tsne_cws'][0],'r')['tsne_cws'][:]
    assert cws.shape[1]==2
    assert len(cws.shape)==2
    labels = h5py.File(step2h5list['tsne_cws'][0],'r')['labels'][:]
    plt = pipeline.plt
    plt.figure(plotFigH, figsize=(12,12))
    plt.scatter(cws[:,0], cws[:,1], 20, labels)
    plt.legend()
    print (labels)
    plt.pause(.1)
    raw_input('hit enter')

# not hooked in yet
def train_classifier(data_iter, config, pipeline, step2h5list, output_files):
    '''cross entropy loss to predict 0 or 1 labels

    input: take the 8192 fc1 and fc2, do one layer of 100, then two
    '''
    sess = pipeline.session
    num_labels = h5py.File(step2h5list['data_stats'][0],'r')['num_labels'].value

    input_pl = tf.placeholder(tf.float32, shape=(None, 8192), name='cws')
    labels_pl = tf.placeholder(tf.float32, shape=(None, num_labels), name='labels')

    W1 = tf.Variable(tf.truncated_normal([8192, 128], mean=0.0, stddev=0.03))
    B1 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[128]))
    xw_plus_b1 = tf.nn.xw_plus_b(input_pl, W,B)
    nonlinear1 = tf.nn.relu(xw_plus_b)

    W2 = tf.Variable(tf.truncated_normal([128, num_labels], mean=0.0, stddev=0.1))
    B2 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[num_labels]))
    logits =  tf.nn.xw_plus_b(nonlinear1, W2, B2)

    cross_entropy_loss_all = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                     labels_pl)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss_all)

    reg_term = 0.01 * (tf.reduce(W1*W1) + tf.reduce(W2*W2))
    loss = cross_entropy_loss + reg_term

    optimizer = tf.train.MomentumOptimizer(learning_rate=config['learning_rate'], 
                                           momentum=config['optimizer_momentum'])
    train_op = self.optimizer.minimize(loss)
    # psmlearn doesn't have a batchIter yet
    dataiter = psmlearn.batchIter(h5files = [step2h5list['model_layers']],
                                  X=['fc1','fc2'],
                                  Y=['labels'],
                                  batchsize=64)
    
    psmlearn.train(train_op, dataiter)
    
### pipeline ###########
if __name__ == '__main__':
    stepImpl = XtcavVgg16()
    pipeline = Pipeline(stepImpl=stepImpl,
                        outputdir='.')
    stepImpl.add_arguments(pipeline.parser)
    
    pipeline.add_step(name='data_stats', pipeline.channelMean),
                    ('model_layers',pipeline.vgg16codewords),
                    (
    config = pipeline.getConfig()
    
    
pipeline.parser.add_argument('--tsnecomp', type=int, help='for tsne, number of components of the codewords to do', default=200) 
pipeline.parser.add_argument('--perplexity', type=float, help='for tsne, default=30.0', default=30.0) 
pipeline.parser.add_argument('--initial_dims', type=int, help='for tsne, default=50', default=30) 
#pipeline.run()
