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
import psmlearn.plot as psplot
import psmlearn
import h5py
from h5minibatch import H5BatchReader

class XtcavVgg16(object):
    def __init__(self):
        '''Step implementation for pipeline, doing xtcav analysis using transfer learning with 
        vgg16. Don't do any initialization in __init__, wait until command line args are parsed
        and entire config can be created - use init
        '''
        self.hdr='XtcavVgg16'
        self.doTrace = False
        
    def add_arguments(self, parser):
        parser.add_argument('--num_batches',type=int,help='debugging, fewer batches', default=0)

    def init(self, config, pipeline):
        '''pipeline will have set the random seeds for python and numpy, 
        '''
        self.doTrace = pipeline.doTrace
        dset = psmlearn.get_dataset('xtcav', X='img', Y='enPeak', verbose=True)
        dset.split(train=90, validation=5, test=5, seed=config.seed)
        self.dset = dset
        
    def prepare_for_vgg16(self, img, config, channel_mean=None):
        prep_img = img.astype(np.float32)
        thresh = config.thresh
        util.logTrace('prepare_for_vgg16', 'thresh=%.1f' % thresh)
        util.replaceWithLogIfAbove(prep_img, thresh)
        prep_resized = imresize(prep_img,(224,224), interp='lanczos', mode='F')
        
        if channel_mean: prep_resized -= channel_mean
        rgb = util.replicate(prep_resized, numChannels=3, dtype=np.float32)
        return rgb

    def compute_channel_mean(self, config, pipeline, step2h5list, output_files):
        prep_config = pipeline.get_config(name='prepare_for_vgg16')
        basic_iter = self.dset.train_iter(batchsize=1, epochs=1, num_batches=config.num_batches)
        means = []
        for X,Y,meta,batchinfo in basic_iter:
            img_batch = X[0]
            img = img_batch[0,:,:]
            prep_img = self.prepare_for_vgg16(img, config=prep_config, channel_mean=0)
            means.append(np.mean(prep_img[:,:,0]))
        h5=h5py.File(output_files[0],'w')
        h5['channel_mean'] = np.mean(np.array(means))
        h5['number_samples'] = len(means)

    def plot_vgg16_img_prep(self, plot, pipeline, plotFigH, config, step2h5list):
        prep_config = pipeline.get_config(name='prepare_for_vgg16')
        channel_mean = h5util.read_from_h5(step2h5list['compute_channel_mean'][0])['channel_mean']
        util.logTrace("plot_vgg16_img_prep", "channel_mean is %.2f" % channel_mean)
        basic_iter = self.dset.train_iter(batchsize=1, epochs=1, num_batches=config.num_batches)
        plt = pipeline.plt
        plt.figure(plotFigH)
        plt.clf()
        for X,Y,meta,batchinfo in basic_iter:
            img_batch = X[0]
            img = img_batch[0,:,:]
            prep_img = self.prepare_for_vgg16(img, config=prep_config, channel_mean=channel_mean)
            psplot.compareImages(plt, plotFigH, ("orig",img), ("vgg16 prep",prep_img))
            if pipeline.stop_plots(): break
            
    def tsne_on_img_prep(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')
    
    def compute_vgg16_codewords(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')
    
    def tsne_on_vgg16_codewords(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')
    
    def train_on_codewords(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')
    
    def gbprop(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')
    
    
### pipeline ###########
if __name__ == '__main__':
    stepImpl = XtcavVgg16()
    pipeline = Pipeline(outputdir='.', inst=stepImpl)
    stepImpl.add_arguments(pipeline.parser)
    pipeline.add_step_method(name='compute_channel_mean')
    pipeline.add_step_method_plot(name='plot_vgg16_img_prep')
    pipeline.add_step_method(name='tsne_on_img_prep')
    pipeline.add_step_method(name='compute_vgg16_codewords')
    pipeline.add_step_method(name='tsne_on_vgg16_codewords')
    pipeline.add_step_method(name='train_on_codewords')
    pipeline.add_step_method(name='gbprop')
    pipeline.init()
    pipeline.run()
