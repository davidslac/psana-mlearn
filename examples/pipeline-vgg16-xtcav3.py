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
import numpy as np
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
        self._vgg16 = None

    def add_arguments(self, parser):
        parser.add_argument('--num_batches',type=int,help='debugging, fewer batches', default=0)

    def init(self, config, pipeline):
        '''pipeline will have set the random seeds for python and numpy, 
        '''
        self.doTrace = pipeline.doTrace
        dset = psmlearn.get_dataset('xtcav', X='img', Y='enPeak', verbose=True)
        dset.split(train=90, validation=5, test=5, seed=config.seed)
        self.dset = dset

    def vgg16(self):
        if self._vgg16 is None:
            vgg16_weights_file = psmlearn.dataloc.getProjectCalibFile(project='vgg16', fname='vgg16_weights.npz')
            self._vgg16 = psmlearn.vgg16.create(session=pipeline.session, 
                                                weights=vgg16_weights_file)
        return self._vgg16
                                           
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
        num_outputs = 0
        for X,Y,meta,batchinfo in basic_iter:
            if num_outputs==0:
                num_outputs = Y[0].shape[1]
            img_batch = X[0]
            img = img_batch[0,:,:]
            prep_img = self.prepare_for_vgg16(img, config=prep_config, channel_mean=0)
            means.append(np.mean(prep_img[:,:,0]))
            pipeline.debug("compute_channel_mean: %d" % len(means), checkcache=False)
        h5=h5py.File(output_files[0],'w')
        h5['channel_mean'] = np.mean(np.array(means))
        h5['number_samples_train'] = len(means)
        h5['num_outputs'] = num_outputs
        h5['files'] = basic_iter.get_h5files()

        # get number of validation and test samples
        basic_iter = self.dset.validation_iter(batchsize=1, epochs=1, num_batches=config.num_batches)
        validation_num=0
        for batch in basic_iter: validation_num+=1

        basic_iter = self.dset.test_iter(batchsize=1, epochs=1, num_batches=config.num_batches)
        test_num=0
        for batch in basic_iter: test_num += 1
        
        h5['number_samples_validation'] = validation_num
        h5['number_samples_test'] = test_num
        pipeline.trace("compute_channel_mean: finished", checkcache=False)

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
        data_stats = h5util.read_from_h5(step2h5list['compute_channel_mean'][0])
        channel_mean = data_stats['channel_mean']
        imgprep_config = pipeline.get_config('prepare_for_vgg16')

        assert output_files[0].endswith('_train.h5')
        assert output_files[1].endswith('_validation.h5')
        assert output_files[2].endswith('_test.h5')

        iterList = [('train',
                     output_files[0],
                     self.dset.train_iter(batchsize=1,
                                          epochs=1,
                                          num_batches=config.num_batches)),
                    ('validation',
                     output_files[1],
                     self.dset.validation_iter(batchsize=1,
                                               epochs=1,
                                               num_batches=config.num_batches)),
                    ('test',
                     output_files[2],
                     self.dset.test_iter(batchsize=1,
                                         epochs=1,
                                         num_batches=config.num_batches))
        ]

        for splitFnameIter in iterList:
            split, fname, batch_iter = splitFnameIter
            datalen = data_stats['number_samples_%s' % split]
            h5out = h5py.File(fname, 'w')
            idx = 0
            all_meta = None
            all_Y = None
            for Xlist,Ylist,meta,batchinfo in batch_iter:
                img_batch = Xlist[0]
                img = img_batch[0]
                Y=Ylist[0]
                if all_meta is None:
                    all_meta = np.zeros(datalen, dtype=meta.dtype)
                if all_Y is None:
                    shape = [datalen]
                    if len(Y.shape)>1:
                        shape += list(Y.shape[1:])
                    all_Y = np.zeros(shape, dtype=Y.dtype)
                all_meta[idx]=meta
                all_Y[idx,:]=Y
                prep_img = self.prepare_for_vgg16(img, imgprep_config, channel_mean=channel_mean)
                prep_img = np.reshape(prep_img, [1]+list(prep_img.shape))
                fc1, fc2 = self.vgg16().get_model_layers(sess=pipeline.session,
                                                         imgs=prep_img,
                                                         layer_names=['fc1','fc2'])
                if idx == 0:
                    shape = (datalen, fc1.shape[1] + fc2.shape[1])
                    h5out.create_dataset('codewords', shape=shape, dtype=fc2.dtype)
                h5out['codewords'][idx,0:fc1.shape[1]] = fc1[:]
                h5out['codewords'][idx,fc1.shape[1]:] = fc2[:]
                idx += 1
                pipeline.debug("compute_codewords: split=%s idx=%d" % (split,idx), checkcache=False)
            h5out['meta']=all_meta
            h5out['Y'] = all_Y
            h5out['files']=batch_iter.get_h5files()
        pipeline.trace("compute_codewords: done with all", checkcache=False)
        
    def tsne_on_vgg16_codewords(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')

    def get_num_features(self, step2h5list):
        fname = step2h5list['compute_vgg16_codewords'][0]
        h5=h5py.File(fname,'r')
        return h5['codewords'].shape[1]

    def get_num_outputs(self, step2h5list):
        data_stats = h5util.read_from_h5(step2h5list['compute_channel_mean'][0])
        return data_stats['num_outputs']
        
    def train_on_codewords(self, config, pipeline, step2h5list, output_files):
        num_outputs = self.get_num_outputs(step2h5list)
        num_features = self.get_num_features(step2h5list)
        model = psmlearn.models.LinearClassifier(num_features=num_features,
                                                 num_outputs=num_outputs,
                                                 config=config)

        h5files_train = [step2h5list['compute_vgg16_codewords'][0]]
        h5files_validation = [step2h5list['compute_vgg16_codewords'][1]]

        train_iter = self.dset.iter_from(h5files=h5files_train,
                                         X=['codewords'], Y=['Y'],
                                         batchsize=config.train_batchsize)

        validation_iter = self.dset.iter_from(h5files=h5files_validation,                                              
                                              X=['codewords'], Y=['Y'],
                                              batchsize=config.train_batchsize)

        trainer = psmlearn.ClassificationTrainer(train_iter=train_iter,
                                                 validation_iter=validation_iter,
                                                 config=config,
                                                 sess=pipeline.session,
                                                 save_fname=output_files[0],
                                                 force=config.force,
                                                 model=model)
        trainer.run()

    def activations(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')
        return
        num_outputs = self.get_num_outputs(step2h5list)
        num_features = self.get_num_features(step2h5list)
        model = psmlearn.models.LinearClassifier(num_features=num_features,
                                                 num_outputs=num_outputs,
                                                 config=config)
        model.restore_from_fname(step2h5list['train_on_codewords'][0], pipelines.session)
        W,B = model.get_W_B(pipeline.session)
        h5files = step2h5list['compute_vgg16_codewords'][0:2]
        dataiter = self.dset.iter_from(h5files=h5files_train,
                                       X=['codewords'], Y=['Y'],
                                       batchsize=config.batchsize)
        
        allmeta = []
        for Xlist, Ylist, meta, batchinfo in dataiter:
            if batchinfo['epoch']==1: break
            X=Xlist[0]
            Y=Ylist[0]
            allmeta.append(meta)
            logits = np.matmul(X,W)+B
            import IPython
            IPython.embed()
        
    def gbprop(self, config, pipeline, step2h5list, output_files):
        
        h5=h5py.File(output_files[0],'w')
    
    
### pipeline ###########
if __name__ == '__main__':
    stepImpl = XtcavVgg16()
    outputdir = psmlearn.dataloc.getDefalutOutputDir(project='xtcav')
    pipeline = Pipeline(stepImpl=stepImpl, outputdir=outputdir)
    stepImpl.add_arguments(pipeline.parser)
    pipeline.add_step_method(name='compute_channel_mean')
    pipeline.add_step_method_plot(name='plot_vgg16_img_prep')
    pipeline.add_step_method(name='tsne_on_img_prep')
    pipeline.add_step_method(name='compute_vgg16_codewords',
                             output_files=['_train','_validation','_test'])
    pipeline.add_step_method(name='tsne_on_vgg16_codewords')
    pipeline.add_step_method(name='train_on_codewords')
    pipeline.add_step_method(name='activations')
    pipeline.add_step_method(name='gbprop')
    pipeline.init()
    pipeline.run()
