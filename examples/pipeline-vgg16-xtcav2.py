from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import tensorflow as tf
from scipy.misc import imresize
from psmlearn.pipeline import Vgg16Pipeline
import psmlearn.h5util as h5util
import psmlearn.util as util
from psmlearn import tsne
import psmlearn
import h5py
from h5minibatch import H5BatchReader

##########
def data_gen(randomize=False, seed=109, args=None, num=10,
             feats=['imgs'], labels_2_return=['signal','box']):
    '''
    a data generator returns an iterator that goes once through a datasource.
    The parameters that pipeline will use are
    
    randomize - shuffle the data before iteratoring (for batch training)
    seed      - randome number generator seed
    args      - command line arguments as parsed by pipeline

    The iterator must return 3 items:
      1) list of featurs, i.e, [image, bld vector]
      2) list of labels, i.e, [categoryA, regressValueB], or None or empty list if no labels
      3) meta - anything user wants, file/row to keep track of data, usually a dictionary, 
                should be able to write the dict to hdf5 - appending each ky/value to growing
                datasets.
                meta can be none as well.
    '''
    np.random.seed(seed)
    random.seed(seed)
    psmlearn.dataset(project='ImgMLearnSmall')
        
    # example of more complicated meta for a sample, a dict of numbers/string
    data_meta = [{'row':k, 'sec':k, 'nano':k, 'filename':'dummy'} for k in range(num)]
    data={'imgs':[np.random.rand(300,300) for k in range(num)]}
    labels = {'signal':[mm['row'] % 2 for mm in data_meta],
              'box':[np.random.rand(4) for mm in data_meta]}
    if randomize:
        random.shuffle(data_meta)

    for meta in data_meta:
        row = meta['row']
        sample_feats = [data['imgs'][row]]
        sample_labels = [labels['signal'][row], labels['box'][row]]
        yield sample_feats, sample_labels, meta

def prepare_for_vgg16(img, config, channel_mean=None):
    img = img.astype(np.float32)
    thresh = config['thresh']
    util.logTrace('prepare_for_vgg16', 'thresh=%.1f' % thresh)
    replace = img >= thresh
    newval = np.log(1.0 + img[replace] - thresh)
    img[replace]=thresh + newval
    img = imresize(img,(224,224), interp='lanczos', mode='F')
    if channel_mean:
        img -= channel_mean
    rgb = np.empty((224,224,3), np.float32)
    for ch in range(3):
        rgb[:,:,ch] = img[:]
    return rgb

def tsne_imgs(data_iter, config, pipeline, step2h5list, output_files):
    initial_dims = config['initial_dims']
    perplexity = config['perplexity']
    data = []
    sig_label = []
    meta = []
    for sample_feats, sample_labels, sample_meta in data_iter:
        img = sample_feats[0]
        ch = img[:,:,0]
        data.append(ch.flatten())
        sig_label.append(sample_labels[0])
        meta.append(sample_meta)
    data = np.vstack(data)
    labels = np.vstack(sig_label)
    util.logTrace(hdr='tsne_imgs', msg='just using first %d comp of images. perplexity=%.1f initial_dims=%d' %
                   (config['tsnecomp'], perplexity, initial_dims))
    Y = tsne(X=data[:,0:config['tsnecomp']], no_dims=2,
             initial_dims=initial_dims, perplexity=perplexity)
    res = {'tsne_imgs':Y, 'labels':sig_label}
    h5util.write_to_h5(output_files[0], datadict=res, meta=meta)
    
def tsne_cws(config, pipeline, step2h5list, output_files):
    '''run tsne on cws for the images that we run before 
    '''
    initial_dims = config['initial_dims']
    perplexity = config['perplexity']
    meta_ky2data = h5util.read_meta(step2h5list['tsne_imgs'][0])
    labels = h5py.File(step2h5list['tsne_imgs'][0],'r')['labels'][:]
    fc2 = h5util.match_meta(meta_ky2data, 'fc2', step2h5list['model_layers'][0])
    util.logTrace(hdr='tsne_cws', msg="about to run tsne on data.shape=%s perplexity=%.1f initial_dims=%d using first %d comp of fc2" % \
                  (fc2.shape, perplexity, initial_dims, config['tsnecomp']))
    Y = tsne(X=fc2[:,0:config['tsnecomp']], no_dims=2,
             initial_dims=initial_dims, perplexity=perplexity)
    res = {'tsne_cws':Y, 'labels':labels}
    h5util.write_to_h5(output_files[0], datadict=res, meta=h5util.convert_meta(meta_ky2data))

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
pipeline = Vgg16Pipeline(outputdir='.',
                         default_data_gen=data_gen,
                         default_data_gen_params={},
                         prepare_for_vgg16=prepare_for_vgg16,
                         layers=['fc1','fc2'])
# pipeline now has the steps
# 'data_stats' - iterates original data, calls prepare_for_vgg16. Creates output file with 'datalen' and 'channel_mean' (of prepared data)
# 'plot_imgprep' - iterates original data, calls prepare_for_vgg16. Compares raw to prepared.
# 'model_layers' - creates output file with 'fc1' 'fc2' - last layers

# this takes too long
pipeline.add_step_fn_with_imgprep_iter(name='tsne_imgs', fn=tsne_imgs,
                                    data_gen_params={'randomize':True},
                                    help="compute tsne embedding on images fed to CNN")
pipeline.add_step_fn_no_iter(name='tsne_cws', fn=tsne_cws,
                             help="compute tsne embedding on condewords after CNN")

pipeline.add_step_fn_plot_no_iter('plot_tsne_cws', fn=plot_tsne_cws)

pipeline.parser.add_argument('--thresh', type=float, help='threshold for starting log with vgg16 image prep, default=30', default=30.0) 
pipeline.parser.add_argument('--tsnecomp', type=int, help='for tsne, number of components of the codewords to do', default=200) 
pipeline.parser.add_argument('--perplexity', type=float, help='for tsne, default=30.0', default=30.0) 
pipeline.parser.add_argument('--initial_dims', type=int, help='for tsne, default=50', default=30) 
#pipeline.run()
