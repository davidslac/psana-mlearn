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
import h5py

VALIDATION_FILES = ['amo86815_mlearn-r071-c0021.h5',
                    'amo86815_mlearn-r070-c0018.h5',
                    'amo86815_mlearn-r071-c0054.h5',
                    'amo86815_mlearn-r070-c0031.h5',
                    'amo86815_mlearn-r071-c0042.h5']

def train_data_gen(randomize=False, seed=1031, args=None, num=0):
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
    global VALIDATION_FILES
    np.random.seed(seed)
    random.seed(seed)
    h5files = []
    runs = map(int,args.runs.split(','))
    for run in runs:
        h5files.extend(glob(os.path.join(args.datadir, 'amo86815_mlearn-r%3.3d*.h5' % run)))
    h5files = [h5file for h5file in h5files if os.path.basename(h5file) not in VALIDATION_FILES]
    assert len(h5files)>0
    random.shuffle(h5files)
    if args.dev:
        h5files=h5files[0:5]
    datareader = H5MiniBatchReader(h5files=h5files,
                                   minibatch_size=args.minibatch_size,
                                   validation_size=args.valid_size,
                                   feature_dataset=args.X,
                                   label_dataset=args.Y,
                                   return_as_one_hot=True,
                                   feature_preprocess=None,
                                   number_of_batches=None,
                                   class_labels_max_imbalance_ratio=args.imb,
                                   max_mb_to_preload_all=None,
                                   add_channel_to_2D='row_column_channel',
                                   random_seed=None,
                                   verbose=True)
    return datareader

def train(args, datareader, model):
    return vgg16net.train(args, datareader, model)

def predict(args, datareader, model):
    return vgg16net.predict(args, datareader, model)

def gbprop(args, datareader, model):
    return vgg16net.gbprop(args, datareader, model)

def pipeline(args):
    random.seed(args.seed)
    datareader = make_datareader(args)

    with tf.Graph().as_default():
        model = None
        if not (args.predict or args.gbprop):
            model = train(args, datareader, model)
        if not args.gbprop:
            model = predict(args, datareader, model)
        if args.gbprop:
            gbprop(args, datareader, model)


### MAIN #########################################################################
if __name__ == '__main__':
pipeline = Vgg16Pipeline(outputdir='.',
                         default_data_gen=data_gen,
                         default_data_gen_params={},
                         prepare_for_vgg16=prepare_for_vgg16,
                         layers=['fc1','fc2'])

    default_seed = 92839
    default_l1reg = 0.005
    default_l2reg = 0.0005
    default_optimizer = 'momentum'
    default_momentum = 0.9
    default_learning_rate = 0.1
    default_lr_decay_rate = 0.97
    default_lr_decay_steps = 50
    default_minibatch_size = 64
    default_validation_size = 128
    default_train_steps = 8000
    default_steps_between_evals = 50

    default_runs='70,71'
    default_datadir = '/reg/d/ana01/temp/davidsch/ImgMLearnSmall'
    default_X = 'xtcavimg'
    default_Y = 'acq.peaksLabel'
    default_dimg = 'gbprop'
    default_imbalance = 4.0
    default_preprocess = 'log'  # can also be none

    parser.add_argument('--dev', action='store_true', help='development mode')
    parser.add_argument('--seed', type=int, help='seed for python random module.', default=92839) 
    parser.add_argument('--l1reg', type=float, help='l1reg during training. def=%f' % default_l1reg, default=default_l1reg)
    parser.add_argument('--l2reg', type=float, help='l2reg during training. def=%f' % default_l2reg, default=default_l2reg)
    parser.add_argument('--opt', type=str, help='training optimizer, momentum, adadelta, adagrad, adam, ftrl or rmsprop. default=%s' % default_optimizer, default=default_optimizer)
    parser.add_argument('--mom', type=float, help='momentum optimizers momentum, default=%f' % default_momentum, default=default_momentum)    
    parser.add_argument('--lr', type=float, help='learning rate. defaults to %f' % default_learning_rate, default=default_learning_rate)
    parser.add_argument('--lr_decay_rate', type=float, help='decay rate for learning rate. defaults to %f' % default_lr_decay_rate, default=default_lr_decay_rate)
    parser.add_argument('--lr_decay_steps', type=int, help='decay steps for learning rate. defaults to %d' % default_lr_decay_steps, default=default_lr_decay_steps)
    parser.add_argument('--staircase', action='store_true', help='staricase for learning rate decay')
    parser.add_argument('--trainable', action='store_true', help='make all vgg variables trainable')
    parser.add_argument('--train_steps', type=int, help='number of training steps default=%d' % default_train_steps, default=default_train_steps)
    parser.add_argument('--train_save', type=str, help='name of trainer to save', default='vgg16_t12_model')
    parser.add_argument('--force', action='store_true', help='force overwrite of existing model name')
    parser.add_argument('--eval_steps', type=int, help='number of steps between evals default=%d' % default_steps_between_evals, default=default_steps_between_evals)
    parser.add_argument('--minibatch_size', type=int, help='minibatch size, default=%d' % default_minibatch_size, default=default_minibatch_size)
    parser.add_argument('--valid_size', type=int, help='validation size, default=%d' % default_validation_size, default=default_validation_size)
    parser.add_argument('--intra_op_parallelism_threads', type=int, help='number of intra op threads, default=12', default=12)

    parser.add_argument('--runs',     type=str, help='comma separated list of runs to process for dataset, default=%s' % default_runs, default=default_runs)
    parser.add_argument('--datadir',  type=str, help='full path to data directory default=%s' % default_datadir, default=default_datadir)
    parser.add_argument('--X',        type=str, help='dataset for features, default=%s' % default_X, default=default_X)
    parser.add_argument('--Y',        type=str, help='dataset for labels/Y, default=%s' % default_Y, default=default_Y)
    parser.add_argument('--predict',  action='store_true', help='jump to prediction')
    parser.add_argument('--gbprop',   action='store_true', help='do gbprop')
    parser.add_argument('--dimg',     type=str, help='saliency map, one of bprop gpropb, default=%s' % default_dimg, default=default_dimg)

    parser.add_argument('--imb',      type=float, help='max imbalance in classes, defaults to %.2f' % default_imbalance, default=default_imbalance)
    parser.add_argument('--prep',     type=str,   help='preprocess, none or log,  default=%s' % default_preprocess, default=default_preprocess)

    args = parser.parse_args()
    import tensorflow as tf
    from h5minibatch.H5MiniBatchReader import H5MiniBatchReader
    import vgg16net

    pipeline(args)
    
