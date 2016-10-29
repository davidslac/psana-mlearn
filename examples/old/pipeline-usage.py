from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import random
import tensorflow as tf
from psmlearn.pipeline import Pipeline
from psmlearn.h5util import h52dict, dict2h5

##########
def data_generator(randomize=False, labels=False, meta=False, NN=10, seed=109, num_labels=3):
    np.random.seed(seed)
    random.seed(seed)
    data_meta = range(NN)
    data = [np.random.rand((100,100)) for k in range(NN)]
    data_labels = [mm % num_labels for mm in data_meta]
    if randomize:
        random.shuffle(data_meta)
    for mm in data_meta:
        if meta:
            yield data[mm], data_labels[mm], mm
        elif labels:
            yield data[mm], data_labels[mm]
        else:
            yield data[mm]

#########
def view_orig_data(data_iter, stepinfo):
    plt = stepinfo.plt
    for img in data_iter:
        plt.imshow(img, interpolation='none')
        plt.pause(.1)
        raw_input('hit enter')

def data_stats(data_iter, stepinfo):
    means = []
    stddevs = []
    for img in data_iter:
        means.append(np.mean(img))
        stddevs.append(np.std(img))
    res = {'mean':np.mean(means),
           'stddev':np.mean(stddevs)}
    dict2h5(res, stepinfo.outputfile)

def view_data_stats(data_iter, stepinfo):
    plt = stepinfo.plt
    for img in data_iter:
        plt.imshow(img, interpolation='none')
        plt.pause(.1)
        raw_input('hit enter')

pipeline = Pipeline()
parser = pipeline.get_parser(output_dir='.')
args = parser.parse_args()
pipeline.add_step(name='view_orig_data', fn=view_orig_data, data_gen=data_generator, makes_output=False)
pipeline.add_step(name='data_stats', fn=data_stats, data_gen=data_generator, makes_output=True)
pipeline.add_step(name='view_data_stats', fn=view_data_stats, data_gen=None, makes_output=False)
pipeline.run(args=args)

