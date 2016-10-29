from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import random
import tensorflow as tf
from psmlearn.pipeline import Pipeline
from psmlearn.h5util import h52dict, dict2h5

def preprocess(x, stats=None):
    if stats:
        x -= stats['mean'].value
    return x

class MyPipeline(Pipeline):
    def __init__(self):
        super(MyPipeline,self).__init__()

    def init(self, args, preprocess, stats_data_generator, layers_data_generator):
        super(MyPipeline,self).init(args, data_generator, preprocess)

    def view_orig_data(self, data_iter):
        pass

    def data_stats(self, data_iter, preprocess, input_files, output_files):
        data = [preprocess(datum) for datum in data_iter]
        stats = {'mean':sum(data)/float(len(data))}
        dict2h5(output_files[0],stats)

    def model_layers(self, data_iter, preprocess, input_files, output_files):
        stats = h52dict(input_files[0])
        layers = {'layer':[preprocess(data,stats) for data in data_iter]}
        dict2h5(output_files[0],layers)
    
def data_generator(randomize, labels, meta, preprocess):
    '''return an iterator for Pipeline that does one pass
    through the data. It supports the following 4 parameters for
    pipeline:
      randomize  - randomize the data, yield  X, one feature vector or image per iteration
      labels     - yield X,Y where Y is a label - can be None if not known
      meta       - yield X,Y,meta where meta is a dictionary of meta information on the sample
                   that can be appeneded row by row to an h5 file
      preprocess -  a function that takes just X as argument. 
    '''
    NN=100
    meta = range(NN)
    data = [10*row for row in meta]
    labels = [row%3 for row in meta]
    allinfo = zip(data, labels, meta)
    alldata = zip(range(NN)
    if randomize:
        random.shuffle(alldata)
    for meta,n in enumerate(alldata):
        if preprocess:
            n = preprocess(n)
        if meta:
            yield n,None
        yield n

pipeline = MyPipeline()
parser = pipeline.get_parser(output_dir='.')
args = parser.parse_args()
orig_data_gen = functools.partial(data_generator, randomize=False, labels=False, meta=False)
                  pipeline.add_step_method('view_orig_data', )
                  pipeline.add_step_function('data_stats',)
pipeline.add_step_method('view_data_stats')
pipeline.add_step_method('model_layers')
pipeline.add_step_method('view_model_layers')

stats_data_generator = 
layers_data_generator = functools.partial(data_generator, randomize=False, labels=False, meta=False)
pipeline.init(args=args, preprocess=preprocess,
              stats_data_generator=stats_data_generator, 
              layers_data_generator=layers_data_generator)
pipeline.run()

