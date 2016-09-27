from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

########### details - 
class Dataset(object):
    def __init__(self, name, **kwargs):
        self.name=name

    def describe():
        pass

    
class amo86815_lasing(Dataset):
    def __init__(self, name, **kwargs):
        Dataset.__init__(self, name, **kwargs)

    def describe(self):
        msg = ''''
amo86815 runs 69 (no-lasing) 71/71 (lasing, training), 72/73 (lasing, predict, molecular runs).
Xtcav images are 1/8 size, reduced by 1/2 1/2 and stored as 8 bit unsigned integers (scaled to about 2000)
'''
        return msg
    
######## interface
DATASETS = {'amo86815-small-lasing':amo86815_lasing,
            }

def dataset(name, **kwargs):
    assert name in DATASETS, "name=%s not in known datasets. known datsets are:\n  %s" % '\n  '.join(DATASETS.keys())
    return DATASETS[name](name, **kwargs)

