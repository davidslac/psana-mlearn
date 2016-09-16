from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def is_integral(dtype):
    return dtype in  [np.int, 
                      np.int8, np.uint8, 
                      np.int16, np.uint16, 
                      np.int32, np.uint32,
                      np.int64, np.uint64]

def hcat_load(h5, dsets, include):
    inc = h5[include][:]
    assert is_integral(inc.dtype), "include dataset %s is not integral" % include
    assert 0 == len(set(inc).difference(set([0,1]))), "include dataset %s is not 0 or 1, it is %r" % (include, set(inc))
    selIdx = inc==1

    for dset in dsets:
        assert len(h5[dset].shape)==2, "dset %s does not have shape 2" % dset

    X = np.hstack([h5[dset][:] for dset in dsets])
    return X[selIdx,:]
                
