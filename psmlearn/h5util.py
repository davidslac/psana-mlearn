from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

def is_integral(dtype):
    return dtype in  [np.int, 
                      np.int8, np.uint8, 
                      np.int16, np.uint16, 
                      np.int32, np.uint32,
                      np.int64, np.uint64]

def hcat_load(h5, dsets, include):
    if not isinstance(h5, h5py.File):
        h5 = h5py.File(h5,'r')
    inc = h5[include][:]
    assert is_integral(inc.dtype), "include dataset %s is not integral" % include
    assert 0 == len(set(inc).difference(set([0,1]))), "include dataset %s is not 0 or 1, it is %r" % (include, set(inc))
    selIdx = inc==1

    for dset in dsets:
        assert len(h5[dset].shape)==2, "dset %s does not have shape 2" % dset

    X = np.hstack([h5[dset][:] for dset in dsets])
    return X[selIdx,:]
                

def write_meta(h5out, all_meta):
    if not isinstance(h5out, h5py.File):
        h5out = h5py.File(h5out,'r+')
    if len(all_meta)==0: return
    grp = h5out.create_group('meta')
    meta = all_meta[0]
    keys = meta.keys()
    for key in keys:
        data = [meta[key] for meta in all_meta]
        grp[key]=data

def read_meta(h5in):
    if not isinstance(h5in, h5py.File):
        h5in = h5py.File(h5in,'r')
    assert 'meta' in h5in.keys()
    meta = h5in['meta']
    assert isinstance(meta, h5py.Group), "meta is not a group in h5"
    key2data = {}
    for key in meta.keys():
        key2data[key]=meta[key][:]
    return key2data

def convert_meta(ky2data):
    meta = []
    kys = ky2data.keys()
    num = len(ky2data.values()[0])
    for ii in range(num):
        row = {}
        for ky in kys:
            row[ky]=ky2data[ky][ii]
        meta.append(row)
    return meta

def write_to_h5(output_file, datadict, meta=None):
    '''writes a dict, and a list of dicts into a 'meta' group
    '''
    h5 = h5py.File(output_file,'w')
    for ky, val in datadict.iteritems():
        h5[ky]=val
    if meta:
        write_meta(h5, meta)
    h5.close()

def read_from_h5(h5in):
    if not isinstance(h5in, h5py.File):
        h5in = h5py.File(h5in, 'r')
    res = {}    
    for ky in h5in.keys():
        if ky == 'meta':
            res['meta'] = read_meta(h5in)
            continue
        try:
            res[ky]=h5in[ky][:]
        except:
            try:
                res[ky]=h5in[ky].value
            except:
                sys.stderr.write('WARNING: h5dict - root key=%s for file=%s is not data, skipping' % (ky, h5in.filename))
    h5in.close()
    return res

def match_meta(match, dsetname, h5):
    if not isinstance(h5, h5py.File):
        h5 = h5py.File(h5,'r')
    matchFrom = read_meta(h5)
    assert set(match.keys())==set(matchFrom.keys())
    metaKeys = match.keys()
    metaKeys.sort()
    toFind = set()    
    numberToMatch = len(match.values()[0])
    for ii in range(numberToMatch):
        toFind.add(tuple([match[ky][ii] for ky in metaKeys]))
    numberToMatchFrom = len(matchFrom.values()[0])
    dset = h5[dsetname]
    data = None
    idx = 0
    for row in range(numberToMatchFrom):
        thisrow = tuple([matchFrom[ky][row] for ky in metaKeys])
        try:
            toFind.remove(thisrow)
        except KeyError:
            continue
        nxtdata = dset[row,:]
        if data is None:
            target_shape = [numberToMatch,] + list(nxtdata.shape)
            data = np.empty(target_shape, dtype=nxtdata.dtype)
        data[idx,:] = nxtdata[:]
        idx += 1
    return  data
        
    
