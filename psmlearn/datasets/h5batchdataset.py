from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5minibatch
assert h5minibatch.__version__=='0.0.4'
from h5minibatch import H5BatchReader

from . dataset import Dataset
from .. import util

class H5BatchDataset(Dataset):
    def __init__(self, project, subproject, verbose, descr, name,
                 h5files, X=[], X_dset_groups=[],
                 Y=[], Y_to_onehot=[],
                 Y_onehot_num_outputs = [],
                 meta=[], 
                 include_if_one_mask_datasets=[],
                 exclude_if_negone_mask_datasets=[]):

        Dataset.__init__(self, 
                         project=project, 
                         subproject=subproject, 
                         verbose=verbose, 
                         descr=descr,
                         name=name)
        assert len(h5files)>0        
        self.h5files = h5files
        self.X=X
        self.X_dset_groups=X_dset_groups
        self.Y=Y
        self.Y_to_onehot=Y_to_onehot
        self.Y_onehot_num_outputs=Y_onehot_num_outputs
        self.meta=meta
        self.include_if_one_mask_datasets = include_if_one_mask_datasets
        self.exclude_if_negone_mask_datasets = exclude_if_negone_mask_datasets

        dsets=self.X + self.Y_to_onehot + self.Y + self.meta
        dset_groups = self.X_dset_groups
        
        self.h5br = H5BatchReader(h5files=h5files,
                                  dsets=dsets,
                                  dset_groups=dset_groups,
                                  include_if_one_mask_datasets=include_if_one_mask_datasets,
                                  exclude_if_negone_mask_datasets=exclude_if_negone_mask_datasets,
                                  verbose=verbose)

    def split(self, **kwargs):
        self.h5br.split(**kwargs)

    def train_iter(self, **kwargs):
        h5br_iter = self.h5br.train_iter(**kwargs)
        return H5BatchDatasetIterWrapper(h5br_iter, self)

    def validation_iter(self, **kwargs):
        h5br_iter = self.h5br.validation_iter(**kwargs)
        return H5BatchDatasetIterWrapper(h5br_iter, self)

    def test_iter(self, **kwargs):
        h5br_iter = self.h5br.test_iter(**kwargs)
        return H5BatchDatasetIterWrapper(h5br_iter, self)

class H5BatchDatasetIterWrapper(object):
    def __init__(self, h5batch_iter, h5batchDataset):
        self.h5batch_iter = h5batch_iter
        self.h5batchDataset = h5batchDataset

    def unpack_batchinfo(self, batch):
        batchinfo = {'epoch':batch['epoch'],
                     'batch':batch['batch'],
                     'filesRows':batch['filesRows'],
                     'readtime':batch['readtime'],
                     'size':batch['size']}
        return batchinfo

    def unpack_XYmeta(self, batch):
        dsets = batch['dsets']
        dset_groups = batch['dset_groups']
        X = [dsets[nm] for nm in self.h5batchDataset.X] 
        for dset_group in self.h5batchDataset.X_dset_groups:
            X.append(dset_groups[dset_group.name])

        Y = []
        for nm, numOutputs in zip(self.h5batchDataset.Y_to_onehot,
                                  self.h5batchDataset.Y_onehot_num_outputs):
            labels = batch['dsets'][nm]
            Y.append(util.convert_to_one_hot(labels, numOutputs))
        for nm in self.h5batchDataset.Y:
            Y.append(batch['dsets'][nm])

        meta = [dsets[nm] for nm in self.h5batchDataset.meta]
        return X, Y, meta
        
    def next(self):
        batch = self.h5batch_iter.next()
        batchinfo = self.unpack_batchinfo(batch)
        X,Y,meta = self.unpack_XYmeta(batch)
        return X,Y,meta,batchinfo

    def __iter__(self):
        return self
        
