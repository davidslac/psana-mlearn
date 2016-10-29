    def classificationNumOutputs(self, dset):
        '''only use with a integral dataset, returns max value of dataset + 1
        '''
        cache_name = 'classificationNumOutputs_%s' % dset
        if cache_name in self._cached_values:
            return self._cached_values[cache_name]
        mx = -1
        if self._verbose:
            print("reading through h5files to see how many outputs for %s" % dset)
            sys.stdout.flush()
        for fname in self._h5files:
            h5 = h5py.File(fname,'r')
            vals = h5[dset][:]
            assert len(vals.shape)==1
            mx = max(mx,np.max(vals))
        numOutputs = mx+1
        self._cached_values[cache_name]=numOutputs
        return numOutputs

        feat_shapes, label_shapes = h5reader.get_shapes()
    for feat, shape in zip(h5reader.features, feat_shapes):
        print("  %s=%s" % (feat, shape))
    for feat, shape in zip(h5reader.labels, label_shapes):
        print("  %s=%s" % (feat, shape))

    numOutputs = h5reader.classificationNumOutputs('acq.enPeaksLabel')
    print("acq.enPeaksLabel has %d outputs" % numOutputs)
    h5reader.balance(dset='acq.enPeaksLabel', ratio=1.0)
    h5reader.split(train=80, validation=10, test=10)
    trainIter = h5reader.get_iter(partition='train',
                                  batchsize=1,
                                  epochs=1,
                                  num=10)
    for X, Y, meta, batch in trainIter:
        xtcavimg, fvec = X
        enPeaksLabel, e1pos, e2pos, e1ampl, e2ampl = Y
        fid, nano, sec, run, runindex, filenum, row = meta
        epoch, num, readtm = batch['epoch'], batch['num'], batch['read.time']


    def save(self, fname, force):
        if os.path.exists(fname) and (not force):
            raise Exception("h5batchreader.save - fname=%s exists, use force=True to overwrite" % fname)

        h5 = h5py.File(fname, 'w')

        h5['files']=self.h5files

        h5['verbose']=self.verbose

        if self.include_if_one_mask_datasets:
            h5['has_include_if_one_mask_datasets']=True
            h5['include_if_one_mask_datasets']=self.include_if_one_mask_datasets
        else:
            h5['has_include_if_one_mask_datasets']=False

        if self.exclude_if_negone_mask_datasets:
            h5['has_exclude_if_negone_mask_datasets']=True
            h5['exclude_if_negone_mask_datasets']=self.exclude_if_negone_mask_datasets
        else:
            h5['has_exclude_if_negone_mask_datasets']=False

        h5['dsets']=self.dsets
        
        h5['all_samples']=self.
        
