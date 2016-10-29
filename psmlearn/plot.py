from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py

def plotRowsLabelSort(dataLoc, labelsLoc):
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    datah5 = h5py.File(dataLoc['file'], 'r')
    data = datah5[dataLoc['dset']][:]
    labelh5 = h5py.File(labelsLoc['file'],'r')
    labels = labelh5[labelsLoc['dset']][:]
    
