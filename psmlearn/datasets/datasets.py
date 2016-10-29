from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import random

from h5minibatch import DataSetGroup as H5DataSetGroup

from  . import dataloc
from .. import util
from . h5batchdataset import H5BatchDataset
from . dataset import Dataset

IMGMLEARN_DESCR='''
amo86815 runs 69 (no-lasing) 
71/71 (lasing, training), 
72/73 (lasing, predict, molecular runs).
for small, Xtcav images are 1/8 size, reduced by 1/2 1/2 and stored as 
8 bit unsigned integers (scaled to about 2000)
'''

bldsel = H5DataSetGroup(name='bldsel', dsets=[
# to much variation from train to predict
#                   'bld.ebeam.ebeamCharge',  

                   'bld.ebeam.ebeamEnergyBC1',
                   'bld.ebeam.ebeamEnergyBC2',
                   'bld.ebeam.ebeamL3Energy',
                   'bld.ebeam.ebeamLTU250',
                   'bld.ebeam.ebeamLTU450',
                   'bld.ebeam.ebeamLTUAngX',
                   'bld.ebeam.ebeamLTUAngY',
                   'bld.ebeam.ebeamLTUPosX',
                   'bld.ebeam.ebeamLTUPosY',
                   'bld.ebeam.ebeamPkCurrBC1',
                   'bld.ebeam.ebeamPkCurrBC2',

# this is completely different for no-lasing runs
#                   'bld.ebeam.ebeamUndAngX',

                   'bld.ebeam.ebeamUndAngY',

# completely differnt for no lasing
#                   'bld.ebeam.ebeamUndPosX',
                   'bld.ebeam.ebeamUndPosY',
                   'bld.ebeam.ebeamXTCAVAmpl',

# this drifts, looks different run 73, run 70 has extra bumps
 #                  'bld.ebeam.ebeamXTCAVPhase',

# we should use gasdet, but for the time being we won't
#                   'bld.gasdet.f_11_ENRC',
#                   'bld.gasdet.f_12_ENRC',
#                   'bld.gasdet.f_21_ENRC',
#                   'bld.gasdet.f_22_ENRC',
])

bldall = H5DataSetGroup(name='bldall', dsets=[
    'bld.ebeam.damageMask',
    'bld.ebeam.ebeamCharge',
    'bld.ebeam.ebeamDumpCharge',
    'bld.ebeam.ebeamEnergyBC1',
    'bld.ebeam.ebeamEnergyBC2',
    'bld.ebeam.ebeamL3Energy',
    'bld.ebeam.ebeamLTU250',
    'bld.ebeam.ebeamLTU450',
    'bld.ebeam.ebeamLTUAngX',
    'bld.ebeam.ebeamLTUAngY',
    'bld.ebeam.ebeamLTUPosX',
    'bld.ebeam.ebeamLTUPosY',
    'bld.ebeam.ebeamPhotonEnergy',
    'bld.ebeam.ebeamPkCurrBC1',
    'bld.ebeam.ebeamPkCurrBC2',
    'bld.ebeam.ebeamUndAngX',
    'bld.ebeam.ebeamUndAngY',
    'bld.ebeam.ebeamUndPosX',
    'bld.ebeam.ebeamUndPosY',
    'bld.ebeam.ebeamXTCAVAmpl',
    'bld.ebeam.ebeamXTCAVPhase',
    'bld.gasdet.f_11_ENRC',
    'bld.gasdet.f_12_ENRC',
    'bld.gasdet.f_21_ENRC',
    'bld.gasdet.f_22_ENRC',
    'bld.gasdet.f_63_ENRC',
    'bld.gasdet.f_64_ENRC',
    'bld.phasecav.charge1',
    'bld.phasecav.charge2',
    'bld.phasecav.fitTime1',
    'bld.phasecav.fitTime2',
])

def getXtcMetaInH5():
    return ['evt.fiducials',
            'evt.nanoseconds',
            'evt.seconds',
            'run',
            'run.index']

class ImgMLearnDataset(H5BatchDataset):
    def __init__(self, 
                 project='xtcav', 
                 subproject='amo86815_small',
                 verbose=True,
                 X='---',
                 Y='---',
                 predict=False,
                 testmode=False):

        name="ImgMLearnDataset(project=%s subproject=%s, X=%s, Y=%s, predict=%s)" % \
            (project, subproject, X, Y, predict)

        if verbose:
            print(name)

        assert X in ['img','img_bldall', 'img_bldsel'], ("X=%s is wrong, it must be one of:\n"%X)+\
        "  img - just return the xtcavimg\n"+\
        "  img_bldall: return two things, xtcav_img and vec of all Bld\n"+\
        "  img_bldsel: return two things, xtcavimg and subset of bld"

        assert predict or Y in ['enPeak', 'timePeak', 'lasing', 'enAll', 'timeAll'], ("Y=%s must be one of:\n"%Y)+\
        "  enPeak:   1 item: acq.enPeaksLabel (as one hot - 4 values)\n"+\
        "  timePeak: 1 item: acq.peaksLabel (as one hot - 4 values)\n"+\
        "  lasing:   1 item: lasing (as one hot - 2 values)\n"+\
        "  enAll:    5 items: enPeak(one hot), acq.e1.pos, acq.e2.pos, acq.e1.ampl, acq.e2.ampl\n"+\
        "  timeAll:  5 items: timePeak(one hot), acq.t1.pos, acq.t2.pos, acq.t1.ampl, acq.t2.ampl\n"

        if predict:
            Y = []

        meta = getXtcMetaInH5()
        subprojectDir = dataloc.getSubProjectDir(project=project, subproject=subproject)
        h5files = self.getH5files(subprojectDir, predict, Y, testmode)
        h5br_X = ['xtcavimg']
        h5br_X_dset_groups, include_if_one_mask_datasets = self.get_X_dset_groups_and_include(X, predict)
        h5br_Y_to_onehot, hbr_Y_onehot_num_outputs, exclude_if_negone_mask_datasets = self.get_Y_onehot_and_exclude(Y)
        h5br_Y = self.get_Y(Y)

        H5BatchDataset.__init__(self, project=project, 
                                subproject=subproject, 
                                verbose=verbose, 
                                descr=IMGMLEARN_DESCR,
                                name=name,
                                h5files      = h5files,
                                X            = h5br_X,
                                X_dset_groups= h5br_X_dset_groups,
                                Y            = h5br_Y,
                                Y_to_onehot  = h5br_Y_to_onehot,
                                Y_onehot_num_outputs = hbr_Y_onehot_num_outputs,
                                meta         = meta,
                                include_if_one_mask_datasets=include_if_one_mask_datasets,
                                exclude_if_negone_mask_datasets=exclude_if_negone_mask_datasets)

    def get_X_dset_groups_and_include(self,X,predict):
        global bldall
        global bldsel

        if X == 'img_bldall':
            if predict:
                return [bldall], []
            else:
                return [bldall], ['bld.ebeam_gasdet_phasecav.complete_mask']
        elif X == 'img_bldsel':
            if predict:
                return [bldsel], []
            else:
                return [bldsel], ['bld.ebeam_gasdet_phasecav.complete_mask']
        return [],[]

    def get_Y(self, Y):
        if Y in ['enPeak', 'timePeak', 'lasing']:
            return []
        if Y in ['timeAll']:
            return ['acq.t1.pos', 'acq.t2.pos', 'acq.t1.ampl', 'acq.t2.ampl']
        if Y in ['enAll']:
            return ['acq.e1.pos', 'acq.e2.pos', 'acq.e1.ampl', 'acq.e2.ampl']
        return []

    def get_Y_onehot_and_exclude(self, Y):
        if Y in ['enPeak', 'enAll']:
            numOutputs = 4
            return ['acq.enPeaksLabel'], [numOutputs], ['acq.enPeaksLabel']
        if Y in ['timePeak', 'timeAll']:
            numOutputs = 4
            return ['acq.peaksLabel'], [numOutputs], ['acq.peaksLabel']
        if Y == 'lasing':
            numOutputs = 2
            return ['lasing'],[numOutputs], []
        return [],[], []
            
    def getH5files(self, subprojectDir, predict, Y, testmode):
        runs = [70,71]
        globmatch = 'amo86815_mlearn-r%3.3d-c*.h5'
        if Y=='lasing':
            runs=[69,70,71]
        if predict:
            globmatch = 'amo86815_pred-r%3.3d-c*.h5'
            runs=[72,73]
        hdf5 = os.path.join(subprojectDir, 'hdf5')
        assert os.path.exists(hdf5), "dir %s doesn't exist" % hdf5

        h5files = []
        for run in runs:
            globpath=os.path.join(hdf5, globmatch % run)
            newh5files = glob.glob(globpath)
            assert len(newh5files)>0, "didn't get any files from %s" % globpath
            h5files.extend(newh5files)

        if testmode:
            random.shuffle(h5files)
            h5files = h5files[0:3]

        return h5files

###############
def getH5files(project, subproject, testmode, globmatch):
    subprojectDir = dataloc.getSubProjectDir(project=project, subproject=subproject)
    hdf5 = os.path.join(subprojectDir, 'hdf5')
    assert os.path.exists(hdf5), "dir %s doesn't exist" % hdf5
    globpath=os.path.join(hdf5, globmatch)
    h5files = glob.glob(globpath)
    assert len(h5files)>0, "didn't get any files from %s" % globpath
    if testmode:
        random.shuffle(h5files)
        h5files = h5files[0:3]
    return h5files

        
ICEWATER_DESCR='''
sorting ice from water in diffraction, ideally locate ice clusters
'''
class IceWaterDataset(H5BatchDataset):
    def __init__(self, 
                 project='ice_water',
                 subproject='cxi25410',
                 verbose=True,
                 testmode=False):
        h5files = getH5files(project=project, subproject=subproject, 
                             testmode=testmode, globmatch='ice_data.h5')
        H5BatchDataset.__init__(self, 
                                project=project, 
                                subproject=subproject, 
                                verbose=verbose, 
                                descr=ICEWATER_DESCR,
                                name='IceWater(subproject=%s)' % subproject,
                                h5files = h5files,
                                X=['data'],
                                Y_to_onehot=['label'],
                                Y_onehot_num_outputs=[2],
                                meta=[],
                                )


CHUCKVIRUS_DESCR='''
diffraction patterns from two different viruses
'''
class ChuckVirusDataset(H5BatchDataset):
    def __init__(self, 
                 project='diffraction',
                 subproject='yoon82_amo86615',
                 verbose=True,
                 testmode=False):
        h5files = getH5files(project=project, subproject=subproject, 
                             testmode=testmode, globmatch='amo86615_chuck_virus.h5')
        H5BatchDataset.__init__(self, 
                                project=project, 
                                subproject=subproject, 
                                verbose=verbose, 
                                descr=CHUCKVIRUS_DESCR,
                                name='ChuckVirus(subproject=%s)' % subproject,
                                h5files = h5files,
                                X=['adu'],
                                Y_to_onehot=['labels'],
                                Y_onehot_num_outputs=[2],
                                meta=[],
                                )


ACCBEAM_DESCR='''
locating beam on VCC and YAG screens for laser diode calibration,
accelerator beam shaping. From Siqi, matlab files.
'''
class AccBeamVccYagDataset(Dataset):
    def __init__(self, project, subproject, verbose, **kwargs):
        Dataset.__init__(self, project=project, subproject=subproject, 
                         verbose=verbose, descr=ACCBEAM_DESCR)
        raise Exception("AccBeamVccYagDataset not implemented")

###### All datasets #######
DATASETS={('xtcav','amo86815_small'):ImgMLearnDataset,
          ('xtcav','amo86815_full'):ImgMLearnDataset,
          ('ice_water','cxi25410'):IceWaterDataset,
          ('diffraction','yoon82_amo86615'):ChuckVirusDataset,
          ('accbeam','siqili'):AccBeamVccYagDataset}

######## interface
def dataset(project, **kwargs):
    projectDir = dataloc.getProjectDir(project)
    verbose = kwargs.pop('verbose',False)
    subproject = kwargs.pop('subproject','default_subproject')
    subProjectDir = dataloc.getSubProjectDir(project, subproject)
    if subproject == 'default_subproject':
        subproject = os.path.basename(os.path.realpath(subProjectDir))
        subProjectDir = dataloc.getSubProjectDir(project, subproject)
        util.logTrace(hdr='datasets',
                      msg="project=%s, default_subproject detected, subproject set to: %s" % 
                      (project,subproject),
                      flag=verbose)

    dataset_id = (project, subproject)
    
    assert dataset_id in DATASETS, \
        "%s not in known datasets. known datsets are:\n%r" % \
        (dataset_id, DATASETS.keys())
    cl=DATASETS[dataset_id]
    return cl(project=project,
              subproject=subproject,
              verbose=verbose,
              **kwargs)
