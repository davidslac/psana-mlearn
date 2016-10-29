from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from  . import dataloc
from .. import util

########### Dataset classes - detail - 
class Dataset(object):
    def __init__(self, project, subproject, verbose, descr, name):
        self.project=project
        self.subproject=subproject
        self.verbose=verbose
        self.projectDir = dataloc.getProjectDir(project)
        self.subProjectDir = dataloc.getSubProjectDir(project, subproject)
        self.descr=descr
        self.name=name

    def __str__(self):
        return self.name
