import os

MLEARNDIR = '/reg/d/ana01/temp/davidsch/mlearn'
VGG16_DIR = os.path.join(MLEARNDIR, 'vgg16')


def getProjectDir(projectdir):
    global MLEARNDIR
    assert os.path.exists(MLEARNDIR), "base dir for mlearning project doesn't exist: %s" % MLEANDIR
    fullpath = os.path.join(MLEARNDIR, projectdir)
    assert os.path.exists(fullpath), "your project dir doesn't exist: %s" % fullpath
    return fullpath

def getProjectFile(projectdir, fname):
    projectDir = getProjectDir(projectdir)
    fullfname = os.path.join(projectDir, fname)
    assert os.path.exists(fullfname), "File: %s not found" % fullfname
    return fullfname
