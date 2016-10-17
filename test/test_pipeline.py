import os
import numpy as np
import h5py

def test_pipeline_basic():
    testdir=os.path.split(os.path.abspath(__file__))[0]
    pkgdir = os.path.split(testdir)[0]
    exampledir = os.path.join(pkgdir, 'examples')
    assert os.path.exists(exampledir)
    script = os.path.join(exampledir, 'pipeline-basic.py')
    assert os.path.exists(script)

    prefix = 'test_pipeline_xxx'
    ret_hdr_cmds = [(0, '--CLEANING ANY OLD--', 'python %s --clean %s' % (script, prefix)),
                    (0, '--CREATING FROM SCRATCH--', 'python %s %s' % (script, prefix)),
                    (-1, '--FAIL TO REDO--', 'python %s --redoall %s' % (script, prefix)),
                    (0, '--FORCE REDO--', 'python %s --force --redoall %s' % (script, prefix)),
    ]
    for ret_hdr_cmd in ret_hdr_cmds:
        ret,hdr,cmd = ret_hdr_cmd
        print(hdr)
        if ret == 0:
            assert 0 == os.system(cmd), "command failure: %s" % cmd
        else:
            os.system(cmd)
    h5=h5py.File('%s_stepA.h5' % prefix, 'r')
    X,Y=h5['X'][:], h5['Y'][:]
    ansX = np.array([1,2,3])
    ansY = np.array([3,1,3])
    assert np.all(np.equal(ansX,X))
    assert np.all(np.equal(ansY,Y))
    
if __name__ == '__main__':
    test_pipeline_basic()
