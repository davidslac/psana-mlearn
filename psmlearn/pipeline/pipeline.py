from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import h5py
import tensorflow as tf
import logging

import psmlearn.util as util
import psmlearn.h5util as h5util

from .step import Step

def xor(A,B):
    if A and (not B): return True
    if (not B) and A: return True
    return False

def _redo_flag_name(name):
    return name
#    return 'do_%s' % name


def _addPipelineArgs(parser, outputdir):
    parser.add_argument('prefix', type=str, help='prefix for filenames')
    parser.add_argument('--redoall', action='store_true', help='redo all steps', default=False)
    parser.add_argument('--outputdir', type=str, help='output directory default=%s' % outputdir, default=outputdir)
    parser.add_argument('--seed', type=int, help='seed for random number generators', default=39819)
    parser.add_argument('--plot', type=int, help='plot level. default=0, no plots, 1 means detailed', default=0)
    parser.add_argument('--log', type=str, help='one of DEBUG,INFO,WARN,ERROR,CRITICAL.', default='DEBUG')
    parser.add_argument('--force', action='store_true', help='overwrite existing filenames')
    parser.add_argument('--clean', action='store_true', help='delete all output for this prefix')
    parser.add_argument('--config', type=str, help='config file for steps. a .yml file', default=None)

    
class Pipeline(object):
    '''base class for pipeline.
    subclasses should
    override init and call base class
    implement 'data_stats', 'model_layers' methods that take lists of input and output files
    '''
    def __init__(self, outputdir='.',
                 default_data_gen=None,
                 default_data_gen_params={},
                 description='', epilog='', sess=None, plt=None, comm=None):
        self.initialized = False
        self.outputdir = outputdir
        self.default_data_gen = default_data_gen
        self.default_data_gen_params = default_data_gen_params
        self.description=description
        self.epilog=epilog
        self.sess = sess
        self.plt = plt
        self.comm = comm

        self.args = None
        self.steps = []
        self.name2step = {}
        self._steps_fixed = False
        
        if self.sess is None:
            self.sess = tf.Session()
            
        self.doTrace=False
        self.doDebug=False
        self.hdr = 'Pipeline'

        self.parser = argparse.ArgumentParser(add_help=False)
        _addPipelineArgs(parser=self.parser, outputdir=self.outputdir)
        
    def _add_step(self, name, inst, fn_or_method, plot, data_gen, data_gen_params):
        assert not xor(data_gen, data_gen_params), "if supplying one of data_gen or data_gen_params, you must supply the other"
        assert not self._steps_fixed, "steps are fixed, run() must have been called, can't add step %s" % name
        if data_gen is None:
            data_gen = self.default_data_gen
            data_gen_params = self.default_data_gen_params
        output_suffixes= [name + '.h5']
        for step in self.steps:
            for suffix in step.output_suffixes:
                assert suffix not in output_suffixes, "step %s has output suffix=%s that collides with suffix for step %s" % (name, suffix, step.name)
        step = Step(name=name,
                    inst=inst, fn_or_method=fn_or_method,
                    data_gen=data_gen,
                    data_gen_params=data_gen_params,
                    plot=plot,
                    pipeline=self,
                    output_suffixes=output_suffixes)
        self.steps.append(step)
        self.name2step[name]=step
        self.parser.add_argument('--%s' % name, action='store_true', help='just execute step %s' % name, default=False)
        
    def add_step_fn_plot(self, name, fn, data_gen=None, data_gen_params={}):
        self._add_step(name=name, inst=None, fn_or_method=fn, plot=True, data_gen=data_gen, data_gen_params=data_gen_params)
        
    def add_step_fn(self, name, fn,data_gen=None, data_gen_params={}):
        self._add_step(name=name, inst=None, fn_or_method=fn, plot=False, data_gen=data_gen, data_gen_params=data_gen_params)

    def add_step_method_plot(self, inst, method, data_gen=None, data_gen_params={}):
        self._add_step(name=name, inst=inst, fn_or_method=method, plot=True, data_gen=data_gen, data_gen_params=data_gen_params)

    def add_step_method(self, inst, method, fn,data_gen=None, data_gen_params={}):
        self._add_step(name=name, inst=inst, fn_or_method=method, plot=False, data_gen=data_gen, data_gen_params=data_gen_params)
    
    def trace(self, msg):
        util.logTrace(self.hdr, msg, self.doTrace)

    def debug(self, msg):
        util.logDebug(self.hdr, msg, self.doDebug)

    def get_step_output_files(self, step):
        assert self.args is not None, "arguments not parsed"
        if step.plot:
            return []
        outputdir = self.args.outputdir
        prefix = self.args.prefix
        output_prefix = os.path.join(outputdir, prefix)
        outputs = []
        for suffix in step.output_suffixes:
            outputs.append('_'.join([output_prefix,  suffix]))
        return outputs

    def set_plt(self):
        import matplotlib.pyplot as plt
        self.plt = plt

    def do_plot_step(self, ran_last_step, step):
        do_step = step.redo(self.args)
        if do_step and not self.plt:
            self.set_plt()
        if do_step or (ran_last_step and self.plt):
            return True
        return False

    def run(self):
        self._steps_fixed=True
        self._set_args_and_plt()
        
        if self.args.clean:
            self.trace("Cleaning output files")
            for step in self.steps:
                output_files = self.get_step_output_files(step)
                for fname in output_files:
                    if os.path.exists(fname):
                        os.unlink(fname)
                        self.trace("step=%s Deleted file: %s" % (step.name, fname))
                    else:
                        self.trace("step=%s output file: %s doesn't exist" % (step.name, fname))
        else:
            self.trace("Running Pipeline")
            step2h5list = {}
            ran_last_step=True
            for step in self.steps:
                msg = str(step)
                if step.plot:
                    if self.do_plot_step(ran_last_step, step):
                        msg += " -- running"
                        self.trace(msg)
                        step.run(step2h5list=step2h5list, output_files=None, plot=self.args.plot)
                    else:
                        self.trace(msg + " -- skipping plot step")
                else:
                    output_files = self.get_step_output_files(step)
                    self.manage_step(step, step2h5list, output_files)
                    step2h5list[step.name]= [h5py.File(fname,'r') for fname in output_files]

    def manage_step(self, step, step2h5list, output_files):
        all_output_exists = all([os.path.exists(fname) for fname in output_files])
        any_output_exists = any([os.path.exists(fname) for fname in output_files])
        if all_output_exists and not (self.args.redoall or step.redo(self.args)):
            self.trace("step=%s already done, all output exists" % step.name)
        else:
            if any_output_exists and not self.args.force:
                raise Exception("Some of the output files: %s already exist, use --force to overwrite" % step)
            self.trace("running step=%s" % step)
            step.run(step2h5list, output_files)
            for fname in output_files:
                assert os.path.exists(fname), "step=%s did not create output file: %s" % (step, fname)
                
    def _set_args_and_plt(self):
        descr = "pipeline for managing sequence of analysis steps. The steps are:\n"
        for step in self.steps:
            descr += '  %s\n' % step.name
        descr += 'The pipeline only re-runs steps if their output files are not present, or if\n'
        descr += 'switches below are used.  A individual step can be re-run with a switch'
        if self.description:
            descr += '\n----------\n%s' % self.description
                                            

        final_parser = argparse.ArgumentParser(parents=[self.parser],
                                               description=descr,
                                               epilog=self.epilog,
                                               formatter_class=argparse.RawDescriptionHelpFormatter)

        args = final_parser.parse_args()

        nm2lvl = {'INFO':logging.INFO,
                  'DEBUG':logging.DEBUG}
#                  'ERROR':logging.ERROR,
#                  'CRITICAL':logging.CRITICAL,
#                  'ERROR':logging.ERROR}
        assert args.log in nm2lvl, "log must be one of %s" % str(nm2lvl.keys())
        self.doTrace=nm2lvl[args.log] <= logging.INFO
        self.doDebug=nm2lvl[args.log] <= logging.DEBUG
        if args.plot:
            self.set_plt()
        self.args=args    
