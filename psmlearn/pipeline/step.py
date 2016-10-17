from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

class Step(object):
    def __init__(self, name, inst, fn_or_method, data_gen=None, data_gen_params={}, plot=False, pipeline=None, output_suffixes=None):
        self.name = name
        self.inst=inst
        self.fn_or_method = fn_or_method
        self.data_gen = data_gen
        self.data_gen_params = data_gen_params
        self.plot = plot
        self.pipeline = pipeline
        self.output_suffixes = output_suffixes

    def isMethod(self):
        return self.inst != None

    def redo(self, args):
        return getattr(args, self.name)
    
    def __str__(self):
        msg='step=%s plt=%s method=%s' % (self.name, self.plot, self.isMethod())
        if self.data_gen:
            msg += ' data_gen=True, params=%s' % self.data_gen_params
        if self.output_suffixes:
            msg += ' output=%s' % ','.join(self.output_suffixes)
        return msg

    def run(self, step2h5list, output_files, plot=0):
        data_iter = None
        if self.data_gen:
            data_iter = self.data_gen(**self.data_gen_params)
        kwargs = {}
        if self.isMethod():
            kwargs['self'] = self.inst
        kwargs['data_iter'] = data_iter
        if self.plot:
            kwargs['plot'] = plot
        kwargs['pipeline'] = self.pipeline
        kwargs['step2h5list'] = step2h5list
        kwargs['output_files'] = output_files
        self.fn_or_method(**kwargs)
