'''
Created on April 30, 2015

@author: norris
'''
from orio.module.loop import ast
from orio.main.util.globals import *
from orio.tool.graphlib import graph
from orio.module.loop import astvisitors
from functools import reduce

class DFA:
    '''Abstract dataflow analysis'''
    def __init__(self):
        self.env= []
        pass
    
    def forward_transfer_function(self, analysis, bb, IN_bb):
        OUT_bb = IN_bb.copy()
        for insn in bb:
            analysis.step_forward(insn, OUT_bb)
        return OUT_bb
     
    def backward_transfer_function(self, analysis, bb, OUT_bb):
        IN_bb = OUT_bb.copy()
        for insn in reversed(bb):
            analysis.step_backward(insn, IN_bb)
        return IN_bb
     
    def update(self, env, bb, newval, todo_set, todo_candidates):
        if newval != self.env.get(bb):
            print('{0} has changed, adding {1}'.format(bb, todo_candidates))
            env[bb] = newval
            todo_set |= todo_candidates
     
    def maximal_fixed_point(self, analysis, cfg, init={}):
        # state at the entry and exit of each basic block
        IN, OUT = {}, {}
        for bb in cfg.nodes():
            IN[bb] = {}
            OUT[bb] = {}
        IN[cfg.entry_point] = init
     
        # first make a pass over each basic block
        todo_forward = cfg.nodes()
        todo_backward = cfg.nodes()
     
        while todo_backward or todo_forward:
            while todo_forward:
                bb = todo_forward.pop()
     
                ####
                # compute the environment at the entry of this BB
                new_IN = reduce(analysis.meet, list(map(OUT.get, cfg.pred(bb))), IN[bb])
                self.update(IN, bb, new_IN, todo_backward, cfg.pred(bb))
     
                ####
                # propagate information for this basic block
                new_OUT = self.forward_transfer_function(analysis, bb, IN[bb])
                self.update(OUT, bb, new_OUT, todo_forward, cfg.succ(bb))
     
            while todo_backward:
                bb = todo_backward.pop()
     
                ####
                # compute the environment at the exit of this BB
                new_OUT = reduce(analysis.meet, list(map(IN.get, cfg.succ(bb))), OUT[bb])
                self.update(OUT, bb, new_OUT, todo_forward, cfg.succ(bb))
     
                ####
                # propagate information for this basic block (backwards)
                new_IN = self.backward_transfer_function(analysis, bb, OUT[bb])
                self.update(IN, bb, new_IN, todo_backward, cfg.pred(bb))
     
        ####
        # IN and OUT have converged
        return IN, OUT