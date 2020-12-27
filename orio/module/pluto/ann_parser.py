#
# The implementation of annotation parser
#

import re, sys
from orio.main.util.globals import *

#----------------------------------------------------------------

class AnnParser:
    '''The class definition for the annotation parser'''

    def __init__(self, perf_params):
        '''To instantiate the annotation parser'''

        self.perf_params = perf_params
    
    #------------------------------------------------------------

    def __evalExp(self, text):
        '''Evaluate the given expression text'''

        try:
            val = eval(text, self.perf_params)
        except Exception as e:
            err('orio.module.pluto.ann_parser: failed to evaluate expression: "%s"\n --> %s: %s' % (text, e.__class__.__name__, e))
        return val

    #------------------------------------------------------------

    def parse(self, text):
        '''Parse the annotation text to get variable-value pairs'''

        # remember the given code text
        orig_text = text

        # regular expressions
        __python_exp_re = r'\s*((.|\n)*?);\s*'
        __var_re = r'\s*([A-Za-z_]\w*)\s*'
        __semi_re = r'\s*;\s*'
        __equal_re = r'\s*=\s*'

        # initialize the data structure to store all variable-value pairs
        var_val_pairs = []

        # get all variable-value pairs
        while True:
            if (not text) or text.isspace():
                break
            m = re.match(__var_re, text)
            if not m:
                err('orio.module.pluto.ann_parser: Pluto: annotation syntax error: "%s"' % orig_text)
            text = text[m.end():]
            var = m.group(1)
            m = re.match(__equal_re, text)
            if not m:
                err('orio.module.pluto.ann_parser: Pluto: annotation syntax error: "%s"' % orig_text)
            text = text[m.end():]
            if text.count(';') == 0:
                err('orio.module.pluto.ann_parser: Pluto: annotation syntax error: "%s"' % orig_text)
            m = re.match(__python_exp_re, text)
            if not m:
                err('orio.module.pluto.ann_parser: Pluto: annotation syntax error: "%s"' % orig_text)
            text = text[m.end():]
            val = m.group(1)
            var_val_pairs.append((var, val))

        # evaluate all values and check their semantics
        n_var_val_pairs = []
        for var, val in var_val_pairs:
            val = self.__evalExp(val)
            n_var_val_pairs.append((var, val))
        var_val_pairs = n_var_val_pairs

        # return all variable value pairs
        return var_val_pairs
