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
        '''To evaluate the given expression text'''

        try:
            val = eval(text, self.perf_params)
        except Exception, e:
            err('orio.module.ortil.ann_parser: failed to evaluate expression: "%s"\n --> %s: %s' % (text,e.__class__.__name__, e))
        return val

    #------------------------------------------------------------

    def parse(self, text):
        '''
        Parse the given text to extract tiling information.
        The given code text has the following syntax:
          (<loop-iter>, ...) : <num-tiling-level>
        '''

        # remember the given code text
        orig_text = text

        # regular expressions
        __num_re = r'\s*(\d+)\s*'
        __var_re = r'\s*([A-Za-z_]\w*)\s*'
        __colon_re = r'\s*:\s*'
        __comma_re = r'\s*,\s*'
        __oparenth_re = r'\s*\(\s*'
        __cparenth_re = r'\s*\)\s*'

        # initialize the default values of the tiling information
        num_level = 1
        iter_names = []

        # get all iterator names of the loops to be tiled
        m = re.match(__oparenth_re, text)
        if not m:
            err('orio.module.ortil.ann_parser: OrTil: annotation syntax error: "%s"' % orig_text)
        text = text[m.end():]        
        m = re.search(__cparenth_re, text)
        if not m:
            err('orio.module.ortil.ann_parser: OrTil: annotation syntax error: "%s"' % orig_text)
        itext = text[:m.end()-1]
        text = text[m.end():]
        while True:
            if (not itext) or itext.isspace():
                break
            m = re.match(__var_re, itext)
            if not m:
                err('orio.module.ortil.ann_parser: OrTil: annotation syntax error: "%s"' % orig_text)
            iname = m.group(1)
            if iname in iter_names:
                err('orio.module.ortil.ann_parser: OrTil: repeated iterator name: "%s"' % iname)
            iter_names.append(iname)
            itext = itext[m.end():]
            m = re.match(__comma_re, itext)
            if m:
                itext = itext[m.end():]

        # check if further parsing is needed
        if (not text) or text.isspace():
            tiling_info = [num_level, iter_names]
            return tiling_info
        
        # get a colon
        m = re.match(__colon_re, text)
        if not m:
            err('orio.module.ortil.ann_parser: OrTil: annotation syntax error: "%s"' % orig_text)
        text = text[m.end():]

        # get the number of tiling levels
        m = re.match(__num_re, text)
        if not m:
            m = re.match(__var_re, text)
        if not m:
            err('orio.module.ortil.ann_parser: OrTil: annotation syntax error: "%s"' % orig_text)
        text = text[m.end():]
        num_level = m.group(1)
        num_level = self.__evalExp(num_level)

        # check the semantic of the number of tiling levels
        if not isinstance(num_level, int) or num_level <= 0:
            err('orio.module.ortil.ann_parser: OrTil: the number of tiling levels must be a positive integer')

        # is there any trailing texts?
        if text and not text.isspace():
            err('orio.module.ortil.ann_parser: OrTil: annotation syntax error: "%s"' % orig_text)
        
        # return the tiling information
        tiling_info = [num_level, iter_names]
        return tiling_info

