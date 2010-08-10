# This file is part of Orio
# Original author: Boyana Norris, norris@mcs.anl.gov
# (c) 2010 UChicago Argonne, LLC
# For copying information, see the file LICENSE

""" Various error-handling related miscellanea
"""
import sys, os
import traceback
from main.util.globals import Globals   # contains the logger

#--------------------------
#--------------------------

def err(errmsg='',errcode=1, doexit=True):
    sys.stderr.write(Globals().error_pre + 'ERROR: ' + errmsg + Globals().error_post + '\n')
    Globals().logger.error(errmsg + '\n' + '\n'.join(traceback.format_stack()))
    if Globals().debug:
        traceback.print_stack()
    if doexit: sys.exit(errcode)

def warn(msg='',end = '\n', pre='', post=''):
    sys.stderr.write(pre + 'WARNING: ' +  msg + post + end)
    Globals().logger.warning(msg)

def info(msg='', end='\n', pre='', post='', logging=True):
    sys.stderr.write(pre + msg + post + end)
    if logging:
        Globals().logger.info(msg)

def debug(msg='', end='\n', pre='', post='', logging=True):
    if Globals().debug:
        sys.stderr.write(pre + 'DEBUG:' + msg + post + end)
        if logging:
            Globals().logger.debug(msg)
        
def exit(msg='',code=1, doexit=True,pre='',post=''): 
    if msg != '': 
        sys.stderr.write(pre + 'ERROR: ' + msg + post + '\n')  
    Globals().logger.error(msg)
    if doexit: sys.exit(code)
