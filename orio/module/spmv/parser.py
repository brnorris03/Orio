# 
# The parser for the SpMV transformation module
#

import re, sys
from tool.ZestyParser import *
from orio.main.util.globals import *

#--------------------------------------------------------------------------------

# a callback function for a single token
def f_token(p, r, c):
    line_no, col_no = p.coord(c)
    line_no += p.start_line_no - 1
    return (r, line_no)

#--------------------------------------------------------------------------------

# tokens
ID          = Token('[A-Za-z_][A-Za-z0-9_]*', group=0) >> f_token
EQUALS      = Token('=')
SEMI        = Token(';')

# right hand side expression
RHSEXP      = Token('[^;]+', group=0) >> f_token

# ignored strings (i.e. whitespaces)
SPACE       = Token('\s+')

#--------------------------------------------------------------------------------

# argument
def f_arg(p, r, c):
    line_no, col_no = p.coord(c)
    line_no += p.start_line_no - 1
    return (line_no, r[0], r[1])

p_arg = ((Skip(SPACE) + (ID ^ 'expected identifier') + Skip(SPACE) +
          (Omit(EQUALS) ^ 'expected equal sign') + Skip(SPACE) + RHSEXP + Skip(SPACE) +
          (Omit(SEMI) ^ 'expected semicolon'))
         >> f_arg)

#--------------------------------------------------------------------------------

# program
def f_program(p, r, c):
    return r[0]

p_program = ((TokenSeries(p_arg, skip=SPACE,
                          until=(Skip(SPACE) + EOF, 'not a valid argument')) +
              Skip(SPACE) + (EOF ^ 'not a valid argument')) 
             >> f_program)

#--------------------------------------------------------------------------------

class Parser:
    '''The parser of the SpMV module.'''

    def __init__(self):
        '''To instantiate a parser instance'''
        pass
        
    #----------------------------------------------------------------------------

    def parse(self, code, line_no):
        '''To parse the given code'''

        # remove all comments
        code = code + '\n'
        code = re.sub(r'#.*?\n', '\n', code)

        # if a blank code
        if code.strip() == '':
            return []

        # create the parser
        p = ZestyParser(code)

        # update the starting line number of the code
        p.start_line_no = line_no

        # parse the tuning specifications
        try:
            args = p.scan(p_program)
        except ParseError as e:
            err('orio.module.spmv.parser:  %s' % e)

        # return the arguments
        return args


