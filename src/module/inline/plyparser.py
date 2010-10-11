#!/usr/bin/env python
#
# File: $id$
# @Package: orio
# @version: $keRevision$
# @lastrevision: $Date$
# @modifiedby: $LastChangedBy$
# @lastmodified: $LastChangedDate$
#
# Description: Fortran 2003 parser (supports Fortran 77 and later)
# 
# Copyright (c) 2009 UChicago, LLC
# Produced at Argonne National Laboratory
# Author: Boyana Norris (norris@mcs.anl.gov)
#
# For detailed license information refer to the LICENSE file in the top-level
# source directory.

import sys, os
import orio.tool.ply.yacc as yacc
#import orio.main.parsers.flexer as lexer
import orio.module.inline.sublexer as lexer
from orio.main.util.globals import *
import orio.module.inline.subroutine as subroutine

f90reserved = ['IF', 'ELSE', 'FOR', 'TRANSFORM', 'NOT', 'AND', 'OR']

   
# Get the token map
tokens = lexer.FLexer.tokens
baseTypes = {}


# R201
def p_program(p):
    'program : program_unit_list'
    p[0] = p[1]
    pass

def p_subroutine_definition(p):
    'subroutine_definition: subroutine_header subroutine_body END SUBROUTINE subroutine_name'
    p[0] = subroutine.SubroutineDefinition(p[1], p[2])
    pass

def p_subroutine_header(p):
    'subroutine_header : SUBROUTINE subroutine_name LPAREN argument_list RPAREN'
    p[0] = subroutine.SubroutineHeader(name=p[2],args=p[4])
    pass

def p_argument_list(p):
    '''argument_list : ID
                    | argument_list COMMA ID
                    | empty
                    '''
    if len(p) > 1: p[1].append(p[3])
    p[0] = p[1]
    pass

def p_subroutine_body(p):
    '''subroutine_body: '''
    pass

def p_subroutine_name(p):
    'subroutine_name : ID'
    p[0] = p[1]
    

# Driver (regenerates parse table)
def setup_regen(debug = 1, outputdir='.'):
    global parser
    
    # Remove the old parse table
    parsetabfile = os.path.join(os.path.abspath(outputdir),'parsetab.py')
    try: os.remove(parsetabfile)
    except: pass

    parser = yacc.yacc(debug=debug, optimize=1, tabmodule='parsetab', write_tables=1, outputdir=os.path.abspath(outputdir))

    return parser

# Driver (does not regenerate parse table)
def setup(debug = 0, outputdir='.'):
    global parser

    parser = yacc.yacc(debug = debug, optimize=1, write_tables=0)
    return parser

def getParser(start_line_no):
    '''Create the parser for the annotations language'''

    # set the starting line number
    global __start_line_no
    __start_line_no = start_line_no

    # create the lexer and parser
    parser = yacc.yacc(method='LALR', debug=1)

    # return the parser
    return parser

if __name__ == '__main__':
    '''To regenerate the parse tables, invoke iparse.py with --regen as the last command-line
        option, for example:
            iparse.py somefile.sidl --regen
    '''
    #import visitor.printer
    #import visitor.commentsmerger
    import sys
    
    #import profile
    # Build the grammar
    #profile.run("yacc.yacc()")
    
    if sys.argv[-1] == '--regen':
        del sys.argv[-1]
        setup_regen(debug=0, outputdir=os.path.dirname(sys.argv[0]))
    else:
        setup()

    lex = lexer.SubroutineLexer()
    lex.build(optimize=1)                     # Build the lexer

    
    for i in range(1, len(sys.argv)):
        debug("[inliner parse] About to parse %s" % sys.argv[i])
        f = open(sys.argv[i],"r")
        s = f.read()
        f.close()
        # print "Contents of %s: %s" % (sys.argv[i], s)
        if s == '' or s.isspace(): sys.exit(0)
        
        #print 'Comments: \n', comments
        
        lex.reset()
        ast = parser.parse(s, lexer=lex.lexer, debug=0)
        debug('[inliner parse] Successfully parsed %s' % sys.argv[i])
        
        #printer = visitor.printer.Printer()
        #ast.accept(printer)
