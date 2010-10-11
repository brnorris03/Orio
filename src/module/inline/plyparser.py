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
   
# Get the token map
tokens = lexer.SubroutineLexer.tokens
baseTypes = {}

def p_file(p):
    '''file : stuff_we_dont_care_about
            | subroutine_definition_list
    '''
    p[0] = p[1]
    
def p_stuff_we_dont_care_about(p):
    '''stuff_we_dont_care_about : idlist
    '''
    p[0] = None
    
def p_subroutine_definition_list(p):
    '''subroutine_definition_list : subroutine_definition
                                | subroutine_definition_list subroutine_definition
                                | empty
                                '''
    p[0] = []
    
def p_subroutine_definition(p):
    'subroutine_definition : subroutine_header varref_list ENDSUBROUTINE'
    p[0] = subroutine.SubroutineDefinition(p[1], p[2], p[3])
    pass

def p_subroutine_header(p):
    'subroutine_header : SUBROUTINE subroutine_name LPAREN argument_list RPAREN'
    p[0] = (p[2],p[4])

def p_argument_list(p):
    '''argument_list : ID
                    | argument_list COMMA ID
                    | empty
                    '''
    if len(p) > 2: 
        p[1].append(p[3])
        p[0] = p[1]
    elif p[1] != None:
        p[0] = [p[1]]
    else:
        p[0] = []
    pass

def p_varref_list(p):
    '''varref_list : ID
                    | varref_list ID
                    | empty
                    '''
    if len(p) > 2:
        p[1].append((p[2],p.lexspan(2)))
        p[0] = p[1]
    elif p[1]:
        p[0] = [(p[1],p.lexspan(1))]
    else:
        p[0] = []
      
# Lists of identifiers outside subroutines  
def p_idlist(p):
    '''idlist : ID
            | idlist ID
            | empty
            '''
    p[0] = []

def p_subroutine_name(p):
    'subroutine_name : ID'
    p[0] = p[1]
    
def p_empty(p):
    'empty : '
    p[0] = None
    
def p_error(t):
    if t:
        line,col = find_column(t.lexer.lexdata,t)
        pos = (col-1)*' '
        err("[orio.module.inline.fparser] unexpected symbol '%s' at line %s, column %s:\n\t%s\n\t%s^" \
            % (t.value, t.lexer.lineno, col, line, pos))
    else:
        err("[orio.module.inline.fparser] internal error, please email source code to norris@mcs.anl.gov")
    
# Compute column. 
#     input is the input text string
#     token is a token instance
def find_column(input,token):
    i = token.lexpos
    startline = input[:i].rfind('\n')
    endline = startline + input[startline+1:].find('\n') 
    line = input[startline+1:endline+1]
    while i > 0:
        if input[i] == '\n': break
        i -= 1
    column = (token.lexpos - i)
    return line, column


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
        sub = parser.parse(s, lexer=lex.lexer, debug=0)
        debug('[inliner parse] Successfully parsed %s' % sys.argv[i])
        print str(sub)
        
        #printer = visitor.printer.Printer()
        #ast.accept(printer)
