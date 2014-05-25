#!/usr/bin/env python
'''
Created on Aug 26, 2011

@author: norris
'''


import sys,os,re
import orio.tool.ply.lex
import orio.tool.ply.yacc
import orio.module.matrix.lexer as lexer
from orio.main.util.globals import *


# Get the token map
tokens    = lexer.MatrixLexer.tokens
baseTypes = {}


__matrix_language_vars = {}
__matrix_language_scalar_name_re = re.compile(r'[a-n]\w*')
__matrix_language_typeinference = True

# List of parser errors.
errors = []

# input
def p_prog_1(p):
    'prog : ID IN param_list INOUT param_list OUT param_list LBRACE stmt_list RBRACE'
    p[0] = p[1]
    # TODO

def p_prog_2(p):
    '''prog : ID IN param_list INOUT param_list LBRACE stmt_list RBRACE
            | ID INOUT param_list OUT param_list LBRACE stmt_list RBRACE
            | ID IN param_list OUT param_list LBRACE stmt_list RBRACE
    '''
    p[0] = p[1]
    # TODO
    
def p_prog_3(p):
    'prog : stmt_list'
    # Accept partial programs, do type inference
    p[0] = p[1]
    
def p_param_list(p):
    '''param_list : param
                | param_list COMMA param
    '''
    if len(p) > 2:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]
        
def p_param(p):
    'param : ID COLON type'
    p[0] = (p[1],p[3])
    __matrix_language_vars[p[1]] = p[3]

def p_type(p):
    '''type : MATRIX LPAREN attrib_list RPAREN
            | VECTOR LPAREN attrib_list RPAREN
            | SCALAR
    '''
    if len(p) > 2:
        p[0] = (p[1],p[3])
    else:
        p[0] = (p[1],None)

def p_attrib_list(p):
    '''attrib_list : attrib
                | attrib_list COMMA attrib
                | empty
                '''
    if len(p) > 2:
        p[0] = p[1] + [p[2]]
    else:
        if not p[1]: p[0] = []
        else: p[0] = [p[1]]


def p_attrib(p):
    '''attrib : ROW
                | ORIENTATION EQUALS ROW
                | COLUMN
                | ORIENTATION EQUALS COLUMN
                | GENERAL
                | FORMAT EQUALS GENERAL
                | TRIANGULAR
                | FORMAT EQUALS TRIANGULAR
                | UPPER
                | UPLO EQUALS UPPER
                | LOWER
                | UPLO EQUALS LOWER
                | UNIT
                | DIAG EQUALS UNIT
                | NONUNIT
                | DIAG EQUALS NONUNIT
                '''
    if len(p) > 3:
        p[0] = (p[1],p[3])
    else:
        p[0] = (None,p[1])


def p_stmt_list(p):
    '''stmt_list : stmt
                | stmt_list stmt
                | empty
                '''
    if len(p) > 2:
        p[0] = p[1] + [p[2]]
    else:
        if not p[1]: p[0] = []
        else: p[0] = [p[1]]
    
def p_stmt(p):
    'stmt : ID EQUALS expr'
    p[0] = (p[1], p[3])
    
def p_expr_1(p):
    '''expr : FCONST
            | ICONST
    '''
    # I don't care, only want to capture variables
    p[0] = (p[1],False)
    
def p_expr_2(p):
    'expr : ID'
    # True indicates that this is a variable
    p[0] = (p[1],True)

def p_expr_3(p):
    '''expr : expr PLUS expr
            | expr MINUS expr
            | expr TIMES expr
            | MINUS expr
            | expr SQUOTE
            | LPAREN expr RPAREN
            '''
    
    if len(p) > 3:
        expressions = [p[1],p[3]]
    elif p[1] == '\'':
        expressions = [p[1]]
    else:
        expressions = [p[2]]
    for exp in expressions:
        # exp is a tuple, second arg is True if variable
        if len(exp)>1 and exp[1]:
            # Variable name is exp[0]
            var = exp[0]
            
            # Simple type inference in expressions
            if __matrix_language_typeinference and not var in __matrix_language_vars.keys():
                if var[0].isupper():
                    type = 'matrix'
                    orientation = 'row' # default
                else:
                    # Use Fortran implicit rules to decide whether variable 
                    # is scalar or vector -- a-n scalar, o-z vector
                    if __matrix_language_scalar_name_re.match(var):
                        type = 'scalar'
                    else:
                        type = 'vector'
                    orientation = None
                __matrix_language_vars[exp[0]] = (type,orientation)
    p[0] = [] # TODO: eventually may want to store expressions


def p_empty(p):
    'empty : '
    p[0] = None
    
def p_error(p):
    global errors
    errors.append("Syntax error at %s on line number %d." % (p.value, p.lineno))

def getParser(start_line_no):
    '''Create the parser for the annotations language'''

    # set the starting line number
    global __start_line_no
    __start_line_no = start_line_no

    # create the lexer and parser
    matrixlexer = orio.tool.ply.lex.lex(debug=0, optimize=1)
    matrixparser = orio.tool.ply.yacc.yacc(method='LALR', debug=1, optimize=0)

    # return the parser
    return matrixparser

def getVars(): 
  return __matrix_language_vars

# Driver (regenerates parse table)
def setup_regen(debug = 1, outputdir='.'):
    global parser
    
    # Remove the old parse table
    parsetabfile = os.path.join(os.path.abspath(outputdir),'MatrixParser.parsetab.py')
    try: os.remove(parsetabfile)
    except: pass

    parser = orio.tool.ply.yacc.yacc(debug=debug, optimize=1, tabmodule='MatrixParser.parsetab', write_tables=1, outputdir=os.path.abspath(outputdir))

    return parser

# Driver (does not regenerate parse table)
def setup(debug = 0, outputdir='.', typeinference=True):
    global parser, __matrix_language_typeinference

    parser = orio.tool.ply.yacc.yacc(debug = debug, optimize=1, write_tables=0)
    __matrix_language_typeinference = typeinference
    return parser
    
if __name__ == '__main__':
    '''To regenerate the parse tables, invoke parser.py with --regen as the last command-line
        option, for example:
            parser.py somefile.m --regen
    '''
   
    if sys.argv[-1] == '--regen':
        # Forse regeneration of the parse tables
        del sys.argv[-1]
        setup_regen(debug=0, outputdir=os.path.dirname(sys.argv[0]))
    else:
        setup()
    
    #matrixlexer = MatrixParser.lexer.MatrixLexer(debug=1, optimize=0)
    import lexer
    matrixlexer = lexer.MatrixLexer(debug=0, optimize=1)
    matrixlexer.build()

    for i in range(1, len(sys.argv)):
        print >>sys.stderr, "[parse] About to parse %s" % sys.argv[i]
        os.system('cat %s' % sys.argv[i])
        f = open(sys.argv[i],"r")
        s = f.read()
        f.close()
        # print "Contents of %s: %s" % (sys.argv[i], s)
        if s == '' or s.isspace(): sys.exit(0)
        if not s.endswith('\n'): 
            print 'WARNING: file does not end with newline.'
            s += '\n'

        theresult = parser.parse(s, lexer=matrixlexer.lexer, debug=1)
        print >>sys.stderr, '[parser] Successfully parsed %s' % sys.argv[i]

        print 'All variables and their types:'
        for key,val in __matrix_language_vars.items():
            print "%s : %s" % (key,val)

