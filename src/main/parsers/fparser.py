#!/usr/bin/env python
#
# File: $id$
# @Package: orio
# @version: $Revision$
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
#
# Contain syntax and grammar specifications of the annotations language
#

import sys, os
import ast, tool.ply.yacc
import main.parsers.flexer as lexer

# Get the token map
tokens = lexer.tokens
baseTypes = {}

# annotation
def p_annotation(p):
    'annotation : statement_list_opt'
    p[0] = p[1]
    
# statement-list
def p_statement_list_opt_1(p):
    'statement_list_opt :'
    p[0] = []
    
def p_statement_list_opt_2(p):
    'statement_list_opt : statement_list'
    p[0] = p[1]
    
def p_statement_list_1(p):
    'statement_list : statement'
    p[0] = [p[1]]
    
def p_statement_list_2(p):
    'statement_list : statement_list statement'
    p[1].append(p[2])
    p[0] = p[1]
    
# statement:
def p_statement(p):
    '''statement : expression_statement
                 | compound_statement
                 | selection_statement
                 | iteration_statement
                 | transformation_statement
                 '''
    p[0] = p[1]
    
# expression-statement:
def p_expression_statement(p):
    'expression_statement : expression_opt SEMI'
    p[0] = ast.ExpStmt(p[1], p.lineno(1) + __start_line_no - 1)

# compound-statement:
def p_compound_statement(p):
    'compound_statement : LBRACE statement_list_opt RBRACE'
    p[0] = ast.CompStmt(p[2], p.lineno(1) + __start_line_no - 1)
    
# selection-statement
# Note:
#   This results in a shift/reduce conflict. However, such conflict is not harmful
#   because PLY resolves such conflict in favor of shifting.
def p_selection_statement_1(p):
    'selection_statement : IF LPAREN expression RPAREN statement'
    p[0] = ast.IfStmt(p[3], p[5], None, p.lineno(1) + __start_line_no - 1)
    
def p_selection_statement_2(p):
    'selection_statement : IF LPAREN expression RPAREN statement ELSE statement'
    p[0] = ast.IfStmt(p[3], p[5], p[7], p.lineno(1) + __start_line_no - 1)

# iteration-statement
def p_iteration_statement(p):
    'iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement'
    p[0] = ast.ForStmt(p[3], p[5], p[7], p[9], p.lineno(1) + __start_line_no - 1)

# transformation-statement
def p_transformation_statement(p):
    'transformation_statement : TRANSFORM ID LPAREN transformation_argument_list_opt RPAREN statement'
    p[0] = ast.TransformStmt(p[2], p[4], p[6], p.lineno(1) + __start_line_no - 1)

# transformation-argument-list
def p_transformation_argument_list_opt_1(p):
    'transformation_argument_list_opt :'
    p[0] = []

def p_transformation_argument_list_opt_2(p):
    'transformation_argument_list_opt : transformation_argument_list'
    p[0] = p[1]

def p_transformation_argument_list_1(p):
    'transformation_argument_list : transformation_argument'
    p[0] = [p[1]]

def p_transformation_argument_list_2(p):
    'transformation_argument_list : transformation_argument_list COMMA transformation_argument'
    p[1].append(p[3])
    p[0] = p[1]

# transformation-argument
def p_transformation_argument(p):
    'transformation_argument : ID EQUALS py_expression'
    p[0] = [p[1], p[3], p.lineno(1) + __start_line_no - 1]

# expression:
def p_expression_opt_1(p):
    'expression_opt :'
    p[0] = None

def p_expression_opt_2(p):
    'expression_opt : expression'
    p[0] = p[1]

def p_expression_1(p):
    'expression : assignment_expression'
    p[0] = p[1]

def p_expression_2(p):
    'expression : expression COMMA assignment_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.COMMA, p.lineno(1) + __start_line_no - 1)

# assignment_expression:
def p_assignment_expression_1(p):
    'assignment_expression : logical_or_expression'
    p[0] = p[1]

def p_assignment_expression_2(p):
    'assignment_expression : unary_expression assignment_operator assignment_expression'
    if (p[2] == '='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ_ASGN, p.lineno(1) + __start_line_no - 1)
    elif p[2] in ('*=', '/=', '%=', '+=', '-='):
        lhs = p[1].replicate()
        rhs = None
        if (p[2] == '*='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '/='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '%='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '+='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD, p.lineno(1) + __start_line_no - 1)
        elif (p[2] == '-='):
            rhs = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB, p.lineno(1) + __start_line_no - 1)
        else:
            print 'internal error: missing case for assignment operator'
            sys.exit(1)
        p[0] = ast.BinOpExp(lhs, rhs, ast.BinOpExp.EQ_ASGN, p.lineno(1) + __start_line_no - 1)
    else:
        print 'internal error: unknown assignment operator'
        sys.exit(1)

# assignment-operator:
def p_assignment_operator(p):
    '''assignment_operator : EQUALS
                           | TIMESEQUAL
                           | DIVEQUAL
                           | MODEQUAL
                           | PLUSEQUAL
                           | MINUSEQUAL
                           '''
    p[0] = p[1]

# logical-or-expression
def p_logical_or_expression_1(p):
    'logical_or_expression : logical_and_expression'
    p[0] = p[1]

def p_logical_or_expression_2(p):
    'logical_or_expression : logical_or_expression LOR logical_and_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LOR, p.lineno(1) + __start_line_no - 1)

# logical-and-expression
def p_logical_and_expression_1(p):
    'logical_and_expression : equality_expression'
    p[0] = p[1]

def p_logical_and_expression_2(p):
    'logical_and_expression : logical_and_expression LAND equality_expression'
    p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LAND, p.lineno(1) + __start_line_no - 1)

# equality-expression:
def p_equality_expression_1(p):
    'equality_expression : relational_expression'
    p[0] = p[1]

def p_equality_expression_2(p):
    'equality_expression : equality_expression equality_operator relational_expression'
    if p[2] == '==':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.EQ, p.lineno(1) + __start_line_no - 1)
    elif p[2] == '!=':
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.NE, p.lineno(1) + __start_line_no - 1)
    else:
        print 'internal error: unknown equality operator'
        sys.exit(1)

# equality-operator:
def p_equality_operator(p):
    '''equality_operator : EQ
                         | NE'''
    p[0] = p[1]

# relational-expression:
def p_relational_expression_1(p):
    'relational_expression : additive_expression'
    p[0] = p[1]

def p_relational_expression_2(p):
    'relational_expression : relational_expression relational_operator additive_expression'
    if (p[2] == '<'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LT, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '>'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GT, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '<='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.LE, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '>='):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.GE, p.lineno(1) + __start_line_no - 1)
    else:
        print 'internal error: unknown relational operator'
        sys.exit(1)
        
# relational-operator
def p_relational_operator(p):
    '''relational_operator : LT
                           | GT
                           | LE
                           | GE'''
    p[0] = p[1]

# additive-expression
def p_additive_expression_1(p):
    'additive_expression : multiplicative_expression'
    p[0] = p[1]

def p_additive_expression_2(p):
    'additive_expression : additive_expression additive_operator multiplicative_expression'
    if (p[2] == '+'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.ADD, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '-'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.SUB, p.lineno(1) + __start_line_no - 1)
    else:
        print 'internal error: unknown additive operator' 
        sys.exit(1)

# additive-operator:
def p_additive_operator(p):
    '''additive_operator : PLUS
                         | MINUS'''
    p[0] = p[1]

# multiplicative-expression
def p_multiplicative_expression_1(p):
    'multiplicative_expression : unary_expression'
    p[0] = p[1]

def p_multiplicative_expression_2(p):
    'multiplicative_expression : multiplicative_expression multiplicative_operator unary_expression'
    if (p[2] == '*'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MUL, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '/'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.DIV, p.lineno(1) + __start_line_no - 1)
    elif (p[2] == '%'):
        p[0] = ast.BinOpExp(p[1], p[3], ast.BinOpExp.MOD, p.lineno(1) + __start_line_no - 1)
    else:
        print 'internal error: unknown multiplicative operator'
        sys.exit(1)

# multiplicative-operator
def p_multiplicative_operator(p):
    '''multiplicative_operator : TIMES
                               | DIVIDE
                               | MOD'''
    p[0] = p[1]

# unary-expression:
def p_unary_expression_1(p):
    'unary_expression : postfix_expression'
    p[0] = p[1]

def p_unary_expression_2(p):
    'unary_expression : PLUSPLUS unary_expression'
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_INC, p.lineno(1) + __start_line_no - 1)

def p_unary_expression_3(p):
    'unary_expression : MINUSMINUS unary_expression'
    p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PRE_DEC, p.lineno(1) + __start_line_no - 1)

def p_unary_expression_4(p):
    'unary_expression : unary_operator unary_expression'
    if p[1] == '+':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.PLUS, p.lineno(1) + __start_line_no - 1)
    elif p[1] == '-':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.MINUS, p.lineno(1) + __start_line_no - 1)
    elif p[1] == '!':
        p[0] = ast.UnaryExp(p[2], ast.UnaryExp.LNOT, p.lineno(1) + __start_line_no - 1)
    else:
        print 'internal error: unknown unary operator'
        sys.exit(1)

# unary-operator
def p_unary_operator(p):
    '''unary_operator : PLUS
                      | MINUS
                      | LNOT '''
    p[0] = p[1]

# postfix-expression
def p_postfix_expression_1(p):
    'postfix_expression : primary_expression'
    p[0] = p[1]

def p_postfix_expression_2(p):
    'postfix_expression : postfix_expression LBRACKET expression RBRACKET'
    p[0] = ast.ArrayRefExp(p[1], p[3], p.lineno(1) + __start_line_no - 1)

def p_postfix_expression_3(p):
    'postfix_expression : postfix_expression LPAREN argument_expression_list_opt RPAREN'
    p[0] = ast.FunCallExp(p[1], p[3], p.lineno(1) + __start_line_no - 1)

def p_postfix_expression_4(p):
    'postfix_expression : postfix_expression PLUSPLUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_INC, p.lineno(1) + __start_line_no - 1)

def p_postfix_expression_5(p):
    'postfix_expression : postfix_expression MINUSMINUS'
    p[0] = ast.UnaryExp(p[1], ast.UnaryExp.POST_DEC, p.lineno(1) + __start_line_no - 1)

# primary-expression
def p_primary_expression_1(p):
    'primary_expression : ID'
    p[0] = ast.IdentExp(p[1], p.lineno(1) + __start_line_no - 1)

def p_primary_expression_2(p):
    'primary_expression : ICONST'
    val = int(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.INT, p.lineno(1) + __start_line_no - 1)

def p_primary_expression_3(p):
    'primary_expression : FCONST'
    val = float(p[1])
    p[0] = ast.NumLitExp(val, ast.NumLitExp.FLOAT, p.lineno(1) + __start_line_no - 1)

def p_primary_expression_4(p):
    'primary_expression : SCONST_D'
    p[0] = ast.StringLitExp(p[1], p.lineno(1) + __start_line_no - 1)

def p_primary_expression_5(p):
    '''primary_expression : LPAREN expression RPAREN'''
    p[0] = ast.ParenthExp(p[2], p.lineno(1) + __start_line_no - 1)

# argument-expression-list:
def p_argument_expression_list_opt_1(p):
    'argument_expression_list_opt :'
    p[0] = []
     
def p_argument_expression_list_opt_2(p):
    'argument_expression_list_opt : argument_expression_list'
    p[0] = p[1]

def p_argument_expression_list_1(p):
    'argument_expression_list : assignment_expression' 
    p[0] = [p[1]]

def p_argument_expression_list_2(p):
    'argument_expression_list : argument_expression_list COMMA assignment_expression' 
    p[1].append(p[3])
    p[0] = p[1]

# grammatical error
def p_error(p):
    print 'error:%s: grammatical error: "%s"' % ((p.lineno + __start_line_no - 1), p.value)
    sys.exit(1)

#------------------------------------------------

# Below is a grammar subset of Python expression
# py-expression
def p_py_expression_1(p):
    'py_expression : py_m_expression'
    p[0] = p[1]

def p_py_expression_2(p):
    'py_expression : py_conditional_expression'
    p[0] = p[1]

# py-expression-list
def p_py_expression_list_opt_1(p):
    'py_expression_list_opt : '
    p[0] = ''

def p_py_expression_list_opt_2(p):
    'py_expression_list_opt : py_expression_list'
    p[0] = p[1]

def p_py_expression_list_1(p):
    'py_expression_list : py_expression'
    p[0] = p[1]

def p_py_expression_list_2(p):
    'py_expression_list : py_expression_list COMMA py_expression'
    p[0] = p[1] + p[2] + p[3]

# py-conditional-expression
def p_py_conditional_expression(p):
    'py_conditional_expression : py_m_expression IF py_m_expression ELSE py_m_expression'
    p[0] = p[1] + ' ' + p[2] + ' ' + p[3] + ' ' + p[4] + ' ' + p[5]

# py-m-expression
def p_py_m_expression_1(p):
    'py_m_expression : py_u_expression'
    p[0] = p[1]

def p_py_m_expression_2(p):
    'py_m_expression : py_m_expression py_binary_operator py_u_expression'
    p[0] = p[1] + ' ' +  p[2] + ' ' + p[3]

# py-binary-operator
def p_py_binary_operator(p):
    '''py_binary_operator : PLUS
                          | MINUS
                          | TIMES
                          | DIVIDE
                          | MOD
                          | LT
                          | GT
                          | LE
                          | GE
                          | EQ
                          | NE
                          | AND
                          | OR'''
    p[0] = p[1]

# py-u-expression
def p_py_u_expression_1(p):
    'py_u_expression : py_primary'
    p[0] = p[1]

def p_py_u_expression_2(p):
    '''py_u_expression : PLUS py_u_expression
                       | MINUS py_u_expression
                       | NOT py_u_expression'''
    p[0] = p[1] + ' ' + p[2]

# py-primary
def p_py_primary(p):
    '''py_primary : py_atom
                  | py_subscription
                  | py_attribute_ref
                  | py_call
                  | py_list_display
                  | py_dict_display'''
    p[0] = p[1]

# py-subscription
def p_py_subscription(p):
    'py_subscription : py_primary LBRACKET py_expression_list RBRACKET'
    p[0] = p[1] + p[2] + p[3] + p[4]

# py-attribute-ref
def p_py_attribute_ref(p):
    'py_attribute_ref : py_primary PERIOD ID'
    p[0] = p[1] + p[2] + p[3]

# py-call
def p_py_call(p):
    'py_call : py_primary LPAREN py_expression_list_opt RPAREN'
    p[0] = p[1] + p[2] + p[3] + p[4]

# py-list-display
def p_py_list_display(p):
    'py_list_display : LBRACKET py_expression_list_opt RBRACKET'
    p[0] = p[1] + p[2] + p[3]

# py-dict-display
def p_py_dict_display(p):
    'py_dict_display : LBRACE py_key_datum_list_opt RBRACE'
    p[0] = p[1] + p[2] + p[3]

# py-key-datum-list
def p_py_key_datum_list_opt_1(p):
    'py_key_datum_list_opt : '
    p[0] = ''

def p_py_key_datum_list_opt_2(p):
    'py_key_datum_list_opt : py_key_datum_list'
    p[0] = p[1]

def p_py_key_datum_list_1(p):
    'py_key_datum_list : py_key_datum'
    p[0] = p[1]

def p_py_key_datum_list_2(p):
    'py_key_datum_list : py_key_datum_list COMMA py_key_datum'
    p[0] = p[1] + p[2] + p[3]

# py-key-datum
def p_py_key_datum(p):
    'py_key_datum : py_expression COLON py_expression'
    p[0] = p[1] + p[2] + p[3]

# py-atom
def p_py_atom_1(p):
    '''py_atom : ID
               | ICONST
               | FCONST
               | SCONST_D
               | SCONST_S'''
    p[0] = p[1]

def p_py_atom_2(p):
    'py_atom : LPAREN py_expression_list_opt RPAREN'
    p[0] = p[1] + p[2] + p[3]

#------------------------------------------------

def getParser(start_line_no):
    '''Create the parser for the annotations language'''

    # set the starting line number
    global __start_line_no
    __start_line_no = start_line_no

    # create the lexer and parser
    lexer = tool.ply.lex.lex()
    parser = tool.ply.yacc.yacc(method='LALR', debug=0)

    # return the parser
    return parser

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

    parser = parse.ply.yacc.yacc(debug=debug, optimize=1, tabmodule='parsetab', write_tables=1, outputdir=os.path.abspath(outputdir))

    return parser


if __name__ == '__main__':
    '''To regenerate the parse tables, invoke iparse.py with --regen as the last command-line
        option, for example:
            iparse.py somefile.sidl --regen
    '''
    #import visitor.printer
    
    DEBUGSTREAM = Devnull()

    if True or sys.argv[-1] == '--regen':
        del sys.argv[-1]
        DEBUGSTREAM = sys.stderr
        setup_regen(debug=0, outputdir=os.path.dirname(sys.argv[0]))
    else:
        setup()

    lex = lexer.FLexer()
    lex.build(optimize=1)                     # Build the lexer

    for i in range(1, len(sys.argv)):
        fname = sys.argv[i]
        print >>DEBUGSTREAM, "[fparser] About to parse %s" % fname
        f = open(fname,"r")
        s = f.read()
        f.close()
        # print "Contents of %s: %s" % (fname, s)
        if s == '' or s.isspace(): sys.exit(0)
        if not s.endswith('\n'): 
            print 'Orio WARNING: file does not end with newline.'
            s += '\n'
        
        lex.reset(fname)
        ast = parser.parse(s, lexer=lex.lexer, debug=0)
        print >>DEBUGSTREAM, '[iparse] Successfully parsed %s' % fname

        
        #printer = visitor.printer.Printer()
        #ast.accept(printer)


