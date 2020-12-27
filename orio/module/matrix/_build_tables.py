#-----------------------------------------------------------------
# pycparser: _build_tables.py
#
# A dummy for generating the lexing/parsing tables and and
# compiling them into .pyc for faster execution in optimized mode.
# Also generates AST code from the configuration file.
# Should be called from the pycparser directory.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-----------------------------------------------------------------

# Insert '.' and '..' as first entries to the search path for modules.
# Restricted environments like embeddable python do not include the
# current working directory on startup.
#import sys
#sys.path[0:0] = ['.', '..']

# Generate matrix_ast.py
from ._ast_gen import ASTCodeGenerator
ast_gen = ASTCodeGenerator('_matrix_ast.cfg')
ast_gen.generate(open('matrix_ast.py', 'w'))

from orio.module.matrix.mparser import MParser

# Generates the tables
#
MParser(
    lex_optimize=False,
    yacc_debug=False,
    yacc_optimize=False)

# Load to compile into .pyc
#
#import lextab
#import yacctab
#import matrix_ast
