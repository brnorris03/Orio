#!/usr/bin/env python
'''
Created on Aug 26, 2011

@author: norris
'''

import sys, os, re
import orio.tool.ply.lex
import orio.tool.ply.yacc
import orio.module.matrix.mlexer as lexer
from orio.main.util.globals import *

from mlexer import *
from . import matrix_ast as m_ast
from plyparser import PLYParser
from orio.tool.ply import yacc
from orio.module.matrix.plyparser import PLYParser, Coord, ParseError, parameterized, template


@template
class MParser(PLYParser):
    """
    Matrix language (BTO-based) parser
    """

    def __init__(
            self,
            lex_optimize=False,
            lexer=MLexer,
            lextab='lextab',
            yacc_optimize=False,
            yacc_debug=True,
            yacctab='yacctab',
            taboutputdir='',
            printToStderr=True):
        """ Create a new MParser.

            Some arguments for controlling the debug/optimization
            level of the parser are provided. The defaults are
            tuned for release/performance mode.
            The simple rules for using them are:
            *) When tweaking MParser/MLexer, set these to False
            *) When releasing a stable parser, set to True

            lex_optimize:
                Set to False when you're modifying the lexer.
                Otherwise, changes in the lexer won't be used, if
                some lextab.py file exists.
                When releasing with a stable lexer, set to True
                to save the re-generation of the lexer table on
                each run.

            lexer:
                Set this parameter to define the lexer to use if
                you're not using the default MLexer.

            lextab:
                Points to the lex table that's used for optimized
                mode. Only if you're modifying the lexer and want
                some tests to avoid re-generating the table, make
                this point to a local lex table file (that's been
                earlier generated with lex_optimize=True)

            yacc_optimize:
                Set to False when you're modifying the parser.
                Otherwise, changes in the parser won't be used, if
                some parsetab.py file exists.
                When releasing with a stable parser, set to True
                to save the re-generation of the parser table on
                each run.

            yacctab:
                Points to the yacc table that's used for optimized
                mode. Only if you're modifying the parser, make
                this point to a local yacc table file

            yacc_debug:
                Generate a parser.out file that explains how yacc
                built the parsing table from the grammar.

            taboutputdir:
                Set this parameter to control the location of generated
                lextab and yacctab files.
        """

        self.mlex = lexer(
            error_func = self._lex_error_func,
            on_lbrace_func=self._lex_on_lbrace_func,
            on_rbrace_func=self._lex_on_rbrace_func,
            type_lookup_func=self._lex_type_lookup_func)

        self.mlex.build(
            optimize=lex_optimize,
            lextab=lextab,
            outputdir=taboutputdir,
            printToStderr=printToStderr)
        self.tokens = self.mlex.tokens

        self.mparser = yacc.yacc(
            module=self,
            start="translation_unit_or_empty",
            tabmodule=yacctab,
            debug=yacc_debug,
            optimize=yacc_optimize,
            outputdir=taboutputdir)

        self.errorlog = []
        self.parser_errors = []
        self.debug = yacc_debug
        # =================================================================

        # Get the token map
        self.baseTypes = {}

        # this is a simplistic symbol table, a dictionary with
        #  keys: function names
        #  values: dictionary of variable names and types (and intent for parameters)
        self.matrix_language_vars = {}
        self.matrix_language_scalar_name_re = re.compile(r'[a-n]\w*')
        self.matrix_language_typeinference = True

        # self.mlex.errors = matrixparser.errors


    def _processString(self, input=''):
        if input == '' or input.isspace():
            return None
        else:
            return self.parser.parse(input, lexer=self.mlex.lexer, debug=self.debug)


    def _processFile(self, inputfile=''):
        if not os.path.exists(inputfile):
            self.error(0, "Input file not found: %s" % inputfile)
            return None
        else:
            f = open(inputfile, "r")
            s = f.read()
            f.close()

            return self.mparser.parse(s, lexer=self.mlex.lexer, debug=self.debug)


    def error(self, msg):
        self.errorlog.append(msg)
        if self.printToStderr:
           sys.stderr.write(msg+"\n")


    def _lex_error_func(self, msg, line, column):
        self._parse_error(msg, self._coord(line, column))

    def _lex_on_lbrace_func(self):
        self._push_scope()

    def _lex_on_rbrace_func(self):
        self._pop_scope()

    def _lex_type_lookup_func(self, name):
        """ Looks up types that were previously defined with
            typedef.
            Passed to the lexer for recognizing identifiers that
            are types.
        """
        is_type = self._is_type_in_scope(name)
        return is_type

    def _build_declarations(self,intent, param_list):
        'param_list: a list of variable names'
        pass

    #====================================================================
    # Parsing rules

    # input
    def p_translation_unit_or_empty(self, p):
        """ translation_unit_or_empty   : translation_unit
                            | empty
        """
        if p[1] is None:
            p[0] = m_ast.FileAST([])
        else:
            p[0] = m_ast.FileAST(p[1])

    def p_translation_unit_1(self, p):
        """ translation_unit    : external_declaration
        """
        # Note: external_declaration is already a list
        #
        p[0] = p[1]

    def p_translation_unit_2(self, p):
        """ translation_unit    : translation_unit external_declaration
        """
        p[1].extend(p[2])
        p[0] = p[1]


    # Declarations always come as lists (because they can be
    # several in one line), so we wrap the function definition
    # into a list as well, to make the return value of
    # external_declaration homogeneous.
    #
    def p_external_declaration_1(self, p):
        """ external_declaration    : function_definition
        """
        p[0] = [p[1]]

    def p_external_declaration_2(self, p):
        """ external_declaration    : declaration
        """
        p[0] = p[1]

    def p_external_declaration_3(self, p):
        """ external_declaration    : SEMI
        """
        p[0] = []

    # In function definitions, the declarator can be followed by
    # a declaration list
    # ID IN param_list INOUT param_list OUT param_list compound_stmt
    #
    def p_function_definition_1(self, p):
        """ function_definition : ID param_list_opt compound_statement
        """

        # Create entry in symbol table (dictionary)
        if not p[1] in self.matrix_language_vars.keys():
            self.matrix_language_vars[p[1]] = {}

        # no declaration specifiers - 'int' becomes the default type
        spec = dict(
            qual=[],
            storage=[],
            type=[m_ast.IdentifierType(['void'],
                                       coord=self._token_coord(p, 1))],
            function=[])

        p[0] = self._build_function_definition(
            spec=spec,
            decl=p[1],
            param_decls=p[2],
            body=p[3])

    def p_function_definition_2(self, p):
        """ function_definition : type_specifier id_declarator param_list_opt compound_statement
        """
        spec = p[1]

        p[0] = self._build_function_definition(
            spec=spec,
            decl=p[2],
            param_decls=p[3],
            body=p[4])

#    def p_function_definition_3(self, p):
#        """ function_definition : compound_statement"""
        # Accept partial programs, do type inference
#        p[0] = p[1]
        # TODO

    def p_param_list_opt(self, p):
        """ declaration_list    : param_declaration
                                | param_list_opt param_declaration
                                | empty
        """
        if not p[1]:
            p[0] = []
        elif len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_param_declaration(self, p):
        """ param_declaration   : IN param_list
                                | INOUT param_list
                                | OUT param_list
        """
        intent = p[1]
        self._build_declarations(intent=intent, param_list=p[2])


    def p_param_list(self, p):
        """param_list : param
                    | param_list COMMA param
        """
        if len(p) > 2:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]


    def p_param(self, p):
        """param : ID COLON type"""
        p[0] = (p[1], p[3])
        self.matrix_language_vars[p[1]] = p[3]

    def p_type_specifier(self, p):
        """type : MATRIX LPAREN attrib_list RPAREN
                | VECTOR LPAREN attrib_list RPAREN
                | SCALAR
        """
        if len(p) > 2:
            p[0] = (p[1], p[3])
        else:
            p[0] = (p[1], None)


    def p_attrib_list(self, p):
        """attrib_list : attrib
                    | attrib_list COMMA attrib
                    | empty
                    """
        if len(p) > 2:
            p[0] = p[1] + [p[2]]
        else:
            if not p[1]:
                p[0] = []
            else:
                p[0] = [p[1]]


    def p_attrib(self, p):
        """attrib : ROW
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
                    """
        if len(p) > 3:
            p[0] = (p[1], p[3])
        else:
            p[0] = (None, p[1])


    def p_stmt_list(self,p):
        """stmt_list : stmt
                    | stmt_list stmt
                    | empty
                    """
        if len(p) > 2:
            p[0] = p[1] + [p[2]]
        else:
            if not p[1]:
                p[0] = []
            else:
                p[0] = [p[1]]


    def p_stmt(self, p):
        """stmt : ID EQUALS expr"""
        p[0] = (p[1], p[3])


    def p_expr_1(self, p):
        """expr : FCONST
                | ICONST
        """
        # I don"""t care, only want to capture variables
        p[0] = (p[1], False)


    def p_expr_2(self, p):
        """expr : ID"""
        # True indicates that this is a variable
        p[0] = (p[1], True)


    def p_expr_3(self, p):
        """expr : expr PLUS expr
                | expr MINUS expr
                | expr TIMES expr
                | MINUS expr
                | expr SQUOTE
                | LPAREN expr RPAREN
                """

        if len(p) > 3:
            expressions = [p[1], p[3]]
        elif p[1] == '\'':
            expressions = [p[1]]
        else:
            expressions = [p[2]]
        for exp in expressions:
            # exp is a tuple, second arg is True if variable
            if len(exp) > 1 and exp[1]:
                # Variable name is exp[0]
                var = exp[0]

                # Simple type inference in expressions
                if self.matrix_language_typeinference and not var in self.matrix_language_vars.keys():
                    if var[0].isupper():
                        type = 'matrix'
                        orientation = 'row'  # default
                    else:
                        # Use Fortran implicit rules to decide whether variable
                        # is scalar or vector -- a-n scalar, o-z vector
                        if __matrix_language_scalar_name_re.match(var):
                            type = 'scalar'
                        else:
                            type = 'vector'
                        orientation = None
                    self.matrix_language_vars[exp[0]] = (type, orientation)
        p[0] = []  # TODO: eventually may want to store expressions


    def p_empty(self, p):
        """empty : """
        p[0] = None


    def p_error(self, p):
        self.parser_errors.append("Syntax error at %s on line number %d." % (p.value, p.lineno))


    def getVars(self):
        return self.matrix_language_vars

#=====================================================================


if __name__ == '__main__':
    """To regenerate the parse tables, invoke parser.py with --regen as the last command-line
        option, for example:
            parser.py somefile.m --regen
    """

    lex_optimize = True
    yacc_optimize = True
    yacc_debug = False
    outputdir = os.path.dirname(sys.argv[0])
    if sys.argv[-1] == '--regen':
        # Forse regeneration of the parse tables
        del sys.argv[-1]
        lex_optimize = False
        yacc_optimize = False
        yacc_debug = True
        # Remove the old parse table
        try:
            os.remove(os.path.join(os.path.abspath(outputdir), 'yacctab.py'))
            os.remove(os.path.join(os.path.abspath(outputdir), 'lextab.py'))
        except:
            pass


    mparser = MParser(lex_optimize=lex_optimize, yacc_optimize=yacc_optimize,
                      yacc_debug=False, taboutputdir=outputdir,
                      printToStderr=False)

    for i in range(1, len(sys.argv)):
        sys.stderr.write("[parse] About to parse %s\n" % sys.argv[i])
        os.system('cat %s' % sys.argv[i])
        theresult = mparser._processFile(sys.argv[i])
        if theresult and len(mparser.lex.errors) == 0:
            sys.stderr.write('[parser] Successfully parsed %s\n' % sys.argv[i])

        print('All variables and their types:')
        for key, val in mparser.getVars().items():
            print("%s : %s" % (key, val))

        if mparser.mlex.errors:
            sys.err.write('***Errors\n', mparser.mlex.errors)
