#
# The pretty printer of the AST
#

import sys
from . import ast
from orio.main.util.globals import *

#-------------------------------------------------

class PrettyPrinter:
    '''The pretty printer of the AST structure'''

    def __init__(self):
        '''To instantiate the pretty printer'''
        pass

    #----------------------------------------------

    def pprint(self, tnode, indent = '  ', extra_indent = '  '):
        '''To generate code that corresponds to the given AST'''

        s = ''

        if isinstance(tnode, ast.NumLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.StringLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.IdentExp):
            s += str(tnode.name)

        elif isinstance(tnode, ast.ArrayRefExp):
            s += self.pprint(tnode.exp, indent, extra_indent)
            s += '[' + self.pprint(tnode.sub_exp, indent, extra_indent) + ']'

        elif isinstance(tnode, ast.FunCallExp):
            s += self.pprint(tnode.exp, indent, extra_indent) + '('
            s += ','.join([self.pprint(x, indent, extra_indent) for x in tnode.args])
            s += ')'

        elif isinstance(tnode, ast.UnaryExp):
            s = self.pprint(tnode.exp, indent, extra_indent)
            if tnode.op_type == tnode.PLUS:
                s = '+' + s
            elif tnode.op_type == tnode.MINUS:
                s = '-' + s
            elif tnode.op_type == tnode.LNOT:
                s = '!' + s
            elif tnode.op_type == tnode.PRE_INC:
                s = ' ++' + s
            elif tnode.op_type == tnode.PRE_DEC:
                s = ' --' + s
            elif tnode.op_type == tnode.POST_INC:
                s = s + '++ '
            elif tnode.op_type == tnode.POST_DEC:
                s = s + '-- '
            else:
                err('orio.module.tilic.pprinter internal error: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, ast.BinOpExp):
            s += self.pprint(tnode.lhs, indent, extra_indent)
            if tnode.op_type == tnode.MUL:
                s += '*'
            elif tnode.op_type == tnode.DIV:
                s += '/'
            elif tnode.op_type == tnode.MOD:
                s += '%'
            elif tnode.op_type == tnode.ADD:
                s += '+'
            elif tnode.op_type == tnode.SUB:
                s += '-'
            elif tnode.op_type == tnode.LT:
                s += '<'
            elif tnode.op_type == tnode.GT:
                s += '>'
            elif tnode.op_type == tnode.LE:
                s += '<='
            elif tnode.op_type == tnode.GE:
                s += '>='
            elif tnode.op_type == tnode.EQ:
                s += '=='
            elif tnode.op_type == tnode.NE:
                s += '!='
            elif tnode.op_type == tnode.LOR:
                s += '||'
            elif tnode.op_type == tnode.LAND:
                s += '&&'
            elif tnode.op_type == tnode.COMMA:
                s += ','
            elif tnode.op_type == tnode.EQ_ASGN:
                s += '='
            else:
                err('orio.module.tilic.pprinter internal error: unknown binary operator type: %s' % tnode.op_type)
            s += self.pprint(tnode.rhs, indent, extra_indent)

        elif isinstance(tnode, ast.ParenthExp):
            s += '(' + self.pprint(tnode.exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ast.ExpStmt):
            s += indent
            if tnode.exp:
                s += self.pprint(tnode.exp, indent, extra_indent)
            s += ';\n'

        elif isinstance(tnode, ast.CompStmt):
            s += indent + '{\n'
            for stmt in tnode.stmts:
                s += self.pprint(stmt, indent + extra_indent, extra_indent)
            s += indent + '}\n'

        elif isinstance(tnode, ast.IfStmt):
            s += indent + 'if (' + self.pprint(tnode.test, indent, extra_indent) + ') '
            if isinstance(tnode.true_stmt, ast.CompStmt):
                tstmt_s = self.pprint(tnode.true_stmt, indent, extra_indent)
                s += tstmt_s[tstmt_s.index('{'):]
                if tnode.false_stmt:
                    s = s[:-1] + ' else '
            else:
                s += '\n'
                s += self.pprint(tnode.true_stmt, indent + extra_indent, extra_indent)
                if tnode.false_stmt:
                    s += indent + 'else '
            if tnode.false_stmt:
                if isinstance(tnode.false_stmt, ast.CompStmt):
                    tstmt_s = self.pprint(tnode.false_stmt, indent, extra_indent)
                    s += tstmt_s[tstmt_s.index('{'):]
                else:
                    s += '\n'
                    s += self.pprint(tnode.false_stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.ForStmt):
            if tnode.start_label:
                s += indent + '/* %s */\n' % tnode.start_label
            s += indent + 'for ('
            if tnode.init:
                s += self.pprint(tnode.init, indent, extra_indent)
            s += '; '
            if tnode.test:
                s += self.pprint(tnode.test, indent, extra_indent)
            s += '; '
            if tnode.iter:
                if (isinstance(tnode.iter, ast.BinOpExp) and tnode.iter.op_type == ast.BinOpExp.EQ_ASGN and 
                    isinstance(tnode.iter.rhs, ast.BinOpExp) and tnode.iter.rhs.op_type == ast.BinOpExp.ADD and
                    isinstance(tnode.iter.rhs.lhs, ast.IdentExp) and isinstance(tnode.iter.lhs, ast.IdentExp) and
                    tnode.iter.lhs.name == tnode.iter.rhs.lhs.name):
                    s += str(tnode.iter.lhs.name) + '+=' + self.pprint(tnode.iter.rhs.rhs, indent, extra_indent)
                else:
                    s += self.pprint(tnode.iter, indent, extra_indent)
            s += ') '
            if isinstance(tnode.stmt, ast.CompStmt): 
                stmt_s = self.pprint(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.pprint(tnode.stmt, indent + extra_indent, extra_indent)
            if tnode.end_label:
                s += indent + '/* %s */\n' % tnode.end_label

        else:
            err('orio.module.tilic.pprinter internal error: unrecognized type of AST: %s' % tnode.__class__.__name__)

        return s


