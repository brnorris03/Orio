#==============================================================================
# The unparser of the AST nodes
#==============================================================================

import orio.main.util.globals as g
from orio.module.splingo.ast import *

#------------------------------------------------------------------------------
class Printer(object):
    '''The printer of AST nodes'''

    def __init__(self, indent='  '):
        self.extra_indent = indent

    #--------------------------------------------------------------------------
    def pp(self, n, indent='  '):
        '''Pretty-print the given AST'''

        s = ''
        if isinstance(n, Comment):
            s += indent
            if n.comment:
                s += '/*' + n.comment + '*/'
            s += '\n'
        
        elif isinstance(n, LitExp):
            if n.lit_type == LitExp.STRING:
                s += '"' + str(n.val) + '"'
            else:
                s += str(n.val)

        elif isinstance(n, IdentExp):
            s += str(n.name)

        elif isinstance(n, ArrayRefExp):
            s += self.pp(n.exp, indent)
            s += '[' + self.pp(n.sub, indent) + ']'

        elif isinstance(n, CallExp):
            s += self.pp(n.exp, indent) + '('
            s += ','.join([self.pp(x, indent) for x in n.args])
            s += ')'

        elif isinstance(n, UnaryExp):
            s = self.pp(n.exp, indent)
            if   n.oper == n.PLUS:      s = '+' + s
            elif n.oper == n.MINUS:     s = '-' + s
            elif n.oper == n.LNOT:      s = '!' + s
            elif n.oper == n.TRANSPOSE: s += "'"
            elif n.oper == n.PRE_INC:   s = ' ++' + s
            elif n.oper == n.PRE_DEC:   s = ' --' + s
            elif n.oper == n.POST_INC:  s += '++ '
            elif n.oper == n.POST_DEC:  s += '-- '
            else: g.err('%s: unknown unary operator type: %s' % (self.__class__, n.oper))

        elif isinstance(n, BinOpExp):
            s += self.pp(n.lhs, indent)
            if   n.oper == n.PLUS:    s += '+'
            elif n.oper == n.MINUS:   s += '-'
            elif n.oper == n.MULT:    s += '*'
            elif n.oper == n.DIV:     s += '/'
            elif n.oper == n.MOD:     s += '%'
            elif n.oper == n.LT:      s += '<'
            elif n.oper == n.GT:      s += '>'
            elif n.oper == n.LE:      s += '<='
            elif n.oper == n.GE:      s += '>='
            elif n.oper == n.EE:      s += '=='
            elif n.oper == n.NE:      s += '!='
            elif n.oper == n.LOR:     s += '||'
            elif n.oper == n.LAND:    s += '&&'
            elif n.oper == n.EQ:      s += '='
            elif n.oper == n.EQPLUS:  s += '+='
            elif n.oper == n.EQMINUS: s += '-='
            elif n.oper == n.EQMULT:  s += '*='
            elif n.oper == n.EQDIV:   s += '/='
            elif n.oper == n.COMMA:   s += ','
            else: g.err('%s: unknown binary operator type: %s' % (self.__class__, n.oper))
            s += self.pp(n.rhs, indent)

        elif isinstance(n, ParenExp):
            s += '(' + self.pp(n.exp, indent) + ')'

        elif isinstance(n, ExpStmt):
            s += indent + self.pp(n.exp, indent) + ';\n'

        elif isinstance(n, CompStmt):
            s += indent + '{\n'
            for stmt in n.stmts:
                s += self.pp(stmt, indent + self.extra_indent)
            s += indent + '}\n'

        elif isinstance(n, IfStmt):
            s += indent + 'if (' + self.pp(n.test, indent) + ') '
            if isinstance(n.then_s, CompStmt):
                tstmt_s = self.pp(n.then_s, indent)
                s += tstmt_s[tstmt_s.index('{'):]
                if n.else_s:
                    s = s[:-1] + ' else '
            else:
                s += '\n'
                s += self.pp(n.then_s, indent + self.extra_indent)
                if n.else_s:
                    s += indent + 'else '
            if n.else_s:
                if isinstance(n.else_s, CompStmt):
                    tstmt_s = self.pp(n.else_s, indent)
                    s += tstmt_s[tstmt_s.index('{'):]
                else:
                    s += '\n'
                    s += self.pp(n.else_s, indent + self.extra_indent)

        elif isinstance(n, ForStmt):
            #if n.getLabel(): s += n.getLabel() + ':'
            s += indent + 'for ('
            if n.init:
                s += self.pp(n.init, indent)
            s += '; '
            if n.test:
                s += self.pp(n.test, indent)
            s += '; '
            if n.itr:
                s += self.pp(n.itr, indent)
            s += ') '
            if isinstance(n.stmt, CompStmt): 
                stmt_s = self.pp(n.stmt, indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.pp(n.stmt, indent + self.extra_indent)

        elif isinstance(n, WhileStmt):
            s += indent + 'while (' + self.pp(n.test, indent)
            s += ') '
            if isinstance(n.stmt, CompStmt): 
                stmt_s = self.pp(n.stmt, indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.pp(n.stmt, indent + self.extra_indent)

        elif isinstance(n, VarInit):
            s += self.pp(n.var_name, indent)
            if n.init_exp:
                s += '=' + self.pp(n.init_exp, indent)

        elif isinstance(n, VarDec):
            if len(n.quals) > 0:
              s += ''.join(n.quals) + ' '
            s += str(n.type_name) + ' '
            s += ', '.join(map(self.pp, n.var_inits))
            if n.isAtomic:
                s = indent + s + ';\n'

        elif isinstance(n, ParamDec):
            s += self.pp(n.ty, indent) + ' ' + self.pp(n.name, indent)

        elif isinstance(n, FunDec):
            s += ' '.join(n.quals) + ''
            s += self.pp(n.rtype, indent) + ' '
            s += self.pp(n.name, indent) + '('
            s += ', '.join(map(self.pp, n.params)) + ') '
            s += self.pp(n.body, indent)

        elif isinstance(n, TransformStmt):
            g.err('%s: a transformation statement is never generated as an output' % self.__class__)

        else:
            g.err('%s: unrecognized type of AST: (%s, %s)' % (self.__class__, n.__class__.__name__,n))

        return s
    #--------------------------------------------------------------------------
