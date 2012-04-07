#
# The code generator (i.e. unparser) for the AST classes for CUDA
#

from ast import *
import orio.main.util.globals as g

class CodeGen:
    '''The code generator for AST classes'''

    def __init__(self):
        pass

    #----------------------------------------------
    def generate(self, tnode, indent='  ', extra_indent='  '):
        '''To generate code that corresponds to the given AST'''

        s = ''
        if isinstance(tnode, Comment):
            s += indent
            if tnode.text:
                s += '/*' + tnode.text + '*/'
            s += '\n'
        
        elif isinstance(tnode, LitExp):
            if tnode.lit_type == LitExp.STRING:
                s += '"' + str(tnode.val) + '"'
            else:
                s += str(tnode.val)

        elif isinstance(tnode, IdentExp):
            s += str(tnode.name)

        elif isinstance(tnode, ArrayRefExp):
            s += self.generate(tnode.exp, indent, extra_indent)
            s += '[' + self.generate(tnode.sub, indent, extra_indent) + ']'

        elif isinstance(tnode, CallExp):
            s += self.generate(tnode.exp, indent, extra_indent) + '('
            s += ','.join(map(lambda x: self.generate(x, indent, extra_indent), tnode.args))
            s += ')'

        elif isinstance(tnode, UnaryExp):
            s = self.generate(tnode.exp, indent, extra_indent)
            if   tnode.op_type == tnode.PLUS: s = '+' + s
            elif tnode.op_type == tnode.MINUS: s = '-' + s
            elif tnode.op_type == tnode.LNOT: s = '!' + s
            elif tnode.op_type == tnode.PRE_INC: s = ' ++' + s
            elif tnode.op_type == tnode.PRE_DEC: s = ' --' + s
            elif tnode.op_type == tnode.POST_INC: s += '++ '
            elif tnode.op_type == tnode.POST_DEC: s += '-- '
            else: g.err('orio.module.lasp.codegen: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, BinOpExp):
            s += self.generate(tnode.lhs, indent, extra_indent)
            if   tnode.op_type == tnode.PLUS: s += '+'
            elif tnode.op_type == tnode.MINUS: s += '-'
            elif tnode.op_type == tnode.MULT: s += '*'
            elif tnode.op_type == tnode.DIV: s += '/'
            elif tnode.op_type == tnode.MOD: s += '%'
            elif tnode.op_type == tnode.LT: s += '<'
            elif tnode.op_type == tnode.GT: s += '>'
            elif tnode.op_type == tnode.LE: s += '<='
            elif tnode.op_type == tnode.GE: s += '>='
            elif tnode.op_type == tnode.EE: s += '=='
            elif tnode.op_type == tnode.NE: s += '!='
            elif tnode.op_type == tnode.LOR: s += '||'
            elif tnode.op_type == tnode.LAND: s += '&&'
            elif tnode.op_type == tnode.EQ: s += '='
            elif tnode.op_type == tnode.EQPLUS: s += '+='
            elif tnode.op_type == tnode.EQMINUS: s += '-='
            elif tnode.op_type == tnode.EQMULT: s += '*='
            elif tnode.op_type == tnode.EQDIV: s += '/='
            else: g.err('orio.module.loop.codegen_cuda internal error: unknown binary operator type: %s' % tnode.op_type)
            s += self.generate(tnode.rhs, indent, extra_indent)

        elif isinstance(tnode, ParenExp):
            s += '(' + self.generate(tnode.exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ExpStmt):
            s += indent + self.generate(tnode.exp, indent, extra_indent) + ';\n'

        elif isinstance(tnode, CompStmt):
            s += indent + '{\n'
            for stmt in tnode.stmts:
                s += self.generate(stmt, indent + extra_indent, extra_indent)
            s += indent + '}\n'

        elif isinstance(tnode, IfStmt):
            s += indent + 'if (' + self.generate(tnode.test, indent, extra_indent) + ') '
            if isinstance(tnode.true_stmt, CompStmt):
                tstmt_s = self.generate(tnode.true_stmt, indent, extra_indent)
                s += tstmt_s[tstmt_s.index('{'):]
                if tnode.false_stmt:
                    s = s[:-1] + ' else '
            else:
                s += '\n'
                s += self.generate(tnode.true_stmt, indent + extra_indent, extra_indent)
                if tnode.false_stmt:
                    s += indent + 'else '
            if tnode.false_stmt:
                if isinstance(tnode.false_stmt, CompStmt):
                    tstmt_s = self.generate(tnode.false_stmt, indent, extra_indent)
                    s += tstmt_s[tstmt_s.index('{'):]
                else:
                    s += '\n'
                    s += self.generate(tnode.false_stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ForStmt):
            if tnode.getLabel(): s += tnode.getLabel() + ':'
            s += indent + 'for ('
            if tnode.init:
                s += self.generate(tnode.init, indent, extra_indent)
            s += '; '
            if tnode.test:
                s += self.generate(tnode.test, indent, extra_indent)
            s += '; '
            if tnode.iter:
                s += self.generate(tnode.iter, indent, extra_indent)
            s += ') '
            if isinstance(tnode.stmt, CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, WhileStmt):
            s += indent + 'while (' + self.generate(tnode.test, indent, extra_indent)
            s += ') '
            if isinstance(tnode.stmt, CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, VarInit):
            s += self.generate(tnode.var_name, indent, extra_indent)
            if tnode.init_exp:
                s += '=' + self.generate(tnode.init_exp, indent, extra_indent)

        elif isinstance(tnode, VarDec):
            s += indent + str(tnode.type_name) + ' '
            s += ', '.join(map(self.generate, tnode.var_inits))
            s += ';\n'

        elif isinstance(tnode, ParamDec):
            s += tnode.ty + ' ' + tnode.name

        elif isinstance(tnode, FunDec):
            s += indent + ' '.join(tnode.modifiers) + ' '
            s += tnode.return_type + ' '
            s += tnode.name + '('
            s += ', '.join(map(self.generate, tnode.params)) + ') '
            s += self.generate(tnode.body, indent, extra_indent)

        elif isinstance(tnode, TransformStmt):
            g.err('orio.module.lasp.codegen: a transformation statement is never generated as an output')

        else:
            g.err('orio.module.lasp.codegen: unrecognized type of AST: %s' % tnode.__class__.__name__)

        return s

