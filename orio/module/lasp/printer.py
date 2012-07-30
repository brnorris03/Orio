#==============================================================================
# The unparser of the AST nodes
#==============================================================================

import orio.main.util.globals as g
import orio.module.lasp.ast as ast

#------------------------------------------------------------------------------
class Printer(object):
    '''The printer of AST nodes'''

    def __init__(self):
        pass

    #--------------------------------------------------------------------------
    def generate(self, tnode, indent='  ', extra_indent='  '):
        '''To generate code that corresponds to the given AST'''

        s = ''
        if isinstance(tnode, ast.Comment):
            s += indent
            if tnode.text:
                s += '/*' + tnode.text + '*/'
            s += '\n'
        
        elif isinstance(tnode, ast.LitExp):
            if tnode.lit_type == ast.LitExp.STRING:
                s += '"' + str(tnode.val) + '"'
            else:
                s += str(tnode.val)

        elif isinstance(tnode, ast.IdentExp):
            s += str(tnode.name)

        elif isinstance(tnode, ast.ArrayRefExp):
            s += self.generate(tnode.exp, indent, extra_indent)
            s += '[' + self.generate(tnode.sub, indent, extra_indent) + ']'

        elif isinstance(tnode, ast.CallExp):
            s += self.generate(tnode.exp, indent, extra_indent) + '('
            s += ','.join(map(lambda x: self.generate(x, indent, extra_indent), tnode.args))
            s += ')'

        elif isinstance(tnode, ast.UnaryExp):
            s = self.generate(tnode.exp, indent, extra_indent)
            if   tnode.op_type == tnode.PLUS: s = '+' + s
            elif tnode.op_type == tnode.MINUS: s = '-' + s
            elif tnode.op_type == tnode.LNOT: s = '!' + s
            elif tnode.op_type == tnode.TRANSPOSE: s += "'"
            elif tnode.op_type == tnode.PRE_INC: s = ' ++' + s
            elif tnode.op_type == tnode.PRE_DEC: s = ' --' + s
            elif tnode.op_type == tnode.POST_INC: s += '++ '
            elif tnode.op_type == tnode.POST_DEC: s += '-- '
            else: g.err('orio.module.lasp.codegen: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, ast.BinOpExp):
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

        elif isinstance(tnode, ast.ParenExp):
            s += '(' + self.generate(tnode.exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ast.ExpStmt):
            s += indent + self.generate(tnode.exp, indent, extra_indent) + ';\n'

        elif isinstance(tnode, ast.CompStmt):
            s += indent + '{\n'
            for stmt in tnode.stmts:
                s += self.generate(stmt, indent + extra_indent, extra_indent)
            s += indent + '}\n'

        elif isinstance(tnode, ast.IfStmt):
            s += indent + 'if (' + self.generate(tnode.test, indent, extra_indent) + ') '
            if isinstance(tnode.true_stmt, ast.CompStmt):
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
                if isinstance(tnode.false_stmt, ast.CompStmt):
                    tstmt_s = self.generate(tnode.false_stmt, indent, extra_indent)
                    s += tstmt_s[tstmt_s.index('{'):]
                else:
                    s += '\n'
                    s += self.generate(tnode.false_stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.ForStmt):
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
            if isinstance(tnode.stmt, ast.CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.WhileStmt):
            s += indent + 'while (' + self.generate(tnode.test, indent, extra_indent)
            s += ') '
            if isinstance(tnode.stmt, ast.CompStmt): 
                stmt_s = self.generate(tnode.stmt, indent, extra_indent)
                s += stmt_s[stmt_s.index('{'):]
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)

        elif isinstance(tnode, ast.VarInit):
            s += self.generate(tnode.var_name, indent, extra_indent)
            if tnode.init_exp:
                s += '=' + self.generate(tnode.init_exp, indent, extra_indent)

        elif isinstance(tnode, ast.VarDec):
            s += indent + str(tnode.type_name) + ' '
            s += ', '.join(map(self.generate, tnode.var_inits))
            s += ';\n'

        elif isinstance(tnode, ast.ParamDec):
            s += self.generate(tnode.ty, indent, extra_indent) + ' ' + self.generate(tnode.name, indent, extra_indent)

        elif isinstance(tnode, ast.FunDec):
            s += indent + ' '.join(tnode.kids[2]) + ' '
            s += self.generate(tnode.kids[1], indent, extra_indent) + ' '
            s += self.generate(tnode.kids[0], indent, extra_indent) + '('
            s += ', '.join(map(self.generate, tnode.kids[3])) + ') '
            s += self.generate(tnode.kids[4], indent, extra_indent)

        elif isinstance(tnode, ast.TransformStmt):
            g.err('orio.module.lasp.codegen: a transformation statement is never generated as an output')

        else:
            g.err('orio.module.lasp.codegen: unrecognized type of AST: (%s, %s)' % (tnode.__class__.__name__,tnode))

        return s
    #--------------------------------------------------------------------------
