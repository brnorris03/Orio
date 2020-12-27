#
# The code generator (i.e. unparser) for the AST classes
#

import sys
import orio.main.parsers.fAST as ast
from orio.main.util.globals import *

# ==============================================================================================

class CodeGen:
    '''The code generator for the AST classes'''

    def __init__(self):
        '''To instantiate a code generator'''
        self.ftypes = {'int':'integer', 
                       'long': 'integer*4', 
                       'float': 'real', 
                       'double': 'double precision'}
        pass

    #----------------------------------------------

    def generate(self, tnode, indent = '  ', extra_indent = '  '):
        '''To generate code that corresponds to the given AST'''

        s = ''

        if isinstance(tnode, ast.NumLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.StringLitExp):
            s += str(tnode.val)

        elif isinstance(tnode, ast.IdentExp):
            s += str(tnode.name)

        elif isinstance(tnode, ast.ArrayRefExp):
            s += self.generate(tnode.exp, indent, extra_indent)
            s += '(' + self.generate(tnode.sub_exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ast.FunCallExp):
            s += self.generate(tnode.exp, indent, extra_indent) + '('
            s += ','.join([self.generate(x, indent, extra_indent) for x in tnode.args])
            s += ')'

        elif isinstance(tnode, ast.UnaryExp):
            s = self.generate(tnode.exp, indent, extra_indent)
            if tnode.op_type == tnode.PLUS:
                s = '+' + s
            elif tnode.op_type == tnode.MINUS:
                
                s = '-' + s
            elif tnode.op_type == tnode.LNOT:
                s = 'NOT(' + s + ')'
            elif tnode.op_type == tnode.PRE_INC:
                s += '\n' + indent + s + ' = ' + s + ' + 1\n'
            elif tnode.op_type == tnode.PRE_DEC:
                s += '\n' + indent + s + ' = ' + s + ' - 1\n'
            elif tnode.op_type == tnode.POST_INC:
                s += s + '\n' + indent + s + ' = ' + s + ' + 1\n'
            elif tnode.op_type == tnode.POST_DEC:
                s += s + '\n' + indent + s + ' = ' + s + ' - 1\n'
            else:
                err('orio.module.loop.codegen internal error: unknown unary operator type: %s' % tnode.op_type)

        elif isinstance(tnode, ast.BinOpExp):
            if tnode.op_type not in [tnode.MOD, tnode.COMMA]:
                s += self.generate(tnode.lhs, indent, extra_indent)
                if tnode.op_type == tnode.MUL:
                    s += '*'
                elif tnode.op_type == tnode.DIV:
                    s += '/'
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
                    s += '.OR.'
                elif tnode.op_type == tnode.LAND:
                    s += '.AND.'
                elif tnode.op_type == tnode.EQ_ASGN:
                    s += '='
                else:
                    err('orio.module.loop.codegen internal error: unknown binary operator type: %s' % tnode.op_type)
                    
                s += self.generate(tnode.rhs, indent, extra_indent)
                
            else:
                
                if tnode.op_type == tnode.MOD:
                    s += 'MOD(' + self.generate(tnode.lhs, indent, extra_indent) + ', ' \
                        + self.generate(tnode.rhs, indent, extra_indent) + ')'
                elif tnode.op_type == tnode.COMMA:
                    # TODO: We need to implement an AST canonicalization step for Fortran before generating the code.
                    print('internal warning: Fortran code generator does not fully support the comma operator -- the generated code may not compile.')
                    s += self.generate(tnode.rhs, indent, extra_indent) 
                    s += '\n' + indent + self.generate(tnode.lhs, indent, extra_indent)
                    s +='\n! ORIO Warining: check code above and fix problems.'

        elif isinstance(tnode, ast.ParenthExp):
            s += '(' + self.generate(tnode.exp, indent, extra_indent) + ')'

        elif isinstance(tnode, ast.ExpStmt):
            s += indent
            if tnode.exp:
                s += self.generate(tnode.exp, indent, extra_indent)
            s += '\n'

        elif isinstance(tnode, ast.CompStmt):
            s += indent + '\n'
            for stmt in tnode.stmts:
                s += self.generate(stmt, indent + extra_indent, extra_indent)
            s += indent + '\n'

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
            s += indent + 'do ' 
            if tnode.init:
                s += self.generate(tnode.init, indent, extra_indent)
            if not tnode.test:
                err('orio.module.loop.codegen:  missing loop test expression. Fortran code generation requires a loop test expression.')
                
            if not tnode.iter:
                err('orio.module.loop.codegen:  missing loop increment expression. Fortran code generation requires a loop increment expression.')
            s += ', '
            if not isinstance(tnode.test, ast.BinOpExp):
                err('orio.module.loop.codegen internal error: cannot handle code generation for loop test expression')
                
            if tnode.test.op_type not in [tnode.test.LE, tnode.test.LT, tnode.test.GE, tnode.test.GT]: 
                err('orio.module.loop.codegen internal error: cannot generate Fortran loop, only <, >, <=, >= are recognized in the loop limit test')
            
            # Generate the loop bound        
            s += self.generate(tnode.test.rhs, indent, extra_indent)
            
            # Check whether we need to change the bound
            if tnode.test.op_type == tnode.test.LE: s += ' + 1'
            if tnode.test.op_type == tnode.test.GE: s += ' - 1'
            
            s += ', '
            # Generate the loop increment/decrement step
            
            if not isinstance(tnode.iter, (ast.BinOpExp, ast.UnaryExp)):
                err('orio.module.loop.codegen internal error: cannot handle code generation for loop increment expression')
 
            unary = False
            if isinstance(tnode.iter, ast.UnaryExp):
                incr_decr = [tnode.iter.POST_DEC, tnode.iter.op_type.PRE_DEC, tnode.iter.POST_INC, tnode.iter.PRE_INC]
                unary = True
                
            if not ((isinstance(tnode.iter, ast.BinOpExp) and tnode.iter.op_type == tnode.iter.EQ_ASGN)
                    or (isinstance(tnode.iter, ast.UnaryExp) and tnode.iter.op_type in incr_decr)): 
                err('orio.module.loop.codegen internal error: cannot handle code generation for loop increment expression')

            if tnode.test.op_type in [tnode.test.GT, tnode.test.GE] \
                and unary and tnode.iter.op_type in [tnode.iter.PRE_DEC, tnode.iter.POST_DEC]:
                s += '-'
               
            if unary and tnode.iter.op_type in incr_decr:   # ++i
                s += '1'
            else:
                s += self.generate(tnode.iter.rhs, indent, extra_indent)
            
            if isinstance(tnode.stmt, ast.CompStmt): 
                s += self.generate(tnode.stmt, indent, extra_indent)
                s += '\n' + indent + 'end do\n'
            else:
                s += '\n'
                s += self.generate(tnode.stmt, indent + extra_indent, extra_indent)
                s += '\n' + indent + 'end do\n'

        elif isinstance(tnode, ast.TransformStmt):
            err('orio.module.loop.codegen internal error: a transformation statement is never generated as an output')

        elif isinstance(tnode, ast.VarDecl):
            
            if tnode.type_name not in list(self.ftypes.keys()):
                err('orio.module.loop.codegen internal error: Cannot generate Fortran type for ' + tnode.type_name)
                
            s += indent + str(self.ftypes[tnode.type_name]) + ' '
            s += ', '.join(tnode.var_names)
            s += '\n'

        elif isinstance(tnode, ast.Pragma):
            s += '#pragma ' + str(tnode.pstring) + '\n'

        elif isinstance(tnode, ast.Container):
            s += self.generate(tnode.ast, indent, extra_indent)

        else:
            err('orio.module.loop.codegen internal error: unrecognized type of AST: %s' % tnode.__class__.__name__)

        return s




