#
# Contain unparsing procedures.
#

import sys
from orio.main.util.globals import *

#-------------------------------------------

class CodeGen:
    '''The code generator for the Blue Gene's memory alignment optimizer'''

    def __init__(self, vars, annot_body_code, indent, language='c'):
        '''To instantiate a code generator instance'''

        self.generator = None
        
        if not language.strip(): language='c'
        else: language = language.strip().lower()

        if language in ['c','c++','cxx']:
            self.generator = CodeGen_C(vars, annot_body_code, indent)
        elif language in ['f', 'f90', 'fortran']:
            self.generator = CodeGen_F(vars, annot_body_code, indent) # TODO
        else:
            err('orio.module.align.codegen: Unknown language specified for code generation: %s' % language)

        pass
        
    #------------------------------------------------------

    def generate(self):

        # return the generated code
        return self.generator.generate()
    
    
class CodeGen_C(CodeGen):
    '''The code generator for the Blue Gene's memory alignment optimizer'''

    def __init__(self, vars, annot_body_code, indent):
        '''To instantiate a code generator instance'''
        self.vars = vars
        self.annot_body_code = annot_body_code
        self.indent = indent
        self.language = 'c'
        pass

    #------------------------------------------------------

    def __printAddress(self, var):
        '''To Return the starting address location of the given variable (in C/C++)'''

        vname = var.vname
        dims = var.dims[:]
        dims.remove(None)
        s = str(vname)
        if len(dims) > 0:
            s += '[' + ']['.join(map(str, dims)) + ']'
        return s
        
    #------------------------------------------------------

    def generate(self):
        '''To generate the memory-alignment checking code'''

        # initialize the original indentation and the extra indentation
        indent = self.indent
        extra_indent = '  ' 

        # generate the disjoint pragma
        s = '\n'
        s += indent + '#pragma disjoint ('
        s += ','.join(['*' + self.__printAddress(v) for v in self.vars])
        s += ') \n'

        # generate the alignment test
        s += indent + 'if ((('
        s += '|'.join(['(int)(' + self.__printAddress(v) + ')' for v in self.vars])
        s += ') & 0xF) == 0) {\n'

        # generate a sequence of alignment intrinsics
        for v in self.vars:
            s += indent + extra_indent + '__alignx(16,' + self.__printAddress(v) + '); \n'

        # append the annotation body code
        s += self.annot_body_code.replace('\n', '\n' + extra_indent) + '\n'

        # generate the unaligned version
        s += indent + '} else {\n'
        s += self.annot_body_code.replace('\n', '\n' + extra_indent) + '\n'
        s += indent + '} \n'
        s += indent

        # return the generated code
        return s
    

class CodeGen_F(CodeGen):
    '''The code generator for the Blue Gene's memory alignment optimizer'''

    def __init__(self, vars, annot_body_code, indent):
        '''To instantiate a code generator instance'''
        self.vars = vars
        self.annot_body_code = annot_body_code
        self.indent = indent
        self.language = 'f'
        pass

    #------------------------------------------------------

    def generate(self):
        '''To generate the memory-alignment checking code'''

        s = ''
        raise NotImplementedError('%s: Fortran code generation not implemented yet for align module')
        # return the generated code
        return s
    

