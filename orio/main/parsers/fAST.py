#
# The Fortran-specific classes for the abstract syntax tree (ASTNode)

import os
from orio.main.parsers.AST import *

class MainProgram(ASTNode):

    def __init__(self, line_no='', spec_part=None, internal_part=None):
        '''Create a orio.main.program node'''
        ASTNode.__init__(self,line_no)
        self.specificationPart = spec_part
        self.internalSubprogramPart = internal_part

    specificationPart = property(getSpecificationPart, setSpecificationPart, delSpecificationPart, "The specification part of the program")
    internalSubprogramPart = property(getInternalSubprogramPart, setInternalSubprogramPart, delInternalSubprogramPart, "The internal subprogram part of the program")

    def getSpecificationPart(self):
        return self.__specificationPart

    def getInternalSubprogramPart(self):
        return self.__internalSubprogramPart

    def setSpecificationPart(self, value):
        self.__specificationPart = value

    def setInternalSubprogramPart(self, value):
        self.__internalSubprogramPart = value

    def delSpecificationPart(self):
        del self.__specificationPart

    def delInternalSubprogramPart(self):
        del self.__internalSubprogramPart



