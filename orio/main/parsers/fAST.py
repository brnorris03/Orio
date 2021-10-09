#
# The Fortran-specific classes for the abstract syntax tree (ASTNode)

from orio.main.parsers.ast import *

class MainProgram(ASTNode):

    def __init__(self, line_no='', spec_part=None, internal_part=None):
        '''Create a orio.main.program node'''
        ASTNode.__init__(self,line_no)
        self.specificationPart = spec_part
        self.internalSubprogramPart = internal_part

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

    specificationPart = property(getSpecificationPart, setSpecificationPart, delSpecificationPart, "The specification part of the program")
    internalSubprogramPart = property(getInternalSubprogramPart, setInternalSubprogramPart, delInternalSubprogramPart, "The internal subprogram part of the program")




class NodeVisitor(object):
    """ A base NodeVisitor class for visiting fAST nodes.
        Subclass it and define your own visit_XXX methods, where
        XXX is the class name you want to visit with these
        methods.

        For example:

        class ConstantVisitor(NodeVisitor):
            def __init__(self):
                self.values = []

            def visit_Constant(self, node):
                self.values.append(node.value)

        Creates a list of values of all the constant nodes
        encountered below the given node. To use it:

        cv = ConstantVisitor()
        cv.visit(node)

        Notes:

        *   generic_visit() will be called for AST nodes for which
            no visit_XXX method was defined.
        *   The children of nodes for which a visit_XXX was
            defined will not be visited - if you need this, call
            generic_visit() on the node.
            You can use:
                NodeVisitor.generic_visit(self, node)
        *   Modeled after Python's own AST visiting facilities
            (the ast module of Python 3.0)
    """

    def visit(self, node):
        """ Visit a node.
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        for c in node.children():
            self.visit(c)

