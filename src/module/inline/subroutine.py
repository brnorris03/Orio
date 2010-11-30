class SubroutineDeclaration:
    def __init__(self, header, varrefs, body, function=False):
        '''
        @param header: a tuple of (name, arglist)
        @param body: a tuple of (subroutine statements string, span in source file); 
                The span is a (startpos, endpos) tuple.
                
        '''
        self.name = header[0]
        self.arglist = header[1]
        self.varrefs = varrefs
        self.body = body[0]
        self.bodyspan = body[1]         # the location span
        self.function = function        # designates whether subroutine is function or procedure
        return
    
    def __repr__(self):
        buf = 'subroutine:'
        buf += str(self.name) + '\n' + str(self.arglist) + '\n' + str(self.varrefs) + \
            '\n' + str(self.body) + '\n' + str(self.bodyspan)
        return buf
    
    def inline(self, params):
        '''
        Rewrite the body to contain actual arguments in place of the formal parameters.
        @param params: the list of actual parameters in the same order as the formal 
                    parameters in the subroutine definition
        '''
        starpos = self.bodyspan[0]
        for v in self.varrefs:
            arg = v[0]      # the variable name
            argspan = v[1]  # begin and end position 
            
        return
        
    
    # end of class SubroutineDefinition
    
class SubroutineDefinition(SubroutineDeclaration):
    pass