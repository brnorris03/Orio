class SubroutineDefinition:
    def __init__(self, header, varrefs, body):
        '''header is a tuple of (name, arglist)'''
        self.name = header[0]
        self.arglist = header[1]
        self.varrefs = varrefs
        self.body = body[0]
        self.bodyspan = body[1] # the location span
        return
    
    def __repr__(self):
        buf = 'subroutine:'
        buf += self.name + '\n' + str(self.arglist) + '\n' + str(self.varrefs) + \
            '\n' + self.body + '\n' + str(self.bodyspan)
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