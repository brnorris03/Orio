#
# The code generator for the SpMV transformation module
#

import sys

#-------------------------------------------

class CodeGen:
    '''The code generator for the SpMV transformation module'''

    def __init__(self, ainfo, annot_body_code, indent):
        '''To instantiate a code generator instance'''
        
        self.ainfo = ainfo
        self.annot_body_code = annot_body_code
        self.indent = indent

    #------------------------------------------------------

    def __generateOuterLoop():

    #------------------------------------------------------

    def generate(self):
        '''To generate the optimized SpMV code'''

        # initialize some variables used during the code generation phase
        ainfo = self.ainfo
        code = ''

        # create the output variables to store temporary results
        out_vars = [ainfo.out_vector + str(i) for i in range(0, ainfo.out_unroll_factor)]

        # create the initialization code for the output variables
        ovar_inits = ''
        for i,ov in enumerate(out_vars):
            if i > 0:
                ovar_inits += ', '
            ovar_inits += '%s=%s' % (ov, ainfo.init_val)
        
        # create the unrolled outer loop
        oloop  = 'for (%s=0; %s<=%s-1; %s+=%s) { \n' % (ainfo.out_loop_var, ainfo.out_loop_var,
                                                        ainfo.num_rows, ainfo.out_loop_var,
                                                        ainfo.out_unroll_factor)
        oloop += '  %s %s; \n' % (ainfo.elm_type, ovar_inits)
        oloop += '} \n'
        if ainfo.out_loop_var > 1:
            oloop += 'for (%s=0; %s<=%s-1; %s+=1) { \n'
            oloop += ''
            oloop += '} \n'

        print '-------end------'
        sys.exit(1)

        # return the code
        return code
    
