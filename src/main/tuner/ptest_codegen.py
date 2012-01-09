#
# The basic code generator for performance-testing code
#

import random, re, sys
import skeleton_code
from orio.main.util.globals import *
from main.tuner.skeleton_code import SEQ_TIMER

#-----------------------------------------------------

class PerfTestCodeGen:
    '''The code generator used to produce a performance-testing code'''

    # function names
    malloc_func_name = 'malloc_arrays'
    init_func_name = 'init_input_vars'

    #-----------------------------------------------------

    def __init__(self, input_params, input_decls, decl_file, init_file, skeleton_code_file, language='c',
                 random_seed=None, use_parallel_search=False):
        '''To instantiate the testing code generator'''
        
        self.input_params = input_params
        self.input_decls = input_decls
        self.decl_file = decl_file
        self.init_file = init_file
        self.skeleton_code_file = skeleton_code_file
        self.use_parallel_search = use_parallel_search

        self.iparam_code = self.__genIParams(input_params)
        self.decl_code = self.__genDecls(input_decls)
        self.malloc_code = self.__genMAllocs(input_decls)
        self.init_code = self.__genInits(input_decls)

        self.__checkDeclFile()
        self.__checkInitFile()
        scode = self.__checkSkeletonCodeFile()
        self.ptest_skeleton_code = skeleton_code.PerfTestSkeletonCode(scode, use_parallel_search)
        
    #-----------------------------------------------------

    def __genIParams(self, input_params):
        '''
        Generate declaration code for:
         - input parameters (using preprocessor definitions)
        '''
        
        # generate definitions for each input parameter
        iparams = ['#define %s %s' % (pname, rhs) for pname, rhs in input_params]
        
        # generate and return the input parameter code
        iparam_code = '\n'.join(iparams)
        return iparam_code

    #-----------------------------------------------------

    def __genDecls(self, input_decls):
        '''
        Generate declaration code for:
         - declarations for the input variables
        '''

        # generate the input variable declarations
        decls = []
        for is_static, vtype, vname, vdims, rhs in input_decls:

            #TODO: handle structs (look for period in vname)
            if len(vdims) == 0:
                decls.append('%s %s;' % (vtype, vname))
            else:
                if is_static:
                    dim_code = '[%s]' % ']['.join(vdims)
                    decls.append('%s %s%s;' % (vtype, vname, dim_code))
                else:
                    ptr_code = '*' * len(vdims)
                    decls.append('%s %s%s;' % (vtype, ptr_code, vname))

        # generate and return the declaration code
        decl_code = '\n'.join(decls)
        return decl_code

    #-----------------------------------------------------
    
    def __genMAllocs(self, input_decls):
        '''
        Generate declaration code for:
         - memory allocations for input arrays (that are dynamic arrays)
        '''

        # generate iteration variables
        max_dim = 0
        for _,_,_,vdims,_ in input_decls:
            max_dim = max(max_dim, len(vdims))
        iter_vars = map(lambda x: 'i%s' % x, range(1, max_dim+1))

        # generate code for the declaration of the iteration variables
        if len(iter_vars) == 0:
            ivars_decl_code = ''
        else:
            ivars_decl_code = 'int %s;' % ','.join(iter_vars)
        
        # generate memory allocations for dynamic input arrays
        mallocs = []
        for is_static, vtype, vname, vdims, rhs in input_decls:                 
            
            if len(vdims) > 0 and not is_static:
                for i in range(0, len(vdims)):
                    loop_code = ''
                    if i > 0:
                        ivar = iter_vars[i-1]
                        dim = vdims[i-1]
                        loop_code += (' ' * (i-1)) + \
                            'for (%s=0; %s<%s; %s++) {\n' % (ivar, ivar, dim, ivar)
                    dim_code = ''
                    if i > 0:
                        dim_code = '[%s]' % ']['.join(iter_vars[:i])
                    rhs_code = ('(%s%s) malloc((%s) * sizeof(%s%s))' %
                                (vtype, '*' * (len(vdims) - i),
                                 vdims[i], vtype, '*' * (len(vdims) - i - 1)))
                    loop_body_code = (' ' * i) + '%s%s = %s;' % (vname, dim_code, rhs_code)
                    code = loop_code + loop_body_code
                    if code:
                        mallocs.append(code)
                brace_code = '}' * (len(vdims) - 1)
                if brace_code:
                    mallocs.append(brace_code)
        
        # return an empty code if no dynamic memory allocation is needed
        if len(mallocs) == 0:
            return ''

        # generate and return the declaration code
        malloc_code = '\n'.join([ivars_decl_code] + mallocs)
        malloc_code = '  ' + re.sub('\n', '\n  ', malloc_code)
        return malloc_code

    #-----------------------------------------------------
    
    def __genInits(self, input_decls):
        '''
        Generate code for:
         - value initializations for all input variables
        '''

        # generate iteration variables
        max_dim = 0
        for _,_,_,vdims,_ in input_decls:
            max_dim = max(max_dim, len(vdims))
        iter_vars = map(lambda x: 'i%s' % x, range(1, max_dim+1))

        # generate code for the declaration of the iteration variables
        if len(iter_vars) == 0:
            ivars_decl_code = ''
        else:
            ivars_decl_code = 'int %s;' % ','.join(iter_vars)
        
        # generate array value initializations
        inits = []
        for _, vtype, vname, vdims, rhs in input_decls:

            # skip if it does not have an initial value (i.e. RHS == None)
            if rhs == None:
                continue
        

            # if it is a scalar
            if len(vdims) == 0:
                if rhs == 'random':
                    rhs = '(%s) %s' % (vtype, random.uniform(1, 10))
                inits.append('%s = %s;' % (vname, rhs))
                continue

            # generate array value initialization code
            rank = len(vdims)
            used_iter_vars = iter_vars[:rank]
            loop_code = ''
            for i, (ivar, dim) in enumerate(zip(used_iter_vars, vdims)):
                loop_code += (' ' * i) + 'for (%s=0; %s<%s; %s++)\n' % (ivar, ivar, dim, ivar)
            dim_code = '[%s]' % ']['.join(used_iter_vars)
            if rhs == 'random':
                sum = '(%s)' % '+'.join(used_iter_vars)
                rhs = '%s %% %s + %s' % (sum, 5, 1)
            loop_body_code = (' ' * rank) + '%s%s = %s;' % (vname, dim_code, rhs)
            inits.append(loop_code + loop_body_code)

        # return an empty code if no initialization is needed
        if len(inits) == 0:
            return ''

        # generate and return the initialization code
        init_code = '\n'.join([ivars_decl_code] + inits)
        init_code = '  ' + re.sub('\n', '\n  ', init_code)
        return init_code
    
    #-----------------------------------------------------

    def __checkDeclFile(self):
        '''To check the declaration file'''

        # do not perform checking if no declaration file is specified
        if not self.decl_file:
            return

        # check if the file can be opened
        try:
            f = open(self.decl_file)
            f.close()
        except:
            err('orio.main.tuner.ptest_codegen:  cannot read file: "%s"' % self.decl_file)

    #-----------------------------------------------------

    def __checkInitFile(self):
        '''To check the initialization file'''

        # do not perform checking if no initialization file is specified
        if not self.init_file:
            return

        # read the content of the file
        try:
            f = open(self.init_file)
            init_code = f.read()
            f.close()
        except:
            err('orio.main.tuner.ptest_codegen:  cannot read file: "%s"' % self.init_file)

        # check if the file contains the initialization function
        init_func_re = r'void\s+%s\(\s*\)\s*\{' % self.init_func_name
        match_obj = re.search(init_func_re, init_code)
        if not match_obj:
            err (('orio.main.tuner.ptest_codegen: no initialization function (named "%s") can be found in the ' +
                    'initialization file: "%s"') % (self.init_func_name, self.init_file))

    #-----------------------------------------------------

    def __checkSkeletonCodeFile(self):
        '''To check the skeleton-code file, and return the skeleton code'''

        # do not perform checking if no skeleton-code file is specified
        if not self.skeleton_code_file:
            return None

        # read the content of the file
        try:
            f = open(self.skeleton_code_file)
            skton_code = f.read()
            f.close()
        except:
            err('orio.main.tuner.ptest_codegen:  cannot read file: "%s"' % self.decl_file)

        # return the skeleton code
        return skton_code

    #-----------------------------------------------------

    def generate(self, code_map):
        '''
        Generate the testing code, which is evaluated to get the performance cost.

        @return: The test C code string        
        @param code_map: A dictionary index by search space coordinates and containing code to be evaluated. 
        '''

        # generate the macro definition codes for the input parameters
        iparam_code = self.iparam_code

        # generate the declaration code
        if self.decl_file:
            decl_code = '#include "%s"\n' % self.decl_file
        else:
            decl_code = self.decl_code + '\n'
            decl_code += 'void %s() {\n%s\n}\n' % (self.malloc_func_name, self.malloc_code)

        # generate the initialization code
        if self.init_file:
            init_code = '#include "%s"\n' % self.init_file
        else:
            init_code = 'void %s() {\n%s\n}\n' % (self.init_func_name, self.init_code)

        # create code for the global definition section
        global_code = ''
        global_code += iparam_code + '\n'
        global_code += decl_code + '\n'
        global_code += init_code + '\n'

        # create code for the prologue section
        prologue_code = ''
        if not self.decl_file:
            prologue_code += ('%s();' % self.malloc_func_name) + '\n'
        prologue_code += ('%s();' % self.init_func_name) + '\n'

        # create code for the epilogue section
        # TODO: Here we should put validation with original results
        epilogue_code = ''

        # get the performance-testing code
        ptest_code = self.ptest_skeleton_code.insertCode(global_code, prologue_code,
                                                         epilogue_code, code_map)

        # return the performance-testing code
        return ptest_code
    
    def getTimerCode(self, use_parallel_search = False):
        if not use_parallel_search:
            return SEQ_TIMER
        else: 
            return ''     
        
# --------------------------------------------------------------------------------------
class PerfTestCodeGenCUDA(PerfTestCodeGen):

    def foo(self):
        pass
     

# --------------------------------------------------------------------------------------

class PerfTestCodeGenFortran:
    '''
    The code generator used to produce a performance-testing code.
    
    The Fortran driver differs from the C one in the following ways:
        - The timer is in C and built in a separate file because of the difficulties in 
        getting a working, high-resolution, portable timing routine.
        - The declarations and initializations are embedded in the main program
        instead of specified as separate subroutines.
    '''

    #-----------------------------------------------------

    def __init__(self, input_params, input_decls, decl_file, init_file, skeleton_code_file, language='fortran',
                 random_seed=None, use_parallel_search=False):
        '''To instantiate the testing code generator'''
        
        self.input_params = input_params
        self.input_decls = input_decls
        self.decl_file = decl_file
        self.init_file = init_file
        self.skeleton_code_file = skeleton_code_file
        self.random_seed = random_seed
        self.use_parallel_search = use_parallel_search

        self.iparam_code = self.__genIParams(input_params)
        self.decl_code = self.__genDecls(input_decls)
        self.malloc_code = self.__genMAllocs(input_decls)
        self.init_code = self.__genInits(input_decls)

        self.__checkDeclFile()
        self.__checkInitFile()
        scode = self.__checkSkeletonCodeFile()
        self.ptest_skeleton_code = skeleton_code.PerfTestSkeletonCodeFortran(scode, use_parallel_search)
        
    #-----------------------------------------------------

    def __genIParams(self, input_params):
        '''
        Generate declaration code for:
         - input parameters (using preprocessor definitions)
        '''
        
        # generate definitions for each input parameter
        iparams = ['#define %s %s' % (pname, rhs) for pname, rhs in input_params]
        
        # generate and return the input parameter code
        iparam_code = '\n'.join(iparams)
        return iparam_code

    #-----------------------------------------------------

    def __genDecls(self, input_decls):
        '''
        Generate declaration code for:
         - declarations for the input variables
        '''
        

        # generate the input variable declarations
        decls = []
        
        # generate iteration variables
        max_dim = 0
        for _,_,_,vdims,_ in input_decls:
            max_dim = max(max_dim, len(vdims))
        iter_vars = map(lambda x: 'i%s' % x, range(1, max_dim+1))        
        # generate code for the declaration of the iteration variables
        if len(iter_vars) == 0:
            ivars_decl_code = ''
        else:
            ivars_decl_code = 'integer :: %s' % ', '.join(iter_vars)

        for is_static, vt, vname, vdims, rhs in input_decls:
            
            if vt in ['single','double']: vtype = 'real(%s)' % vt
            elif vt == 'int': vtype = 'integer'
            else: vtype = vt
            
            #TODO: handle structs (look for period in vname)
            if len(vdims) == 0:
                decls.append('%s %s' % (vtype, vname))
            else:
                if is_static:
                    dim_code = '(%s)' % ')('.join(':')
                    # e.g., real, dimension(100,200) :: d
                    decls.append('%s, dimension%s :: %s' % (vtype, dim_code, vname))
                else:
                    dim_code = '(%s)' % ')('.join(':')
                    decls.append('%s, dimension%s, allocatable :: %s' % (vtype, dim_code, vname))
                            
        # generate and return the declaration code
        decl_code = ivars_decl_code + '\n' + '\n'.join(decls)
        return decl_code

    #-----------------------------------------------------
    
    def __genMAllocs(self, input_decls):
        '''
        Generate declaration code for:
         - memory allocations for input arrays (that are dynamic arrays)
        '''

        
        # generate memory allocations for dynamic input arrays
        mallocs = []
        for is_static, vt, vname, vdims, rhs in input_decls:

            if len(vdims) > 0 and not is_static:
                dim_code = '(%s)' % ')('.join(vdims)
                mallocs.append('allocate(%s%s)\n' % (vname, dim_code))
        
        # return an empty code if no dynamic memory allocation is needed
        if len(mallocs) == 0:
            return ''

        # generate and return the declaration code
        malloc_code = '\n'.join(mallocs)
        malloc_code = '  ' + re.sub('\n', '\n  ', malloc_code)
        return malloc_code

    #-----------------------------------------------------
    
    def __genInits(self, input_decls):
        '''
        Generate code for:
         - value initializations for all input variables
        '''

        # generate iteration variables
        max_dim = 0
        for _,_,_,vdims,_ in input_decls:
            max_dim = max(max_dim, len(vdims))
        iter_vars = map(lambda x: 'i%s' % x, range(1, max_dim+1))
        
        # generate array value initializations
        inits = []
        for _, vt, vname, vdims, rhs in input_decls:

            if vt in ['single','double']: vtype = 'real(%s)' % vt
            elif vt == 'int': vtype = 'integer'
            else: vtype = vt
            
            # skip if it does not have an initial value (i.e. RHS == None)
            if rhs == None:
                continue

            # if it is a scalar
            if len(vdims) == 0:
                if rhs == 'random':
                    rhs = '%s' % (random.uniform(1, 10))
                inits.append('%s = %s' % (vname, rhs))
                continue

            # generate array value initialization code
            rank = len(vdims)
            used_iter_vars = iter_vars[:rank]
            loop_code = ''
            for i, (ivar, dim) in enumerate(zip(used_iter_vars, vdims)):
                loop_code += (' ' * i) + 'do %s = 0, %s\n' % (ivar, dim)
            dim_code = '(%s)' % ')('.join(used_iter_vars)
            if rhs == 'random':
                sum = '(%s)' % '+'.join(used_iter_vars)
                rhs = 'modulo(%s,%s) + %s' % (sum, 5, 1)
            loop_body_code = (' ' * rank) + '%s%s = %s' % (vname, dim_code, rhs)
            loop_body_code += '\nend do'
            inits.append(loop_code + loop_body_code)

        # return an empty code if no initialization is needed
        if len(inits) == 0:
            return ''

        # generate and return the initialization code
        init_code = '\n'.join(inits)
        init_code = '  ' + re.sub('\n', '\n  ', init_code)
        return init_code
    
    #-----------------------------------------------------

    def __checkDeclFile(self):
        '''To check the declaration file'''

        # do not perform checking if no declaration file is specified
        if not self.decl_file:
            return

        # check if the file can be opened
        try:
            f = open(self.decl_file)
            f.close()
        except:
            err('orio.main.tuner.ptest_codegen:  cannot read file: "%s"' % self.decl_file)

    #-----------------------------------------------------

    def __checkInitFile(self):
        '''To check the initialization file'''

        # do not perform checking if no initialization file is specified
        if not self.init_file:
            return

        # read the content of the file
        try:
            f = open(self.init_file)
            init_code = f.read()
            f.close()
        except:
            err('orio.main.tuner.ptest_codegen:  cannot read file: "%s"' % self.init_file)

    #-----------------------------------------------------

    def __checkSkeletonCodeFile(self):
        '''To check the skeleton-code file, and return the skeleton code'''

        # do not perform checking if no skeleton-code file is specified
        if not self.skeleton_code_file:
            return None

        # read the content of the file
        try:
            f = open(self.skeleton_code_file)
            skton_code = f.read()
            f.close()
        except:
            err('orio.main.tuner.ptest_codegen:  cannot read file: "%s"' % self.decl_file)

        # return the skeleton code
        return skton_code

    #-----------------------------------------------------

    def getTimerCode(self, use_parallel_search = False):
        if not use_parallel_search:
            return SEQ_TIMER
        else: 
            return ''     
        
    def generate(self, code_map):
        '''
        Generate the testing code, which is evaluated to get the performance cost.

        @return: The test C code string        
        @param code_map: A dictionary index by search space coordinates and containing code to be evaluated. 
        '''

        # generate the macro definition codes for the input parameters
        iparam_code = self.iparam_code

        # generate the declaration code
        if self.decl_file:
            decl_code = '#include "%s"\n' % self.decl_file
        else:
            decl_code = self.decl_code + '\n'
            decl_code += '%s\n' % (self.malloc_code)

        # generate the initialization code
        if self.init_file:
            init_code = '#include "%s"\n' % self.init_file
        else:
            init_code = '%s\n' % (self.init_code)

        # create code for the global definition section
        declaration_code = ''
        declaration_code += iparam_code + '\n'
        declaration_code += decl_code + '\n'

        # create code for the prologue section
        prologue_code = init_code + '\n'


        # create code for the epilogue section
        epilogue_code = ''

        # get the performance-testing code
        ptest_code = self.ptest_skeleton_code.insertCode(declaration_code, prologue_code,
                                                         epilogue_code, code_map)
        
        # return the performance-testing code
        return ptest_code
