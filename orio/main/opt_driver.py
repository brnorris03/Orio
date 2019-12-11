#
# The optimization driver used to initiate the optimization process
#

import sys, traceback, os
from orio.main.util.globals import *
import orio.main.code_frag, orio.main.dyn_loader, orio.main.tuner.tuner

#----------------------------------------------------------------

# the name of the performance-tuning annotation
PTUNE_NAME = 'PerfTuning'
PRESERVE_NAME = 'Preserve'

# the name of the module containing various code transformations
TMOD_NAME = 'orio.module'

#----------------------------------------------------------------

class OptDriver:
    '''The optimization driver whose function is to initiate the optimization process'''

    def __init__(self, src, language='C'):
        '''To instantiate an optimization driver'''
        self.lang = language
        self.srcname = src
        self.ptuner = orio.main.tuner.tuner.PerfTuner(self)
        self.dloader = orio.main.dyn_loader.DynLoader()
        self.input_params = None
    
    #-------------------------------------------------------------

    def optimizeCodeFrags(self, cfrags, perf_params, is_top_level = False):
        '''Apply optimizations specified in the annotations to each code fragment'''

        # check for the validity of performance tuning annotations
        if is_top_level:
            self.__checkPerfTuningAnns(cfrags)

        # apply optimizations on each code fragment

        optimized_code_seq = None
        
        for cf in cfrags:
            
            cur_seq = self.__optimizeCodeFrag(cf, perf_params)
            if optimized_code_seq == None:
                optimized_code_seq = cur_seq
            else:
                optimized_code_seq = [((c1 + c2), (i1 + i2), (e1 + e2)) \
                                      for c1, i1, e1 in optimized_code_seq for c2, i2, e2 in cur_seq]

        # return the optimized code
        return optimized_code_seq

    #-------------------------------------------------------------

    def __checkPerfTuningAnns(self, cfrags):
        '''
        To ensure that:
        1. Nested performance-tuning annotation does not exist
        2. No more than one performance-tuning annotation exists
        '''

        # iterate over all code fragments
        num_ptune_anns = 0
        for cf in cfrags:
    
            # if a top-level annotation code region
            if isinstance(cf, orio.main.code_frag.AnnCodeRegion) and cf.leader_ann.mod_name == PTUNE_NAME:

                # increment the total number of performance-tuning annotations
                num_ptune_anns += 1

                # if the total number of top-level performance-tuning annotations is more than one
                if num_ptune_anns > 1:
                    err(('orio.main.opt_driver: %s: the total number of performance-tuning annotations cannot ' +
                         'be more than one') % cf.leader_ann.mod_name_line_no)

                # iterate over all nested code fragments
                nested_cfrags = cf.cfrags[:]
                while len(nested_cfrags) > 0:
                    nested_cf = nested_cfrags.pop(0)

                    # if the nested code fragment is a performance-tuning annotation
                    if (isinstance(nested_cf, orio.main.code_frag.LeaderAnn) and
                        nested_cf.mod_name == PTUNE_NAME):
                        err(('orio.main.opt_driver: %s: performance-tuning annotations must be defined at ' +
                                'top level and cannot be nested') % nested_cf.line_no)
                        sys.exit(1)
                                                
                    # if the nested code fragment is an annotation code region
                    if isinstance(nested_cf, orio.main.code_frag.AnnCodeRegion):
                        nested_cfrags.append(nested_cf.leader_ann)
                        nested_cfrags.extend(nested_cf.cfrags)
                        nested_cfrags.append(nested_cf.trailer_ann)
                    
    #-------------------------------------------------------------

    def __optimizeCodeFrag(self, cfrag, perf_params):
        '''Apply optimization described in the annotations to the given code fragment.
        '''

        debug('__optimizeCodeFrag: code_frag type is ' + cfrag.__class__.__name__, self)
        
        # apply no optimizations to non-annotation code fragment
        if isinstance(cfrag, orio.main.code_frag.NonAnn):
            debug("OptDriver::__optimizeCodeFrag line 106", self)
            return [(cfrag.code, [], '')]

        # optimize annotated code region
        elif isinstance(cfrag, orio.main.code_frag.AnnCodeRegion):

            debug("OptDriver line 113: %s" % cfrag.leader_ann.mod_name, self)
            # initiate empirical performance tuning
            if cfrag.leader_ann.mod_name == PTUNE_NAME:
                debug("OptDriver line 116, detected tuning spec", self)
                # apply empirical performance tuning
                optimized_code_seq = self.ptuner.tune(cfrag.leader_ann.mod_code,
                                                      cfrag.leader_ann.mod_code_line_no,
                                                      cfrag.cfrags)
                indent = ' ' * cfrag.leader_ann.indent_size
                # after multi-input tuning build an input range selector
                if len(optimized_code_seq) > 1:
                    iselect = '\n' + indent + 'if ('
                    iselect_flag = True
                    externals_acc = ''
                    for optimized_code, input_params, externals in optimized_code_seq:
                        if iselect_flag: iselect_flag=False
                        else: iselect += ' else if ('
                        and_flag = True
                        for pname, pval in input_params:
                            if and_flag: and_flag = False
                            else: iselect += ' && '
                            iselect += '(%s<=%s)' % (pname, pval)
                        iselect += ') {\n%s' % optimized_code + '}'
                        externals_acc += externals
                    iselect += '\n'
                    optimized_code_seq = [(iselect, [], externals_acc)]

            # initiate code transformation and generation
            else:
                debug("OptDriver line 142, detected code annotated for tuning", self)

                # recursively apply optimizations to the annotation body
                optimized_body_code_seq = self.optimizeCodeFrags(cfrag.cfrags, perf_params)

                # check the optimized body code sequence
                if len(optimized_body_code_seq) != 1:
                    err('orio.main.opt_driver internal error:  the optimized annotation body code cannot ' +
                           'be multiple versions')
                    sys.exit(1)

                # get the optimized body code
                optimized_body_code, _, inner_ext = optimized_body_code_seq[0]

                # dynamically load the transformation module class
                class_name = cfrag.leader_ann.mod_name
                mod_name = '.'.join([TMOD_NAME, class_name.lower(), class_name.lower()])
                
                debug('about to load module.class %s.%s corresponding to annotation %s' % (mod_name,class_name,class_name), self)
                try:
                    mod_class = self.dloader.loadClass(mod_name, class_name)
                except Exception, e:
                    err('orio.main.opt_driver: %s: unable to load class %s.%s' % (cfrag.leader_ann.mod_name_line_no,mod_name,class_name))
                    
                debug("about to instantiate transformation class: %s.%s" %(mod_name,class_name), self)
                debug("perf_params=" + str(perf_params),self,level=6)

                # apply code transformations
                # This instantiates the transformation module and initializes it with the
                # code fragments and tuning spec information
                try:
                    if self.lang == 'cuda' or self.lang == 'opencl':
                        transformation = mod_class(perf_params,
                                                  cfrag.leader_ann.mod_code,
                                                  optimized_body_code,
                                                  cfrag.leader_ann.mod_code_line_no,
                                                  cfrag.leader_ann.indent_size,
                                                  language=self.lang,
                                                  tinfo=self.ptuner.tinfo)
                    else:
                        transformation = mod_class(perf_params,
                                                  cfrag.leader_ann.mod_code,
                                                  optimized_body_code,
                                                  cfrag.leader_ann.mod_code_line_no,
                                                  cfrag.leader_ann.indent_size,
                                                  language=self.lang,
                                                  tinfo=self.ptuner.tinfo)
                except Exception, e:
                    err('orio.main.opt_driver: %s: encountered an error when transforming annotation "%s"\n --> %s: %s' %
                           (cfrag.leader_ann.mod_name_line_no, cfrag.leader_ann.mod_name,e.__class__.__name__, e))
                    
                debug("successfully instantiated transformation class: %s.%s" %(mod_name,class_name),self)
                
                try:
                    optimized_code = transformation.transform()
                except Exception, e:
                    err('orio.main.opt_driver: encountered an error during transformation %s:\n %s' % (transformation,e)) 

                
                # create the optimized code sequence
                g = Globals()
                externals = ''
                if len(g.cunit_declarations) > 0:
                    externals = reduce(lambda x,y: x + y, g.cunit_declarations)
                    g.cunit_declarations = []
                optimized_code_seq = [(optimized_code, [], inner_ext + externals)]

            # prepend the leader annotation and append the trailer annotation to each of
            # the optimized code
            leader_code = cfrag.leader_ann.code
            trailer_code = ' ' * cfrag.trailer_ann.indent_size + cfrag.trailer_ann.code
            optimized_code_seq = [((leader_code + c + trailer_code), i, e) for c, i, e in optimized_code_seq]

            # return the optimized code sequence
            return optimized_code_seq

        
        # unexpected type of code fragment
        else:
            err('orio.main.opt_driver internal error:  unexpected type of code fragment',doexit=True)

