#
# The class for the polyhedral-syntactic combined transformation
#

import re, os, sys
import cloop_parser, macro_expander, module.module, parser, poly_transformator, profiler
import syn_transformator, transf_info

#-----------------------------------------

class PolySyn(module.module.Module):
    '''Polyhedral-syntactic combined transformation module'''

    def __init__(self, perf_params, module_body_code, annot_body_code, cmd_line_opts,
                 line_no, indent_size):
        '''To instantiate a polyhedral-syntactic combined transformation module'''

        module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      cmd_line_opts, line_no, indent_size)

    #---------------------------------------------------------------------

    def __insertPolysynTags(self, code):
        '''To add opening and closing tags to indicate the beginning anf ending of the code
        to be transformed using Pluto'''

        # the used tags
        polysyn_open_tag = '/* polysyn start */'
        polysyn_close_tag = '/* polysyn end */'
        pluto_open_tag_re = r'/\*\s*pluto\s+start.*?\*/'
        pluto_close_tag_re = r'/\*\s*pluto\s+end.*?\*/'
        
        # find the opening and closing tags of the pluto code
        open_m = re.search(pluto_open_tag_re, code)
        close_m = re.search(pluto_close_tag_re, code)
        if (not open_m) or (not close_m):
            print ('error:polysyn: cannot find the opening and closing tags for the PLuTo code') 
            sys.exit(1) 

        # insert the polysyn tags
        code = (code[:open_m.start()] + polysyn_open_tag + '\n' + 
                code[open_m.start():close_m.end()] + '\n' + polysyn_close_tag + 
                code[close_m.end():])

        # return the modified code
        return code

    #---------------------------------------------------------------------

    def transform(self):
        '''To apply a polyhedral-syntactic transformation on the annotated code'''

        # remove all existing annotations
        annot_re = r'/\*@((.|\n)*?)@\*/'
        self.annot_body_code = re.sub(annot_re, '', self.annot_body_code)

        # insert polysyn tags
        self.annot_body_code = self.__insertPolysynTags(self.annot_body_code)

        # parse the module body code
        assigns = parser.Parser().parse(self.module_body_code, self.line_no)

        # extract transformation information from the specified assignments
        tinfo = transf_info.TransfInfoGen().generate(assigns, self.perf_params)

        # perform polyhedral transformations
        ptrans = poly_transformator.PolyTransformator(self.cmd_line_opts.verbose,
                                                      tinfo.parallel,
                                                      tinfo.tiles)
        pluto_code = ptrans.transform(self.annot_body_code)
        
        # use a profiling tool (i.e. gprof) to get hotspots information
        prof = profiler.Profiler(self.cmd_line_opts.verbose,
                                 tinfo.profiling_code,
                                 tinfo.compile_cmd,
                                 tinfo.compile_opts)
        hotspots_info = prof.getHotspotsInfo(pluto_code)

        # parse the Pluto code to extract hotspot loops
        pluto_code = cloop_parser.CLoopParser().getHotspotLoopNests(pluto_code, hotspots_info)

        # expand all macro-defined statements
        pluto_code = macro_expander.MacroExpander().replaceStatements(pluto_code)

        # perform syntactic transformations
        strans = syn_transformator.SynTransformator(self.cmd_line_opts.verbose,
                                                    tinfo.permut,
                                                    tinfo.unroll_factors,
                                                    tinfo.scalar_replace,
                                                    tinfo.vectorize,
                                                    tinfo.rect_regtile)
        transformed_code = strans.transform(pluto_code)

        # return the transformed code
        return transformed_code

        
