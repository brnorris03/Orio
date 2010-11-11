#
# The class for the polyhedral-syntactic combined transformation
#

import re, os, sys
from orio.main.util.globals import *
import cloop_parser, macro_expander, orio.module.module, parser, poly_transformation, profiler
import syn_transformation, transf_info


#-----------------------------------------

class PolySyn(orio.module.module.Module):
    '''Polyhedral-syntactic combined transformation module.'''

    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C'):
        '''To instantiate a polyhedral-syntactic combined transformation module.'''

        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)

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
            err('orio.module.polysyn.polysyn: cannot find the opening and closing tags for the Pluto code') 

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

        # parse the orio.module.body code
        assigns = parser.Parser().parse(self.module_body_code, self.line_no)

        # extract transformation information from the specified assignments
        tinfo = transf_info.TransfInfoGen().generate(assigns, self.perf_params)

        # perform polyhedral transformations
        ptrans = poly_transformation.PolyTransformation(Globals().verbose,
                                                      tinfo.parallel,
                                                      tinfo.tiles)
        pluto_code = ptrans.transform(self.annot_body_code)
        
        # use a profiling tool (i.e. gprof) to get hotspots information
        prof = profiler.Profiler(Globals().verbose,
                                 tinfo.profiling_code,
                                 tinfo.compile_cmd,
                                 tinfo.compile_opts)
        hotspots_info = prof.getHotspotsInfo(pluto_code)

        # parse the Pluto code to extract hotspot loops
        pluto_code = cloop_parser.CLoopParser().getHotspotLoopNests(pluto_code, hotspots_info)

        # expand all macro-defined statements
        pluto_code = macro_expander.MacroExpander().replaceStatements(pluto_code)

        # perform syntactic transformations
        strans = syn_transformation.SynTransformation(Globals().verbose,
                                                    tinfo.permut,
                                                    tinfo.unroll_factors,
                                                    tinfo.scalar_replace,
                                                    tinfo.vectorize,
                                                    tinfo.rect_regtile)
        transformed_code = strans.transform(pluto_code)

        # return the transformed code
        return transformed_code

        
