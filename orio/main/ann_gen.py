class LoopAnnotationGenerator:

    def __init__(self, loop_counter, loop_vars, tiling_levels=1):
        self.tile_levels = tiling_levels
        self.loopvars = loop_vars
        self.new_loopvars = []
        self.counter = loop_counter
        self.indent = ''
        self.perf_params = dict(tiling=[],unroll=[],scalar_replacement=[],vector=[],openmp=[])
        pass


    def getComposite(self,indent='  '):
        self.indent = "{0}{1}".format(indent, indent)
        ann = 'transform Composite(\n%s\n) ' % ',\n'.join(
            [
                self.getTile(),
                self.getUnrollJam(),
                self.getScalarReplace(),
                self.getVector(),
                self.getOpenMP(),
            ]
        )
        return indent + ann

    def get_perf_params(self):
        return self.perf_params

    def getTile(self):
        """ Example:
            tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T1_II,'iii'),(('jj','j'),T1_JJ,'jjj'),(('kk','k'),T1_KK,'kkk')],
        """
        triplets = []
        self.new_loopvars = []
        for loopvar in self.loopvars:

            new_loop_var = '_orio_%s%s' % (loopvar,loopvar)
            self.new_loopvars.append(new_loop_var)
            orio_param = ('T%d' % self.counter + '_%s' % loopvar).upper()
            self.perf_params['tiling'].append(orio_param)
            triplets.append('(\'%s\', %s, \'%s\')' % (loopvar,orio_param,new_loop_var))
            if self.tile_levels == 2:
                new_loop_var2 = '%s%s' % (new_loop_var,loopvar)
                self.new_loopvars.append(new_loop_var2)
                orio_param2 = ('T%d' % self.counter +'%s'%new_loop_var).upper()
                self.perf_params['tiling'].append(orio_param2)
                triplets.append('((\'%s\',\'%s\'), %s, \'%s\')' % (new_loop_var,loopvar,orio_param2,new_loop_var2))
        return self.indent + 'tile = [{0}]'.format(','.join(triplets))


    def getUnrollJam(self):
        """Example:
            unrolljam = (['i','j','k'],[U_I,U_J,U_K]) - inside Composite
            transform UnrollJam(ufactor=U1i, parallelize=PAR1i) - on its own; currently not supported
        """
        for v in self.loopvars:
            self.perf_params['unroll'].append('U%d_%s' % (self.counter,v.upper()))
        vars = []
        for v in self.loopvars:
            vars.append('\'%s\'' % v)
        return self.indent + 'unrolljam = ([{0}],[{1}])'.format(','.join(vars), ','.join(self.perf_params['unroll']))

    def getScalarReplace(self):
        """Example:
            scalarreplace = (SCREP, 'double', 'scv_')
        """
        orio_param = 'SCREP%d' % self.counter
        self.perf_params['scalar_replacement'].append(orio_param)
        return self.indent + 'scalarreplace = (%s, \'double\', \'_orio_scv_\')' % orio_param

    def getVector(self):
        """Example:
            vector = (VEC, ['ivdep','vector always'])
        """
        orio_param = 'VEC%d' % self.counter
        self.perf_params['vector'].append(orio_param)
        return self.indent + 'vector = (%s, [\'ivdep\',\'vector always\'])' % orio_param

    def getOpenMP(self):
        """Example:
            openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
        """
        orio_param = 'OMP%d' % self.counter
        self.perf_params['openmp'].append(orio_param)
        return self.indent + 'openmp = (OMP{0}, \'omp parallel for private({1})\')'\
            .format(orio_param, ','.join(self.loopvars + self.new_loopvars))



if __name__ == "__main__":
    a = LoopAnnotationGenerator(1,['i','j','k'],tiling_levels=2)
    print a.getComposite('  ')