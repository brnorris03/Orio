class LoopAnn:

    def __init__(self, loop_counter, loop_vars, tiling_levels=1):
        self.tile_levels = tiling_levels
        self.loopvars = loop_vars
        self.vars = []
        self.counter = loop_counter
        self.indent = ''
        pass


    def getComposite(self,indent='  '):
        self.indent = "{0}{1}".format(indent, indent)
        ann = 'transform Composite(\n%s\n) ' % ',\n'.join(
            [
                self.getTile(),
                self.getUnrollJam(),
                self.getScalarReplace(),
                self.getVector(),
                self.getOpenMP()
            ]
        )
        return indent + ann

    def getTile(self):
        """ Example:
            tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T1_II,'iii'),(('jj','j'),T1_JJ,'jjj'),(('kk','k'),T1_KK,'kkk')],
        """
        triplets = []
        self.vars = []
        for loopvar in self.loopvars:

            new_loop_var = '_orio_%s%s' % (loopvar,loopvar)
            self.vars.append(loopvar)
            self.vars.append(new_loop_var)
            orio_param = ('T%d' % self.counter + '_%s' % loopvar).upper()
            triplets.append('(\'%s\', %s, \'%s\')' % (loopvar,orio_param,new_loop_var))
            if self.tile_levels == 2:
                new_loop_var2 = '_%s%s' % (new_loop_var,loopvar)
                self.vars.append(new_loop_var2)
                orio_param2 = ('T%d' % self.counter +'%s'%new_loop_var).upper()
                triplets.append('((\'%s\',\'%s\'), %s, \'%s\')' % (new_loop_var,loopvar,orio_param2,new_loop_var2))
        return self.indent + 'tile = [{0}]'.format(','.join(triplets))


    def getUnrollJam(self):
        """Example:
            unrolljam = (['i','j','k'],[U_I,U_J,U_K])
        """
        unroll_params = []
        for v in self.loopvars:
            unroll_params.append('U_%s' % v.upper())
        vars = []
        for v in self.loopvars:
            vars.append('\'%s\'' % v)
        return self.indent + 'unrolljam = ([{0}],[{1}])'.format(','.join(vars), ','.join(unroll_params))

    def getScalarReplace(self):
        """Example:
            scalarreplace = (SCREP, 'double', 'scv_')
        """
        return self.indent + 'scalarreplace = (SCREP, \'double\', \'_orio_scv_\')'

    def getVector(self):
        """Example:
            vector = (VEC, ['ivdep','vector always'])
        """
        return self.indent + 'vector = (VEC%d, [\'ivdep\',\'vector always\'])' % self.counter

    def getOpenMP(self):
        """Example:
            openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
        """
        if not self.vars: self.vars = self.loopvars
        return self.indent + 'openmp = (OMP{0}, \'omp parallel for private({1})\')'.format(str(self.counter), ','.join(self.vars))


if __name__ == "__main__":
    a = LoopAnn(1,['i','j','k'],tiling_levels=2)
    print a.getComposite('  ')