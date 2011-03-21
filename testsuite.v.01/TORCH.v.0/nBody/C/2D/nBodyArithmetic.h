

/*
 Header for arithmetic and run routines for N Body calculations.

 Alex Kaiser, LBNL, 10/2010.
 */

// run routines
void runNaive(int n, int numSteps, int forceType, char output, int saveFreq, char verify, forceLog currentLog) ;

void runCutoff(int n, int numSteps, int forceType, char output, int saveFreq, char verify, forceLog currentLog) ;

void runBarnesHut(int n, int numSteps, int forceType, char output, int saveFreq, char verify, forceLog currentLog) ;


// force applications and movement routines.

void apply_force( particle_t *particle, particle_t *neighbor, int type ) ;

void treeForce( particle_t *particles, int j, node nn, int forceType );

void move( particle_t *p, int type ) ;
