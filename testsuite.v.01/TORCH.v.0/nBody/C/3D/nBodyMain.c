#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>



#include "nBodyUtil.h"
#include "nBodyArithmetic.h"
#include "generators.h"


/*
Main routine for N Body kernels in two dimensions.

Alex Kaiser, LBNL, 10/2010.

Sources:
Lonestar Benchmarks:
http://iss.ices.utexas.edu/lonestar/

Jim Demmel's cs267 page
http://www.cs.berkeley.edu/~demmel/cs267_Spr09/
*/


int main(){
	/*
	 Main routine for three dimensional N Body computations.

	 Default settings run each computation twice.
	 First, the naive and naive with cutoff are run with the same non-physical, repulsive force.
	 The RMS error between the forces computed with these methods is tested to be below tolerance.
	 Second, the Naive and Barnes Hut algorithm are run with a gravity type force.
	 Error between these forces is also evaluated.

	 Parameters:
	 int n                       Number of particles to simulate.
	 int numStep                 Number of time steps to simulate.
	 int forceType               Which force to use.
	 	                             Type 0 = gravity-like attractive force.
	                                 Type 1 = repulsive.
	 char output                 If true, prints a log of positions of particles to a text file.
	                                 Default = false.
	 const int saveFreq          If (output), positions of particles will be saved every this many time steps.
	 char verify                 Selects whether force logging for verification will be performed.
	                                 Default = true.
	 int numParticlesToCheck     Number of particles to verify.
	 int numStepsToCheck         Number of time steps to verify.
	 */


	printf("Beginning N-Body tests.\n\n") ;

    int n = 2000;
    int numSteps = 15 ;
    int forceType = 1 ;
    char output = 0;
    const int saveFreq = 2;
    char verify = 1;

    int numParticlesToCheck = 100 ;
    int numStepsToCheck = numSteps ;

    char pass[2] ;

    double tol = 1e-5;

    forceLog naiveForceLogCutoff = allocAndInitForceLog(numParticlesToCheck, numStepsToCheck);

    runNaive(n, numSteps, forceType, output, saveFreq, verify, naiveForceLogCutoff) ;

    forceLog cutoffForceLog = allocAndInitForceLog(numParticlesToCheck, numStepsToCheck);
    runCutoff(n, numSteps, forceType, output, saveFreq, verify, cutoffForceLog) ;

    pass[0] = verifyNBody(cutoffForceLog, naiveForceLogCutoff, tol) ;

    if(pass[0])
    	printf("Cutoff test compared accurately to naive.\nTest passed.\n\n") ;
    else
    	printf("Cutoff test did not compare accurately to naive.\nTest failed.\n\n") ;

    // free logs
    freeForceLog(naiveForceLogCutoff) ;
    freeForceLog(cutoffForceLog) ;

    // Switch to gravity type force for Barnes Hut.
    forceType = 0;

    // allow a more coarse approximation when using tree based approximations.
    tol = 1e-2;

    forceLog naiveForceLogGravity = allocAndInitForceLog(numParticlesToCheck, numStepsToCheck);

    runNaive(n, numSteps, forceType, output, saveFreq, verify, naiveForceLogGravity) ;

    forceLog barnesHutForceLog = allocAndInitForceLog(numParticlesToCheck, numStepsToCheck);
    runBarnesHut(n, numSteps, forceType, output, saveFreq, verify, barnesHutForceLog) ;

    pass[1] = verifyNBody(barnesHutForceLog, naiveForceLogGravity, tol) ;

    if(pass[1])
    	printf("Barnes Hut test compared accurately to naive.\nTest passed.\n\n") ;
    else
    	printf("Barnes Hut test did not compare accurately to naive.\nTest failed.\n\n") ;

    printf("N-Body tests complete.\n\n") ;

    if(pass[0] && pass[1])
    	printf("All tests passed.\n\n") ;
    else
    	fprintf(stderr, "Tests failed.\n\n") ;

    freeForceLog( naiveForceLogGravity ) ;
    freeForceLog( barnesHutForceLog ) ;

	return (pass[0] && pass[1]);
}








