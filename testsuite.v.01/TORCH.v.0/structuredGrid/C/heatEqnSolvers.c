
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "gridUtil.h"
#include "heatEqnSolvers.h"
#include "testFunctions.h"
#include "differentialOps.h"

#include "sparseUtil.h"
#include "generators.h"
#include "sparseArithmetic.h"

#include "complexUtil.h"
#include "ffts.h"


/*
 A selcetion of methods for solving the three dimensional heat equation.

 Alex Kaiser, LBNL, 7/2010
 */


gridArray initGridsHeatEqn(int tSteps, double height, double maxX, double maxY, double maxZ, double dt){
	/*
	 Allocates a grid array for use in heat equation solves.
	 First grid is initialized to heat equation initial conditions.
	 All others are zeroed.

	 Input:
	 int tSteps   Number of grids to allocate.
	 double height     Grid height.
	 double maxX  Maximum x value.
	 double maxY  Maximum y value.
	 double maxZ  Maximum z value.
	 double dt   Time step.

	 Output:
	 gridArray currentGrids (returned)  An array of allocated grids.
	    currentGrids.grids[0] is initialized to heat equation initial conditions
	    All subsequent grid arrays are zeroed.
	 */


	int i;

	gridArray currentGrids ;
	currentGrids.tSteps = tSteps ;
	currentGrids.dt = dt;
	currentGrids.grids = (grid *) malloc(tSteps * sizeof(grid)) ;
	currentGrids.grids[0] = initGrid3d(heatEqnInitialConds, height, maxX, maxY, maxZ, 0, 0.0) ;

	for(i=1; i<tSteps; i++)
		currentGrids.grids[i] = initZeroGrid3d(currentGrids.grids[0].m, currentGrids.grids[0].n, currentGrids.grids[0].p, currentGrids.grids[0].height) ;


	return currentGrids ;
}


gridArray initSolnHeatEqn(int tSteps, double height, double maxX, double maxY, double maxZ, double dt){
	/*
	 Allocates and initializes a grid array to the analytic solution of the Heat Eqn.
	 Grids were initialized to

	 Input:
	 int tSteps   Number of grids to allocate.
	 double height     Grid height.
	 double maxX  Maximum x value.
	 double maxY  Maximum y value.
	 double maxZ  Maximum z value.
	 double dt   Time step.

	 Output:
	 gridArray currentGrids (returned)  An array of allocated grids, set to the analytic solution of the Heat Eqn.
	 */


	int step, i, j, k;
	double t = 0.0;

	gridArray currentGrids ;
	currentGrids.tSteps = tSteps ;
	currentGrids.dt = dt;
	currentGrids.grids = (grid *) malloc(tSteps * sizeof(grid)) ;
	currentGrids.grids[0] = initGrid3d(heatEqnInitialConds, height, maxX, maxY, maxZ, 0, 0) ;

	int m = currentGrids.grids[0].m ;
	int n = currentGrids.grids[0].n ;
	int p = currentGrids.grids[0].p ;

	for(step=1; step<tSteps; step++){
		t += dt ;
		currentGrids.grids[step] = initZeroGrid3d(m, n, p, currentGrids.grids[0].height) ;
		for(i=0; i<m+2; i++){
			for(j=0; j<n+2; j++){
				for(k=0; k<p+2; k++){
					currentGrids.grids[step].values[i][j][k] = heatEqnAnalyticSoln(i*height, j*height, k*height, t, 0, 0) ;
				}
			}
		}
	}

	return currentGrids ;
}

gridArray solveHeatEquationExplicit(double dt, int tSteps, double height){
	/*
	 Explicit solve for the heat equation in three dimensions.
	 Matrix-free, grid based version.

	 Input:
	 double dt   Time step.
	 int tSteps  Number of time steps to perform.
	 double height Grid height

	 Output:
	 gridArray numericalSolution   Approximation to the solution of the heat equation
	 */

	// Length of region in question.
	// Use 1 for the 3d unit cube
	double l = 1.0;
	gridArray numericalSolution = initGridsHeatEqn(tSteps, height, l, l, l, dt);

	double lambda = dt / (height*height);
	if(lambda > (1.0/6.0))
		fprintf(stderr, "Warning: unstable grid parameters. Solution is unreliable.\nlambda = %f\n\n", lambda) ;

	int j;

	for(j=1; j<tSteps; j++){
		laplacian(numericalSolution.grids[j-1], &numericalSolution.grids[j] ) ;
		multiplyGridByConstant(&numericalSolution.grids[j], dt) ;
		addGrids(numericalSolution.grids[j-1], numericalSolution.grids[j], &numericalSolution.grids[j]) ;
	}

	return numericalSolution ;
}


gridArray solveHeatEquationExplicitVector(double dt, int tSteps, double height){
	/*
	 Explicit solve for the heat equation in three dimensions.
	 Matix based version.

	 Input:
	 double dt   Time step.
	 int tSteps  Number of time steps to perform.
	 double height Grid height

	 Output:
	 gridArray numericalSolution   Approximation to the solution of the heat equation
	 */

	// Length of region in question.
	// Use 1 for the 3d unit cube
	double l = 1.0;
	gridArray numericalSolution = initGridsHeatEqn(tSteps, height, l, l, l, dt);

	int vectorLength = pow(numericalSolution.grids[0].n, 3);



	double lambda = dt / (height*height);
	if(lambda > (1.0/6.0))
		fprintf(stderr, "Warning: unstable grid parameters. Solution is unreliable.\nlambda = %f\n\n", lambda) ;


	csrMatrix A = getHeatEqnMatrix(numericalSolution.grids[0].n, lambda) ;


	int j;

	// Use SpMV. Un-comment the pair of comments below to use this block.
	/*
	double *inputBuf = (double *) malloc(vectorLength * sizeof(double));
	double *outputBuf = (double *) malloc(vectorLength * sizeof(double));
	for(j=1; j<tSteps; j++){
		gridToVector(numericalSolution.grids[j-1], inputBuf) ;
		spmv(A, inputBuf, outputBuf) ;
		vectorToGrid(&numericalSolution.grids[j], outputBuf) ;
	}
	*/

	// Use matrix powers. Re-comment the pair of comments below to cover whole block.
	// /*
	double ** buffer = allocDouble2d( (int) tSteps, vectorLength) ;
	gridToVector(numericalSolution.grids[0], buffer[0]) ;

	matrixPowers(A, (int) tSteps, buffer) ;

	// return to grids.
	for(j=1; j<tSteps; j++)
		vectorToGrid(&numericalSolution.grids[j], buffer[j]) ;

	freeDouble2d(buffer, (int) tSteps, vectorLength) ;
	// */

	return numericalSolution ;
}

gridArray solveHeatEquationImplicit(double dt, int tSteps, double height, double tol){
	/*
	 Implicit solve for the heat equation in three dimensions.
	 Matrix free, grid based version.

	 Input:
	 double dt   Time step.
	 int tSteps  Number of time steps to perform.
	 double height  Grid height.
	 double tol  Tolerance parameter for linear system solve.

	 Output:
	 gridArray numericalSolution   Approximation to the solution of the heat equation
	 */

	// Length of region in question.
	// Use 1 for the 3d unit cube
	double l = 1.0;
	gridArray numericalSolution = initGridsHeatEqn(tSteps, height, l, l, l, dt);

	int maxIt = pow(numericalSolution.grids[0].m, 3) ;
	int j;

	for(j=1; j<tSteps; j++){
		numericalSolution.grids[j] = conjugateGradientGrid(numericalSolution.grids[j-1], numericalSolution.grids[j-1],
																numericalSolution.dt, maxIt, numericalSolution.grids[j], tol) ;
	}

	return numericalSolution ;
}

gridArray solveHeatEquationImplicitVector(double dt, int tSteps, double height, double tol){
	/*
	 Implicit solve for the heat equation in three dimensions.
	 Matrix based version.

	 Input:
	 double dt   Time step.
	 int tSteps  Number of time steps to perform.
	 double height  Grid height.
	 double tol  Tolerance parameter for linear system solve.

	 Output:
	 gridArray numericalSolution   Approximation to the solution of the heat equation
	 */

	// Length of region in question.
	// Use 1 for the 3d unit cube
	double l = 1.0;
	gridArray numericalSolution = initGridsHeatEqn(tSteps, height, l, l, l, dt);

	int maxIt = pow(numericalSolution.grids[0].m, 3) ;
	int j;

	double lambda = dt / (height*height);

	int vectorLength = pow(numericalSolution.grids[0].n, 3);
	double *inputBuf = (double *) malloc(vectorLength * sizeof(double));
	double *outputBuf = (double *) malloc(vectorLength * sizeof(double));

	csrMatrix A = getHeatEqnMatrixImplicit(numericalSolution.grids[0].n, lambda) ;

	for(j=1; j<tSteps; j++){
		gridToVector(numericalSolution.grids[j-1], inputBuf) ;
		outputBuf = conjugateGradient(A, inputBuf, inputBuf, tol, maxIt) ;
		vectorToGrid(&numericalSolution.grids[j], outputBuf) ;
	}

	return numericalSolution ;
}

gridArray solveHeatEquationSpectral(double dt, int tSteps, double height){
	/*
	 Explicit solve for the heat equation in three dimensions.
	 Matrix-free, grid based version.

	 Input:
	 double dt   Time step.
	 int tSteps  Number of time steps to perform.
	 double height Grid height

	 Output:
	 gridArray numericalSolution   Approximation to the solution of the heat equation
	 */

	// Length of region in question.
	// Use 1 for the 3d unit cube
	double l = 1.0;
	gridArray numericalSolution = initGridsHeatEqn(tSteps, height, l, l, l, dt);


	// organize data and twiddle factors for ffts
	int N = numericalSolution.grids[0].n ;
	int n1;
	int n2;

	int i,j,k;
	int iBar, jBar, kBar;

	double invFactor = 1.0 / (double) (N*N*N);

	int log2Length = floor(log2( (double) N )) ;

	if( (log2Length % 2) == 1){
		n1 = 0x1 << ((log2Length / 2) + 1) ;
		n2 = 0x1 << (log2Length / 2) ;
	}
	else{
		n1 = 0x1 << (log2Length / 2) ;
		n2 = 0x1 << (log2Length / 2) ;
	}

	if( n1 * n2 != N ){
		fprintf(stderr, "Problems in factorization size. Size must be a power of two.\nExiting\n") ;
		exit(-1) ;
	}


	// allocate buffers for ffts
	struct complex *** U = allocComplex3d(N,N,N);
	struct complex *** X = allocComplex3d(N,N,N);


	// copy input data to complex arrays
	for(k=0; k<N; k++){
		for(j=0; j<N; j++){
			for(i=0; i<N; i++){
				setComplex(&U[i][j][k], numericalSolution.grids[0].values[i+1][j+1][k+1], 0.0) ;
			}
		}
	}


	// initialize exponential lookup table
	double pi = acos(-1) ;
	int tableSize = N;
	struct complex * table = allocComplex(tableSize);
	for(k=0; k<tableSize; k++){
		setComplex(&table[k], cos(k * 2 * pi / (double) tableSize ), sin( k * 2 * pi / (double) tableSize ) );
	}


	// initialize twiddles
	double *** twiddles = allocDouble3d(N,N,N);
	for(k=0; k<N; k++){
		for(j=0; j<N; j++){
			for(i=0; i<N; i++){

				if(i < N/2) iBar = i;
				else iBar = i-N;

				if(j < N/2) jBar = j;
				else jBar = j-N;

				if(k < N/2) kBar = k;
				else kBar = k-N;

				twiddles[i][j][k] = exp(-4 * pow(pi, 2) * ( pow(iBar,2.0) + pow(jBar,2.0) + pow(kBar,2.0) ));
			}
		}
	}


	// initial transform into frequency space
	U = fft3D(U, N, n1, n2, N, n1, n2, N, n1, n2, -1, table, tableSize );

	double t = 0.0;

	int step;
	for(step=1; step<tSteps; step++){
		t += dt ;

		// advance in time
		for(k=0; k<N; k++){
			for(j=0; j<N; j++){
				for(i=0; i<N; i++){
					X[i][j][k] = multComplexReal( U[i][j][k], pow(twiddles[i][j][k], t)) ;
				}
			}
		}

		// take inverse fft
		X = fft3D(X, N, n1, n2, N, n1, n2, N, n1, n2, 1, table, tableSize);

		// scale for inv fft
		for(k=0; k<N; k++){
			for(j=0; j<N; j++){
				for(i=0; i<N; i++){
					X[i][j][k] = multComplexReal( X[i][j][k], invFactor) ;
				}
			}
		}

		// copy data back to output arrays
		for(k=0; k<N; k++){
			for(j=0; j<N; j++){
				for(i=0; i<N; i++){
					numericalSolution.grids[step].values[i+1][j+1][k+1] = X[i][j][k].real ;
				}
			}
		}


	}

	return numericalSolution ;
}


gridArray solveHeatEquationMultiGrid(double dt, int tSteps, double height, double tol, int depth, int maxItDown, int maxItUp){
	/*
	 Implicit solve for the heat equation in three dimensions.

	 Input:
	 double dt   Time step.
	 int tSteps  Number of time steps to perform.
	 double height  Grid height.
	 double tol  Tolerance parameter for linear system solve.
	 int depth          Number of steps to descend in V-cycle.
     int maxItDown      Maximum number of iterations to perform on down cycle.
     int maxItUp        Maximum number of iterations to perform on up cycle.

	 Output:
	 gridArray numericalSolution   Approximation to the solution of the heat equation
	 */

	// Length of region in question.
	// Use 1 for the 3d unit cube
	double l = 1.0;
	gridArray numericalSolution = initGridsHeatEqn(tSteps, height, l, l, l, dt);

	double lambda = dt / (numericalSolution.grids[0].height * numericalSolution.grids[0].height);
	mgGridArray currentMgArray = allocMultiGridArrays(numericalSolution.grids[0].m, numericalSolution.grids[0].n,
														numericalSolution.grids[0].p, height, lambda, depth);

	int j;
	for(j=1; j<tSteps; j++){
		/*
		numericalSolution.grids[j] = gaussSeidelGrid(numericalSolution.grids[j-1], numericalSolution.grids[j-1],
																numericalSolution.dt, maxIt, numericalSolution.grids[j], tol) ;
		*/

		printf("Step = %d\n", j);

		numericalSolution.grids[j] = multiGridVCycle(numericalSolution.grids[j-1], numericalSolution.grids[j-1],
												currentMgArray, numericalSolution.grids[j], maxItDown, maxItUp, dt, tol) ;
	}

	return numericalSolution ;
}

grid conjugateGradientGrid(grid b, grid guess, double dt, double maxIt, grid x, double tol){
	/*
	 Conjugate gradient method for linear systems.
	 Matrix free version.
	 Specific to the implicit heat equation.
	 Modified from 'Matrix Computations' Golub and Van Loan

	 Input:
	 grid b   Right hand side of the system.
	 grid guess   Initial guess.
	 double dt   Time step.
	 double maxIt   Maximum number of iterations to perform.
	 grid x    Preallocated space for solution.
	 double tol  Tolerance parameter for solution to system.

	 Output:
	 grid x (returned) The solution.
	 */

	double rho, lastRho;
	double alpha, beta;
	int iter;

	copyGridValues(guess, &x) ;

	grid r = initZeroGrid3d(guess.m, guess.n, guess.p, guess.height) ;

	laplacian(x, &r) ;
	gridPlusConstantTimesGrid(x, -dt, r, &r) ;
	subtractGrids(b, r, &r) ;

	rho = normGrid(r) ;
	rho = rho * rho ;
	lastRho = rho ; // junk initial value

	grid w = initZeroGrid3d(guess.m, guess.n, guess.p, guess.height) ;
	grid p = initZeroGrid3d(guess.m, guess.n, guess.p, guess.height) ;

	double stopThreshold = tol * normGrid(b) ;

	iter = 0 ;

	while( sqrt(rho) > stopThreshold){
		if(iter == 0)
			copyGridValues(r, &p);
		else{
			beta = rho / lastRho;
			gridPlusConstantTimesGrid(r, beta, p, &p) ; // p = r + beta * p ;
		}

		laplacian(p, &w) ;
		gridPlusConstantTimesGrid(p, -dt, w, &w) ;   // w = -dt * w + p
		alpha = rho / innerProductGrids(p, w) ;
		gridPlusConstantTimesGrid(x, alpha, p, &x) ; // x = x + alpha * p
		gridPlusConstantTimesGrid(r, -alpha, w, &r) ; // r = r - alpha * w;

		lastRho = rho;
		rho = normGrid(r) ;
		rho = rho * rho ; // l2 norm squared

		iter++;
		if(iter > maxIt){
	        printf("cg grid hit max iterations. %d iterations performed.\n", iter );
	        return x;
		}
	}

	printf("cg grid converged in %d iterations.\n", iter) ;
	return x;
}

grid gaussSeidelGrid(grid b, grid guess, double dt, double maxIt, grid x, double tol){
	/*
	 Gauss Seidel method for linear systems.
     Matrix free version.
	 Specific to the implicit heat equation.
     Include ghost zones in all grids.
     Modified from http://www.netlib.org/linalg/html_templates/Templates.html


	 Input:
	 grid b           Right hand side of the system.
	 grid guess       Initial guess, serves to estimate the initial residual.
	 double dt        Time step.
	 double maxIt     Maximum number of iterations to perform.
	 grid x    Preallocated space for solution.
	 double tol   Tolerance for linear system solve.

	 Output:
	 grid x (returned) The solution.
	 */

	int iter ;
	int i,j,k;
	double lambda = dt / (b.height * b.height) ;
	double sum;

	copyGridValues(guess, &x) ;

	grid r = initZeroGrid3d(guess.m, guess.n, guess.p, guess.height) ;

	laplacian(x, &r) ;
	gridPlusConstantTimesGrid(x, -dt, r, &r) ;
	subtractGrids(b, r, &r) ;

	int m = guess.m;
	int n = guess.n;
	int p = guess.p;
	iter = 0;

	while( normGrid(r) > tol){

		for(i=1; i <= m; i++){
			for(j=1; j <= n; j++){
				for(k=1; k <= p; k++){
					sum = -lambda * ((x.values[i+1][j][k] + x.values[i-1][j][k]) +
									 (x.values[i][j+1][k] + x.values[i][j-1][k]) +
									 (x.values[i][j][k+1] + x.values[i][j][k-1])) ;

					x.values[i][j][k] = (b.values[i][j][k] - sum) / (1.0 + 6.0*lambda) ;
				}
			}
		}

		iter++;
		if(iter > maxIt){
	        printf("gs grid hit max iterations. %d iterations performed.\n", iter );
	        return x;
		}

		laplacian(x, &r) ;
		gridPlusConstantTimesGrid(x, -dt, r, &r) ;
		subtractGrids(b, r, &r) ;
	}

	printf("gs grid converged in %d iterations.\n", iter) ;
	return x;
}


grid multiGridVCycle(grid b, grid guess, mgGridArray currentMgArray, grid x, int maxItDown, int maxItUp, double dt, double tol){
	/*
	 Multi Grid V cyclve method for linear systems.
     Matrix free version.
	 Specific to the implicit heat equation.
	 Uses Gauss Seidel for relaxation steps.
     Include ghost zones in all grids.

     Modified from "A Multigrid Tutorial"
     Briggs, Henson, and McCormic


	 Input:
	 grid b           Right hand side of the system.
	 grid guess       Initial guess, serves to estimate the initial residual.
	 mgGridArray      Array of grids for multigrid solve. Preallocated and reusable.
	 grid x    Preallocated space for solution.
     int maxItDown      Maximum number of iterations to perform on down cycle.
     int maxItUp        Maximum number of iterations to perform on up cycle.
     double dt        Time step.
     double tol  Tolerance parameter for linear system solve.

	 Output:
	 grid x (returned) The solution.
	 */



	// some error checking
	if(currentMgArray.mgGrids[0].guess.m != b.m){
		fprintf(stderr, "Multigrid buffer arrays must match dimension of initial rhs.\nExiting\n");
		exit(-1);
	}
	if(currentMgArray.mgGrids[0].guess.m != guess.m){
		fprintf(stderr, "Multigrid buffer arrays must match dimension of initial guess.\nExiting\n");
		exit(-1);
	}


	copyGridValues(guess, &currentMgArray.mgGrids[0].guess) ;
	copyGridValues(b, &currentMgArray.mgGrids[0].rhs) ;

	int j;

	for(j=1; j<currentMgArray.depth; j++){
		// note:
		// currentMgArray.mgGrids[j-1] = fine
		// currentMgArray.mgGrids[j] = coarse
		currentMgArray.mgGrids[j-1].guess = gaussSeidelGrid(currentMgArray.mgGrids[j-1].rhs, currentMgArray.mgGrids[j-1].guess,
															dt, maxItDown, currentMgArray.mgGrids[j-1].guess, tol) ;

		// calculate residual
		laplacian(currentMgArray.mgGrids[j-1].guess, &currentMgArray.mgGrids[j-1].residual) ;
		gridPlusConstantTimesGrid(currentMgArray.mgGrids[j-1].guess, -dt, currentMgArray.mgGrids[j-1].residual, &currentMgArray.mgGrids[j-1].residual) ;
		subtractGrids(currentMgArray.mgGrids[j-1].rhs, currentMgArray.mgGrids[j-1].residual, &currentMgArray.mgGrids[j-1].residual) ;

		// interpolate residual to get new rhs for coarse grid
		fineToCoarse(currentMgArray.mgGrids[j-1].residual, &currentMgArray.mgGrids[j].rhs) ;

		// interpolate guess to fine grid
		fineToCoarse(currentMgArray.mgGrids[j-1].guess, &currentMgArray.mgGrids[j].guess) ;
	}

	// full solve at coarsest level
	int iterNecessary = pow(currentMgArray.mgGrids[currentMgArray.depth-1].guess.m, 3);
	printf("Full solve at coarsest level:\n");
	currentMgArray.mgGrids[currentMgArray.depth-1].guess = gaussSeidelGrid(currentMgArray.mgGrids[currentMgArray.depth-1].rhs,
																		   currentMgArray.mgGrids[currentMgArray.depth-1].guess, dt, iterNecessary,
																		   currentMgArray.mgGrids[currentMgArray.depth-1].guess, tol) ;


	for(j = currentMgArray.depth - 1 ; j > 0; j--){
		// note:
		// currentMgArray.mgGrids[j] = coarse
		// currentMgArray.mgGrids[j-1] = fine

		// interpolate to fine grid
		// store the result of the interpolation in the residual, because it's no longer needed and otherwise a new temp grid would be needed
		coarseToFine(currentMgArray.mgGrids[j].guess, &currentMgArray.mgGrids[j-1].residual) ;
		addGrids(currentMgArray.mgGrids[j-1].guess, currentMgArray.mgGrids[j-1].residual, &currentMgArray.mgGrids[j-1].guess) ;

		// relax on fine grid
		currentMgArray.mgGrids[j-1].guess = gaussSeidelGrid(currentMgArray.mgGrids[j-1].rhs, currentMgArray.mgGrids[j-1].guess,
															dt, maxItUp, currentMgArray.mgGrids[j-1].guess, tol) ;

	}

	// return data to output array
	copyGridValues(currentMgArray.mgGrids[0].guess, &x) ;
	return x;
}

void fineToCoarse(grid fine, grid *coarse){
	/*
	 Fine to coarse grid transfer operator
     Injection operator from fine grid to coarse grid.
     Only tested for square grids.

     If coarse grid is size n in each dimension, fine grid will be of size 2*n-1

     Input:
     grid coarse      Preallocated coarse grid
     grid fine        Grid to interpolate

     Output:
     grid coarse      Grid formed using injection operator from fine grid
	 */

	// some bounds checking
	if( (fine.m % 2) != 1){
		fprintf(stderr, "Length of fine array must be odd. Exiting.\n");
		exit(-1) ;
	}
	if( (fine.m - 1) != (2 * coarse->m) ){
		fprintf(stderr, "Length of fine minus one must be twice that of coarse. Exiting.\n");
		exit(-1) ;
	}


	int i,j,k;
	int coarseI, coarseJ, coarseK;

	for(i=0, coarseI=0; i <= fine.m + 1; i += 2, coarseI++){
		for(j=0, coarseJ=0; j <= fine.n + 1; j += 2, coarseJ++){
			for(k=0, coarseK=0; k <= fine.p + 1; k += 2, coarseK++){
				coarse->values[coarseI][coarseJ][coarseK] = fine.values[i][j][k] ;
			}
		}
	}
}

void coarseToFine(grid coarse, grid *fine){
	/*
    Converts a coarse grid to a fine grid.
    Uses an averaging operator
    If coarse grid is size n in each dimension, fine grid will be of size 2*n-1

    Input:
    grid coarse      Preallocated 3D coarse array of length (n + 1) / 2 in each dimension
    grid *fine       Address of fine grid

    Output:
    grid coarse      Grid formed using injection operator from fine grid
	 */


	// may want to check coarse dimensions
	if( (fine->m - 1) != (2 * coarse.m) ){
		fprintf(stderr, "Length of fine minus one must be twice that of coarse. Exiting.\n");
		exit(-1) ;
	}

	int i,j,k;
	int lengthCoarse = coarse.m + 2;
	int lengthFine = fine->m + 2;

	// copy existing points first
	for(i=0; i<lengthCoarse; i++){
		for(j=0; j<lengthCoarse; j++){
			for(k=0; k<lengthCoarse; k++){
				fine->values[2*i][2*j][2*k] = coarse.values[i][j][k] ;
			}
		}
	}

	// average available rows in x direction
	for(i=1; i < lengthFine - 1; i+=2){
		for(j=0; j<lengthFine; j+=2){
			for(k=0; k<lengthFine; k+=2){
				fine->values[i][j][k] = .5 * (fine->values[i-1][j][k] + fine->values[i+1][j][k]) ;
			}
		}
	}

	// average available columns in y direction
	for(i=0; i<lengthFine; i++){
		for(j=1; j < lengthFine - 1; j+=2){
			for(k=0; k < lengthFine; k+=2){
				fine->values[i][j][k] = .5 * (fine->values[i][j-1][k] + fine->values[i][j+1][k]) ;
			}
		}
	}

	// average remaining planes in z direction
	for(i=0; i<lengthFine; i++){
		for(j=0; j < lengthFine; j++){
			for(k=1; k < lengthFine - 1; k+=2){
				fine->values[i][j][k] = .5 * (fine->values[i][j][k-1] + fine->values[i][j][k+1]) ;
			}
		}
	}
}



char compareHeatEqnSoln(gridArray numericalSolution, double tolNorm, double tolMax){
	/*
	 Checks whether numerical solution matches discretization of analytic solution.
	 Must match tolerances at all time steps to pass.

	 Input:
	 gridArray numericalSolution  Approximation to the solution to check.
	 double tolNorm    Tolerance for l2 norm.
	 double tolMax     Tolerance for maximum difference.

	 Output:
	 char pass (returned) Whether norms are under tolerances at all time steps.
	 */

	gridArray analyticSoln = initSolnHeatEqn(numericalSolution.tSteps, numericalSolution.grids[0].height, 1.0, 1.0, 1.0, numericalSolution.dt) ;

	int j;
	double relErr, maxDiff ;
	char pass = 1;

	for(j=0; j<numericalSolution.tSteps; j++){

		printf("Step = %d\n", j);

		relErr = l2RelativeErrGrid(numericalSolution.grids[j], analyticSoln.grids[j]) ;

		printf("relErr = %e\n", relErr) ;

		if(relErr < tolNorm) {
			printf("Heat equation relative error tests passed.\n\n") ;
			pass &= 1;
		}
		else{
			fprintf(stderr, "Heat equation relative error tests failed.\n\n") ;
			pass = 0;
		}

		maxDiff = maxDiffGrid(numericalSolution.grids[j], analyticSoln.grids[j]) ;

		printf("maxDiff = %e\n", maxDiff) ;

		if(maxDiff < tolMax){
			printf("Heat equation max difference tests passed.\n\n") ;
			pass &= 1;
		}
		else{
			fprintf(stderr, "Heat equation max difference tests failed.\n\n") ;
			pass = 0;
		}

		printf("\n") ;
	}


	freeGridArray(analyticSoln) ;
	return pass;
}

