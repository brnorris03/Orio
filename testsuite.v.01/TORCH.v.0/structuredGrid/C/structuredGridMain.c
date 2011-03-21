
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gridUtil.h"
#include "testFunctions.h"
#include "differentialOps.h"
#include "heatEqnSolvers.h"

#include "sparseUtil.h"
#include "generators.h"
#include "sparseArithmetic.h"

#include "complexUtil.h"

/*
 Structured grid kernels.

 Alex Kaiser, LNBL, 7/2010
 */


// headers

// verification routines
char check3DCentralDiffs();
char check3DDivergence() ;
char check3DCurl() ;
char check3DGradient();
char check3DLaplacian();
char checkHeatEqnExplicit();
char checkHeatEqnExplicitVector();
char checkHeatEqnImplicit();
char checkHeatEqnImplicitVector();
char checkHeatEqnSpectral();
char checkHeatEqnMultiGrid();
char checkSpMVLaplacian();



int main(){


	int i;
	int numTests = 12;
	char *pass = (char *) malloc(numTests * sizeof(char));
	char allPass = 1;

	double startTime, endTime; 
	startTime = read_timer(); 

	printf("Beginning structured grid tests.\n\n") ;

	pass[0] = check3DCentralDiffs() ;
	pass[1] = check3DDivergence() ;
	pass[2] = check3DCurl() ;
	pass[3] = check3DGradient() ;
	pass[4] = check3DLaplacian() ;
	pass[5] = checkHeatEqnExplicit() ;
	pass[6] = checkHeatEqnExplicitVector() ;
	pass[7] = checkHeatEqnImplicit() ;
	pass[8] = checkHeatEqnImplicitVector() ;
	pass[9] = checkHeatEqnSpectral() ;
	pass[10] = checkHeatEqnMultiGrid() ;
	pass[11] = checkSpMVLaplacian() ;

	endTime = read_timer(); 
	printf("Total time = %f seconds.\n\n", endTime - startTime) ;
	
	
	// Check whether all tests passed.
	for(i=0; i<numTests; i++){
		if(pass[i]){
			printf("Test %d passed.\n", i) ;
			allPass &= 1;
		}
		else{
			fprintf(stderr, "Test %d failed.\n", i) ;
			allPass = 0;
		}
	}

	if( allPass )
		printf("\nAll Structured Grid tests passed.\n\n") ;
	else
		fprintf(stderr, "\nAt least one Structured Grid test failed!\n") ;

	printf("\nEnd of structured grid tests.\n\n");

	return allPass;
}



char check3DCentralDiffs(){
	/*
	 Check central difference approximations to partial derivatives in 3D.
	 All l2 relative errors and all maximum differences must be under the
	 specified tolerance parameters to pass.

	 Parameters:
	 double maxX, maxY, maxZ   Boundaries of grid.
	 int n     Number of grid points to use in x direction.
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	printf("Central differences test:\n\n");

	double maxX, maxY, maxZ ;
	maxX = 1.0;
	maxY = 1.0;
	maxZ = 1.0;

	int n = 100 ;

	int type = 0;
	double toughness = 0.5 ;

	double h = maxX / (double) (n+1) ;

	double tolNorm = 10e-5;
	double tolMax = 10e-5;

	char pass = 1;
	
	double startTime, endTime; 

	// declarations and initializations.
	grid testGrid = initZeroGrid3d(n, n, n, h) ;

	grid dx = initZeroGrid3d(testGrid.m, testGrid.n, testGrid.p, h) ;
	grid dy = initZeroGrid3d(testGrid.m, testGrid.n, testGrid.p, h) ;
	grid dz = initZeroGrid3d(testGrid.m, testGrid.n, testGrid.p, h) ;

	grid dxSoln = initZeroGrid3d(testGrid.m, testGrid.n, testGrid.p, h) ;
	grid dySoln = initZeroGrid3d(testGrid.m, testGrid.n, testGrid.p, h) ;
	grid dzSoln = initZeroGrid3d(testGrid.m, testGrid.n, testGrid.p, h) ;

	double relErrDx ;
	double relErrDy ;
	double relErrDz ;

	double maxDiffDx ;
	double maxDiffDy ;
	double maxDiffDz ;


	for(type = 0; type<3; type++){

		// set initial grid
		setGrid3d(&testGrid, centralDiffsTestFn, type, toughness) ;

		setZeroGrid3d(&dx) ;
		setZeroGrid3d(&dy) ;
		setZeroGrid3d(&dz) ;

		startTime = read_timer(); 
		
		// evaluate derivatives
		centralDifference3D(testGrid, &dx, &dy, &dz) ;

		endTime = read_timer(); 
		printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
		
		// set solutions and compare solutions.
		setGrid3d(&dxSoln, centralDiffsDxSoln, type, toughness) ;
		setGrid3d(&dySoln, centralDiffsDySoln, type, toughness) ;
		setGrid3d(&dzSoln, centralDiffsDzSoln, type, toughness) ;

		relErrDx = l2RelativeErrGrid(dx, dxSoln) ;
		relErrDy = l2RelativeErrGrid(dy, dySoln) ;
		relErrDz = l2RelativeErrGrid(dz, dzSoln) ;

		printf("relErrDx = %e\nrelErrDy = %e\nrelErrDz = %e\n", relErrDx, relErrDy, relErrDz) ;

		if((relErrDx < tolNorm) && (relErrDy < tolNorm) && (relErrDz < tolNorm)){
			printf("Central difference relative error tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Central difference relative error tests failed on type %d.\n\n", type) ;
			pass = 0;
		}

		maxDiffDx = maxDiffGrid(dx, dxSoln) ;
		maxDiffDy = maxDiffGrid(dy, dySoln) ;
		maxDiffDz = maxDiffGrid(dz, dzSoln) ;

		printf("maxDiffDx = %e\nmaxDiffDy = %e\nmaxDiffDz = %e\n", maxDiffDx, maxDiffDy, maxDiffDz) ;

		if((maxDiffDx < tolMax) && (maxDiffDy < tolMax) && (maxDiffDz < tolMax)){
			printf("Central difference max difference tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Central difference max difference tests failed on type %d.\n\n", type) ;
			pass = 0;
		}

	}

	if(pass)
		printf("Central difference tests passed.\n\n") ;
	else
		fprintf(stderr, "Central difference tests failed.\n\n") ;
	printf("\n") ;

	// free resources
	freeGrid(testGrid);
	freeGrid(dx) ;
	freeGrid(dy) ;
	freeGrid(dz) ;
	freeGrid(dxSoln) ;
	freeGrid(dySoln) ;
	freeGrid(dzSoln) ;

	return pass;
}


char check3DDivergence(){
	/*
	 Check central difference approximations to divergence in 3D.
	 All l2 relative errors and all maximum differences must be under the
	 specified tolerance parameters to pass.

	 Parameters:
	 double maxX, maxY, maxZ   Boundaries of grid.
	 int n     Number of grid points to use in x direction.
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.	  
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	printf("Divergence test:\n\n");

	double maxX, maxY, maxZ ;
	maxX = 1.0;
	maxY = 1.0;
	maxZ = 1.0;

	int type = 0;
	double toughness = 0.25 ;

	int n = 100 ;
	double h = maxX / (double) (n+1) ;

	double tolNorm = 10e-5;
	double tolMax = 10e-5;

	double startTime, endTime;

	char pass = 1;

	vectorField testVectorField = initZeroVectorField3d(n,n,n,h);
	grid div = initZeroGrid3d(n,n,n,h);
	grid divSoln = initZeroGrid3d(n,n,n,h);

	double maxDiff;
	double relErr;

	for(type = 0; type < 3; type++){

		setVectorField3d(&testVectorField, divergenceTestFn, type, toughness) ;
		setZeroGrid3d(&div) ;

		startTime = read_timer(); 
		
		// evaluate
		divergence(testVectorField, &div) ;

		endTime = read_timer(); 
		printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
		
		setGrid3d(&divSoln, divergenceSoln, type, toughness) ;

		maxDiff = maxDiffGrid(div, divSoln) ;

		printf("maxDiff = %e\n", maxDiff) ;

		if(maxDiff < tolMax){
			printf("Divergence max difference tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Divergence max difference tests failed on type %d.\n\n", type) ;
			pass = 0;
		}

		// if analytic solution implies a divergence free vector field, ignore relative error
		if(normGrid(divSoln) != 0){
			relErr = l2RelativeErrGrid(div, divSoln) ;

			printf("relErr = %e\n", relErr) ;

			if(relErr < tolNorm) {
				printf("Divergence relative error tests passed on type %d.\n\n", type) ;
			}
			else{
				fprintf(stderr, "Divergence relative error tests failed on type %d.\n\n", type) ;
				pass = 0;
			}
		}
		else{
			printf("Analytic solution has norm exactly zero, which implies divergence free vector field on type %d.\n", type);
			printf("Relative error test skipped.\n\n");
		}

	}

	if(pass)
		printf("Divergence tests passed.\n\n") ;
	else
		fprintf(stderr, "Divergence tests failed.\n\n") ;
	printf("\n") ;

	// free resources
	freeVectorField(testVectorField);
	freeGrid(div) ;
	freeGrid(divSoln) ;

	return pass;
}


char check3DCurl(){
	/*
	 Check central difference approximations to curl in 3D.
	 All l2 relative errors and all maximum differences must be under the
	 specified tolerance parameters to pass.

	 Parameters:
	 double maxX, maxY, maxZ   Boundaries of grid.
	 int n     Number of grid points to use in x direction.
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	printf("Curl test:\n\n");

	double maxX, maxY, maxZ ;
	maxX = 1.0;
	maxY = 1.0;
	maxZ = 1.0;

	int type = 0;
	double toughness = 0.25 ;

	int n = 100 ;
	double h = maxX / (double) (n+1) ;

	double tolNorm = 10e-5;
	double tolMax = 10e-5;

	char pass = 1;
	
	double startTime, endTime;

	vectorField testVectorField = initZeroVectorField3d(n,n,n,h);
	vectorField curlGuess = initZeroVectorField3d(n,n,n,h);
	vectorField curlAnalyticSoln = initZeroVectorField3d(n,n,n,h);

	vector relErr;
	vector maxDiff;
	vector temp ;

	for(type=0; type<3; type++){

		setVectorField3d(&testVectorField, curlTestFn, type, toughness) ;
		setZeroVectorField3d(&curlGuess) ;

		startTime = read_timer(); 
		
		// evaluate
		curl(testVectorField, &curlGuess) ;

		endTime = read_timer(); 
		printf("Elapsed time = %f seconds.\n", endTime - startTime) ;

		setVectorField3d(&curlAnalyticSoln, curlSoln, type, toughness) ;

		maxDiff = maxDiffVectorField(curlGuess, curlAnalyticSoln) ;

		printf("maxDiff.x = %e\nmaxDiff.y = %e\nmaxDiff.z = %e\n", maxDiff.x, maxDiff.y, maxDiff.z) ;

		if((maxDiff.x < tolMax) && (maxDiff.y < tolMax) && (maxDiff.z < tolMax)){
			printf("Curl max difference tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Curl max difference tests failed on type %d.\n\n", type) ;
			pass = 0;
		}

		// if maxDiff suggests a conservative vector field, ignore relative error
		temp = normVectorField(curlAnalyticSoln) ;
		if( !( (temp.x == 0.0) && (temp.y == 0.0) && (temp.z == 0.0) )){
			relErr = l2RelativeErrVectorField(curlGuess, curlAnalyticSoln) ;

			printf("relErr.x = %e\nrelErr.y = %e\nrelErr.z = %e\n", relErr.x, relErr.y, relErr.z) ;

			if((relErr.x < tolNorm) && (relErr.y < tolNorm) && (relErr.z < tolNorm)){
				printf("Curl relative error tests passed on type %d.\n\n", type) ;
			}
			else{
				fprintf(stderr, "Curl relative error tests failed on type %d.\n\n", type) ;
				pass = 0;
			}
		}
		else{
			printf("Analytic solution is exactly zero, implying a conservative vector field on type %d.\n", type);
			printf("Relative error test skipped.\n\n");
		}

	}

	if(pass)
		printf("Curl tests passed.\n\n") ;
	else
		fprintf(stderr, "Curl tests failed.\n\n") ;
	printf("\n") ;

	// free resources
	freeVectorField(testVectorField) ;
	freeVectorField(curlGuess) ;
	freeVectorField(curlAnalyticSoln) ;

	return pass;
}




char check3DGradient(){
	/*
	 Check central difference approximations to gradient in 3D.
	 All l2 relative errors and all maximum differences must be under the
	 specified tolerance parameters to pass.

	 Parameters:
	 double maxX, maxY, maxZ   Boundaries of grid.
	 int n     Number of grid points to use in x direction.
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	printf("Gradient test:\n\n");

	double maxX, maxY, maxZ ;
	maxX = 1.0;
	maxY = 1.0;
	maxZ = 1.0;

	int type = 0;
	double toughness = 0.5 ;

	int n = 100 ;
	double h = maxX / (double) (n+1) ;

	double tolNorm = 10e-5;
	double tolMax = 10e-5;

	char pass = 1;
	
	double startTime, endTime;

	grid testGrid = initZeroGrid3d(n,n,n,h) ;

	vectorField gradGuess = initZeroVectorField3d(n,n,n,h);
	vectorField gradAnalyticSoln = initZeroVectorField3d(n,n,n,h);
	vector relErr ;
	vector maxDiff ;

	for(type = 0; type < 3; type++){

		setGrid3d(&testGrid, gradientTestFn, type, toughness) ;
		setZeroVectorField3d(&gradGuess) ;

		startTime = read_timer(); 
		
		// evaluate
		gradient(testGrid, &gradGuess) ;
		
		endTime = read_timer(); 
		printf("Elapsed time = %f seconds.\n", endTime - startTime) ;

		setVectorField3d(&gradAnalyticSoln, gradientSoln, type, toughness) ;

		relErr = l2RelativeErrVectorField(gradGuess, gradAnalyticSoln) ;

		printf("relErr.x = %e\nrelErr.y = %e\nrelErr.z = %e\n", relErr.x, relErr.y, relErr.z) ;

		if((relErr.x < tolNorm) && (relErr.y < tolNorm) && (relErr.z < tolNorm)){
			printf("Gradient relative error tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Gradient relative error tests failed on type %d.\n\n", type) ;
			pass = 0;
		}


		maxDiff = maxDiffVectorField(gradGuess, gradAnalyticSoln) ;

		printf("maxDiff.x = %e\nmaxDiff.y = %e\nmaxDiff.z = %e\n", maxDiff.x, maxDiff.y, maxDiff.z) ;

		if((maxDiff.x < tolMax) && (maxDiff.y < tolMax) && (maxDiff.z < tolMax)){
			printf("Gradient max difference tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Gradient max difference tests failed on type %d.\n\n", type) ;
			pass = 0;
		}
	}

	if(pass)
		printf("Gradient tests passed.\n\n") ;
	else
		fprintf(stderr, "Gradient tests failed.\n\n") ;

	printf("\n") ;

	// free resources
	freeGrid(testGrid) ;
	freeVectorField(gradGuess) ;
	freeVectorField(gradAnalyticSoln) ;

	return pass;
}

char check3DLaplacian(){
	/*
	 Check central difference approximations to Laplacian in 3D.
	 All l2 relative errors and all maximum differences must be under the
	 specified tolerance parameters to pass.

	 Parameters:
	 double maxX, maxY, maxZ   Boundaries of grid.
	 int n     Number of grid points to use in x direction.
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	printf("Laplacian test:\n\n");

	double maxX, maxY, maxZ ;
	maxX = 1.0;
	maxY = 1.0;
	maxZ = 1.0;

	int type ;
	double toughness = 0.5 ;

	int n = 100 ;
	double h = maxX / (double) (n+1) ;

	double tolNorm = 10e-5;
	double tolMax = 10e-5;

	char pass = 1;

	double startTime, endTime;
	
	grid testGrid = initZeroGrid3d(n,n,n,h);
	grid laplacianGuess = initZeroGrid3d(n,n,n,h);
	grid laplacianAnalyticSoln = initZeroGrid3d(n,n,n,h);

	double relErr;
	double maxDiff;

	for(type = 0; type < 3; type++){

		setGrid3d(&testGrid, laplacianTestFn, type, toughness) ;

		setZeroGrid3d(&laplacianGuess) ;

		startTime = read_timer(); 
		
		// evaluate
		laplacian(testGrid, &laplacianGuess) ;
		
		endTime = read_timer(); 
		printf("Elapsed time = %f seconds.\n", endTime - startTime) ;

		setGrid3d(&laplacianAnalyticSoln, laplacianSoln, type, toughness) ;


		relErr = l2RelativeErrGrid(laplacianGuess, laplacianAnalyticSoln) ;

		printf("relErr = %e\n", relErr) ;

		if(relErr < tolNorm) {
			printf("Laplacian relative error tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Laplacian relative error tests failed on type %d.\n\n", type) ;
			pass = 0;
		}

		maxDiff = maxDiffGrid(laplacianGuess, laplacianAnalyticSoln) ;

		printf("maxDiff = %e\n", maxDiff) ;

		if(maxDiff < tolMax){
			printf("Laplacian max difference tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Laplacian max difference tests failed on type %d.\n\n", type) ;
			pass = 0;
		}
	}

	if(pass)
		printf("Laplacian tests passed.\n\n") ;
	else
		fprintf(stderr, "Laplacian tests failed.\n\n") ;
	printf("\n") ;

	// free resources
	freeGrid(testGrid);
	freeGrid(laplacianGuess) ;
	freeGrid(laplacianAnalyticSoln) ;

	return pass;
}


char checkHeatEqnExplicit(){
	/*
	 Checks the 3D heat equation.
	 Explicit, stencil based method.

	 Parameters:
	 double dt   Time step.
	 int tSteps  Number of time steps to take.
	 double height Grid height
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	double height = 1.0 / 64.0 ;
	double dt = 0.0001 ;
	int tSteps = 10 ;
	double tolNorm = 10e-4;
	double tolMax = 10e-4;

	printf("Heat equation test. Explicit method:\n\n");

	double startTime, endTime;
	startTime = read_timer(); 
	
	gridArray numericalSoln = solveHeatEquationExplicit(dt, tSteps, height) ;

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	char pass = compareHeatEqnSoln(numericalSoln, tolNorm, tolMax) ;

	freeGridArray(numericalSoln) ;
	return pass;
}


char checkHeatEqnExplicitVector(){
	/*
	 Checks the 3D heat equation.
	 Explicit, matrix based method.

	 Parameters:
	 double dt   Time step.
	 int tSteps  Number of time steps to take.
	 double height Grid height
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	double height = 1.0 / 64.0 ;
	double dt = 0.0001 ;
	int tSteps = 10 ;
	double tolNorm = 10e-4;
	double tolMax = 10e-4;

	printf("Heat equation test. Matrix based explicit method:\n\n");

	double startTime, endTime;
	startTime = read_timer(); 
	
	gridArray numericalSoln = solveHeatEquationExplicitVector(dt, tSteps, height) ;

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	char pass = compareHeatEqnSoln(numericalSoln, tolNorm, tolMax) ;

	freeGridArray(numericalSoln) ;
	return pass;
}

char checkHeatEqnImplicit(){
	/*
	 Checks the 3D heat equation.
	 Implicit, stencil based method.

	 Parameters:
	 double dt   Time step.
	 int tSteps  Number of time steps to take.
	 double height Grid height
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	double height = 1.0 / 128.0 ;
	double dt = 0.001 ;
	int tSteps = 5 ;
	double tolNorm = 5e-2;
	double tolMax = 5e-2;

	printf("Heat equation test. Implicit method:\n\n");

	double startTime, endTime;
	startTime = read_timer(); 
	
	double tol = 2.220446049250313e-16;
	gridArray numericalSoln = solveHeatEquationImplicit(dt, tSteps, height, tol) ;

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	char pass = compareHeatEqnSoln(numericalSoln, tolNorm, tolMax) ;

	freeGridArray(numericalSoln) ;
	return pass;
}

char checkHeatEqnImplicitVector(){
	/*
	 Checks the 3D heat equation.
	 Implicit, matrix based method.

	 Parameters:
	 double dt   Time step.
	 int tSteps  Number of time steps to take.
	 double height Grid height
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	double height = 1.0 / 128.0 ;
	double dt = 0.001 ;
	int tSteps = 5 ;
	double tolNorm = 5e-2;
	double tolMax = 5e-2;

	printf("Heat equation test. Matrix based implicit method:\n\n");

	double startTime, endTime;
	startTime = read_timer(); 
	
	double tol = 2.220446049250313e-16;
	gridArray numericalSoln = solveHeatEquationImplicitVector(dt, tSteps, height, tol) ;

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	char pass = compareHeatEqnSoln(numericalSoln, tolNorm, tolMax) ;

	freeGridArray(numericalSoln) ;
	return pass;
}


char checkHeatEqnSpectral(){
	/*
	 Checks the 3D heat equation.
	 Implicit, stencil based method.

	 Parameters:
	 double dt   Time step.
	 int tSteps  Number of time steps to take.
	 double height Grid height. Must be such that the number of interior points is a power of two
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	double height = 1.0 / (128.0 + 1.0) ;
	double dt = 0.001 ;
	int tSteps = 5 ;
	double tolNorm = 5e-2;
	double tolMax = 5e-2;

	printf("Heat equation test. Spectral method:\n\n");

	double startTime, endTime;
	startTime = read_timer(); 
	
	gridArray numericalSoln = solveHeatEquationSpectral(dt, tSteps, height) ;

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	char pass = compareHeatEqnSoln(numericalSoln, tolNorm, tolMax) ;

	freeGridArray(numericalSoln) ;
	return pass;
}


char checkHeatEqnMultiGrid(){
	/*
	 Checks the 3D heat equation.
	 Implicit, stencil based method.

	 Parameters:
	 double dt   Time step.
	 int tSteps  Number of time steps to take.
	 double height Grid height
	 int depth          Number of steps to descend in V-cycle.
     int maxItDown      Maximum number of iterations to perform on down cycle.
     int maxItUp        Maximum number of iterations to perform on up cycle.
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	double height = 1.0 / 128.0 ;
	double dt = 0.001 ;
	int tSteps = 5 ;
	int depth = 4 ;
	int maxItDown = 15 ;
	int maxItUp = 30 ;
	double tolNorm = 5e-2;
	double tolMax = 5e-2;


	printf("Heat equation test. MultiGrid method:\n\n");

	double startTime, endTime;
	startTime = read_timer(); 
	
	double tol = 1e-10;
	gridArray numericalSoln = solveHeatEquationMultiGrid(dt, tSteps, height, tol, depth, maxItDown, maxItUp) ;

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	char pass = compareHeatEqnSoln(numericalSoln, tolNorm, tolMax) ;

	freeGridArray(numericalSoln) ;
	return pass;
}

char checkSpMVLaplacian(){
	/*
	 Check central difference approximations to Laplacian in 3D.
	 All l2 relative errors and all maximum differences must be under the
	 specified tolerance parameters to pass.

	 Parameters:
	 double maxX, maxY, maxZ   Boundaries of grid.
	 int n     Number of grid points to use in x direction.
	 double tolNorm   Tolerance for l2 relative error.
	 double tolMax    Tolerance for max difference.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	printf("SpMV test of Laplacian:\n\n");

	double maxX, maxY, maxZ ;
	maxX = 1.0;
	maxY = 1.0;
	maxZ = 1.0;

	int type ;
	double toughness = 1.0 ;

	int n = 100 ;
	double h = maxX / (double) (n+1) ;

	double tolNorm = 10e-5;
	double tolMax = 10e-5;

	char pass = 1;

	double startTime, endTime;
	
	csrMatrix laplacianMatrix = getLaplacianMatrix(n,h) ;

	
	grid testGrid = initZeroGrid3d(n,n,n,h);
	grid laplacianGuess = initZeroGrid3d(n,n,n,h);
	grid laplacianAnalyticSoln = initZeroGrid3d(n,n,n,h);

	double *inputBuf = (double *) malloc(n*n*n * (sizeof(double))) ;
	double *outputBuf = (double *) malloc(n*n*n * (sizeof(double))) ;

	double relErr;
	double maxDiff;

	for(type = 0; type < 2; type++){

		setGrid3d(&testGrid, laplacianTestFnHomogeneous, type, toughness) ;

		setZeroGrid3d(&laplacianGuess) ;

		// evaluate
		gridToVector(testGrid, inputBuf);
		startTime = read_timer();
		
		spmv(laplacianMatrix, inputBuf, outputBuf) ;
		
		endTime = read_timer(); 
		printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
		
		vectorToGrid(&laplacianGuess, outputBuf) ;

		
		setGrid3d(&laplacianAnalyticSoln, laplacianSolnHomogeneous, type, toughness) ;

		relErr = l2RelativeErrGrid(laplacianGuess, laplacianAnalyticSoln) ;

		printf("relErr = %e\n", relErr) ;

		if(relErr < tolNorm) {
			printf("Laplacian relative error tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Laplacian relative error tests failed on type %d.\n\n", type) ;
			pass = 0;
		}

		maxDiff = maxDiffGrid(laplacianGuess, laplacianAnalyticSoln) ;

		printf("maxDiff = %e\n", maxDiff) ;

		if(maxDiff < tolMax){
			printf("Laplacian max difference tests passed on type %d.\n\n", type) ;
		}
		else{
			fprintf(stderr, "Laplacian max difference tests failed on type %d.\n\n", type) ;
			pass = 0;
		}
	}

	if(pass)
		printf("Laplacian tests passed.\n\n") ;
	else
		fprintf(stderr, "Laplacian tests failed.\n\n") ;
	printf("\n") ;

	// free resources
	free(inputBuf);
	free(outputBuf);
	freeGrid(testGrid);
	freeGrid(laplacianGuess) ;
	freeGrid(laplacianAnalyticSoln) ;

	return pass;
}


