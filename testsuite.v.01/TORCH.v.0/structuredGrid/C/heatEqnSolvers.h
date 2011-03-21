

/*
 A selection of methods for solving the three dimensional heat equation.

 Alex Kaiser, LBNL, 7/2010
 */




// initialization
gridArray initGridsHeatEqn(int tSteps, double height, double maxX, double maxY, double maxZ, double dt);
gridArray initSolnHeatEqn(int tSteps, double height, double maxX, double maxY, double maxZ, double dt);

// Solvers and solver wrappers
gridArray solveHeatEquationExplicit(double dt, int tSteps, double height);
gridArray solveHeatEquationImplicit(double dt, int tSteps, double height, double tol) ;
gridArray solveHeatEquationExplicitVector(double dt, int tSteps, double height) ;
gridArray solveHeatEquationImplicitVector(double dt, int tSteps, double height, double tol) ;
gridArray solveHeatEquationSpectral(double dt, int tSteps, double height);
gridArray solveHeatEquationMultiGrid(double dt, int tSteps, double height, double tol, int depth, int maxItDown, int maxItUp);

// linear equation solvers
grid conjugateGradientGrid(grid b, grid guess, double dt, double maxIt, grid x, double tol) ;
grid gaussSeidelGrid(grid b, grid guess, double dt, double maxIt, grid x, double tol);
grid multiGridVCycle(grid b, grid guess, mgGridArray currentMgArray, grid x, int maxItDown, int maxItUp, double dt, double tol);

// util
void fineToCoarse(grid fine, grid *coarse);
void coarseToFine(grid coarse, grid *fine);


// verification
char compareHeatEqnSoln(gridArray numericalSolution, double tolNorm, double tolMax) ;

