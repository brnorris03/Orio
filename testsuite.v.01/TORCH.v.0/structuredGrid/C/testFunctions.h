

/*
 Test functions header for structured grid kernels and tests.

 Alex Kaiser, LBNL, 7/2010
 */

// for central diffs tests
double centralDiffsTestFn(double x, double y, double z, int type, double toughness) ;

double centralDiffsDxSoln(double x, double y, double z, int type, double toughness) ;
double centralDiffsDySoln(double x, double y, double z, int type, double toughness) ;
double centralDiffsDzSoln(double x, double y, double z, int type, double toughness) ;

// for divergence tests
vector divergenceTestFn(double x, double y, double z, int type, double toughness) ;
double divergenceSoln(double x, double y, double z, int type, double toughness) ;


// for curl tests
vector curlTestFn(double x, double y, double z, int type, double toughness) ;
vector curlSoln(double x, double y, double z, int type, double toughness) ;

// for gradient tests
double gradientTestFn(double x, double y, double z, int type, double toughness) ;
vector gradientSoln(double x, double y, double z, int type, double toughness) ;

// for laplacian tests
double laplacianTestFn(double x, double y, double z, int type, double toughness) ;
double laplacianSoln(double x, double y, double z, int type, double toughness) ;

// for homogeneous laplacian with SpMV
double laplacianTestFnHomogeneous(double x, double y, double z, int type, double toughness);
double laplacianSolnHomogeneous(double x, double y, double z, int type, double toughness);


// for heat equation solves
double heatEqnInitialConds(double x, double y, double z, int type, double toughness);
double heatEqnAnalyticSoln(double x, double y, double z, double t, int type, double toughness);


// for debugging
double plane(double x, double y, double z, int type, double toughness) ;

