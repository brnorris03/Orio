

// Header file for matrix generators.

// Alex Kaiser, LBNL, 2010

csrMatrix getStructuredLowerTriangularCSR(int n, int nnz, double distribution[]) ;
csrMatrix getSymmetricDiagonallyDominantCSR(int n, int nnz, double distribution[]) ;

csrMatrix getHeatEqnMatrix(int n, double lambda) ;
csrMatrix getHeatEqnMatrixImplicit(int n, double lambda) ;

csrMatrix getLaplacianMatrix(int n, double height);

void getRandomIndicesFromBand(int *rowInd, int *colInd, int n, int bandNumber, int width) ;

