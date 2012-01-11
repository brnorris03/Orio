void stencil2d(int n, double **grid, double **vec, double **y) {

    /*@ begin StencilMV (
            gridSize = n;
            degreesOfFreedom = 2;
            stencilPoints = 5;
     )
     @*/

    int i, j;
    for (i = 1; i < n + 1; i++) { // assume zero-filled padding on the boundaries
        for (j = 1; j < n + 1; j++) {
            y[i][j] += grid[i][j]   * vec[i][j]   +
                       grid[i-1][j] * vec[i-1][j] +
                       grid[i+1][j] * vec[i+1][j] +
                       grid[i][j-1] * vec[i][j-1] +
                       grid[i][j+1] * vec[i][j+1];
        }
    }

    /*@ end @*/
}
