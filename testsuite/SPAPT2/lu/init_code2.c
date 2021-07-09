void init_input_vars() {
  int i, j, k;

  /* have to initialize this matrix properly to prevent                                              
   * division by zero                                                                                 
   */
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      L[i*N+j] = 0.0;
      U[i*N+j] = 0.0;
    }
  }

  for (i=0; i<N; i++) {
    for (j=0; j<=i; j++) {
      L[i*N+j] = i+j+1;
      U[j*N+i] = i+j+1;
    }
  }

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      for (k=0; k<N; k++) {
	A[i*N+j] += L[i*N+k]*U[k*N+j];
      }
    }
  }
}
