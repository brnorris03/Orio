/* void DGEMV(int A_nrows, int A_ncols, double A[A_ncols][A_nrows], int e_nrows, double e[e_nrows], int w_nrows, double w[w_nrows], int x_nrows, double x[x_nrows], int p_nrows, double p[p_nrows], int y_nrows, double y[y_nrows], int z_nrows, double z[z_nrows]){
	int ii,i,j;
	double t2[A_nrows];
*/

/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'gcc -O3 -lm';
    #arg build_command  = 'icc -fast -openmp -I/usr/local/icc/include -lm';
    #arg libs = '-lm';
  }
   
  def performance_counter         
  {
    arg method = 'basic timer';
    arg repetitions = 35;
  }
  
  def performance_params
  {
    param U1i[] = range(1,31);
    param U1j[] = range(1,31);
    param U3i[] = range(1,31);
    param U3j[] = range(1,31);
    param U3ja[] = range(1,31);
    param U6i[] = range(1,31);
    param U6j[] = range(1,31);
    param U6ja[] = range(1,31);
    param U1[] = range(1,31);
    param U2[] = range(1,31);
    param U4[] = range(1,31);
    param U5[] = range(1,31);
    param U7[] = range(1,31);
    #param U7[] = [1]; 
    param U8[] = range(1,31);
    param U9[] = range(1,31);

    param SCR[] = [True, False];
    param VEC[] = [True, False];
    param OMP[] = [True, False];
    param PAR1i[] = [True, False];
    param PAR1j[] = [True, False];

    #constraint reg_capacity = (2*U_I*U_J + 2*U_I + 2*U_J <= 130);

    #constraint copy_limitA = ((not ACOPY_A) or (ACOPY_A and 
    #                          (T1_I if T1_I>1 else T2_I)*(T1_J if T1_J>1 else T2_J) <= 512*512));
  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 5000;
  }
  
  def input_params
  {
    param M = 1000;
    param N = 1000;
  } 
 
  def input_vars {
    decl int A_nrows = M;
    decl int A_ncols = N;
    decl dynamic double A[M][N] = random;
    decl dynamic double e[N] = random;
    decl dynamic double x[N] = random;
    decl dynamic double w[N] = random;
    decl dynamic double y[M] = random;
    decl dynamic double z[M] = random;
    decl dynamic double p[M] = 0;
    decl dynamic double t2[M] = 0;
    decl dynamic double t6[M] = 0;
    decl dynamic double t10[M] = 0;
  }            
) @*/

int i, j;
register int it, jt;

/*@ begin Loop (
  transform Composite (
    scalarreplace = (SCR, 'double'),
    regtile = (['j'], [U1]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U1j, parallelize=PAR1j)
    for (i = 0; i <= A_ncols-1; i++) 
      transform UnrollJam(ufactor=U1i, parallelize=PAR1i)
        for (j = 0; j <= A_nrows-1; j++) 
            t2[j] = t2[j] + A[i][j]*x[i]; 

  transform Unroll (ufactor=U2)
    for (i = 0; i <= A_nrows-1; i++) 
        y[i] = t2[i]+y[i]; 

  transform Composite(
    scalarreplace = (SCR, 'double'),
    regtile = (['j'], [U3ja]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
    for (i = 0; i <= A_ncols-1; i++) 
        for (j = 0; j <= A_nrows-1; j++) 
            t6[j] = t6[j] + A[i][j]*w[i]; 

  transform Unroll(ufactor=U4)
    for (i = 0; i <= A_nrows-1; i++) 
        z[i] = t6[i]+z[i]; 
    for (i = 0; i <= A_ncols-1; i++) 
        for (j = 0; j <= A_nrows; j++) 
            t10[j] = t10[j] + A[i][j]*e[i]; 

  transform Unroll(ufactor=U5)
    for (i = 0; i <= A_nrows-1; i++) 
        p[i] = t10[i]+p[i]; 

  transform Composite(
    scalarreplace = (SCR, 'double'),
    regtile = (['j'], [U6ja]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
	for (i = 0; i <= A_ncols-1; i++) 
		for (j = 0; j <= A_nrows-1; j++) {
			t10[j] = t10[j] + e[i]*A[i][j]; 
			t6[j] = t6[j] + w[i]*A[i][j]; 
			t2[j] = t2[j] + A[i][j]*x[i]; 
		}
	
  transform UnrollJam(ufactor=U7)
	for (i = 0; i <= A_nrows-1; i++) 
		y[i] = t2[i]+y[i]; 

  transform UnrollJam(ufactor=U8)
	for (i = 0; i <= A_nrows-1; i++) 
		z[i] = t6[i]+z[i]; 

  transform UnrollJam(ufactor=U9)
	for (i = 0; i <= A_nrows-1; i++) 
		p[i] = t10[i]+p[i]; 
) @*/

/* Noop */

/*@ end @*/

/*@ end @*/
