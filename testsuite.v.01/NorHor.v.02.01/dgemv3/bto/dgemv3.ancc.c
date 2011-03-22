/* void DGEMV(int A_nrows, int A_ncols, double A[A_ncols][A_nrows], int e_nrows, double e[e_nrows], int w_nrows, double w[w_nrows], int x_nrows, double x[x_nrows], int p_nrows, double p[p_nrows], int y_nrows, double y[y_nrows], int z_nrows, double z[z_nrows]){
	int ii,i,j;
	double t2[A_nrows];
*/

/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'gcc -O3 -lm';
  }
   
  def performance_counter         
  {
    arg method = 'basic timer';
    arg repetitions = 35;
  }
  
  def performance_params
  {

    param U1ia[] = [1,4,8,16,32,64,128,256,512];
    param U1ja[] = [1,4,8,16,32,64,128,256,512];
    param U2ia[] = [1,4,8,16,32,64,128,256,512];
    param U3ia[] = [1,4,8,16,32,64,128,256,512];
    param U3ja[] = [1,4,8,16,32,64,128,256,512];
    param U4ia[] = [1,4,8,16,32,64,128,256,512];
    param U5ia[] = [1,4,8,16,32,64,128,256,512];
    param U5ja[] = [1,4,8,16,32,64,128,256,512];
    param U6ia[] = [1,4,8,16,32,64,128,256,512];
    param U7ia[] = [1,4,8,16,32,64,128,256,512];
    param U7ja[] = [1,4,8,16,32,64,128,256,512];
    param U8ia[] = [1,4,8,16,32,64,128,256,512];
    param U9ia[] = [1,4,8,16,32,64,128,256,512];
    param U10ia[] = [1,4,8,16,32,64,128,256,512];


    param U1i[] = range(1,31);
    param U1j[] = range(1,31);
    param U2i[] = range(1,31);
    param U3i[] = range(1,31);
    param U3j[] = range(1,31);
    param U4i[] = range(1,31);
    param U5i[] = range(1,31);
    param U5j[] = range(1,31);
    param U6i[] = range(1,31);
    param U7i[] = range(1,31);
    param U7j[] = range(1,31);
    param U8i[] = range(1,31);
    param U9i[] = range(1,31);
    param U10i[] = range(1,31);

    param PAR1i[] = [True, False];
    param PAR1j[] = [True, False];
    param PAR2i[] = [True, False];
    param PAR3i[] = [True, False];
    param PAR3j[] = [True, False];
    param PAR4i[] = [True, False];
    param PAR5i[] = [True, False];
    param PAR5j[] = [True, False];
    param PAR6i[] = [True, False];
    param PAR7i[] = [True, False];
    param PAR7j[] = [True, False];
    param PAR8i[] = [True, False];
    param PAR9i[] = [True, False];
    param PAR10i[] = [True, False];

    param SCR[] = [True, False];
    param VEC[] = [True, False];

  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 10;
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
    regtile = (['i','j'],[U1ia,U1ja]),  
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U1i, parallelize=PAR1i)
    for (i = 0; i <= A_ncols-1; i++) 
      transform UnrollJam(ufactor=U1j, parallelize=PAR1j)
        for (j = 0; j <= A_nrows-1; j++) 
            t2[j] = t2[j] + A[i][j]*x[i]; 

  transform Composite (
    regtile = (['i'], [U2ia]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U2i, parallelize=PAR2i)
    for (i = 0; i <= A_nrows-1; i++) 
        y[i] = t2[i]+y[i]; 

  transform Composite(
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j'],[U3ia,U3ja]),  
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U3i, parallelize=PAR3i)
    for (i = 0; i <= A_ncols-1; i++) 
       transform UnrollJam(ufactor=U3j, parallelize=PAR3j)    
       for (j = 0; j <= A_nrows-1; j++) 
            t6[j] = t6[j] + A[i][j]*w[i]; 

  transform Composite (
    regtile = (['i'], [U4ia]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U4i, parallelize=PAR4i)
  for (i = 0; i <= A_nrows-1; i++) 
        z[i] = t6[i]+z[i]; 


  transform Composite(
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j'],[U5ia,U5ja]),     
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U5i, parallelize=PAR5i)
  for (i = 0; i <= A_ncols-1; i++) 
        transform UnrollJam(ufactor=U5j, parallelize=PAR5j)
        for (j = 0; j <= A_nrows; j++) 
            t10[j] = t10[j] + A[i][j]*e[i]; 


  transform Composite (
    regtile = (['i'], [U6ia]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U6i, parallelize=PAR6i)
  for (i = 0; i <= A_nrows-1; i++) 
    p[i] = t10[i]+p[i]; 


  transform Composite(
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j'],[U7ia,U7ja]),     
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U7i, parallelize=PAR7i)
  for (i = 0; i <= A_ncols-1; i++) 
    transform UnrollJam(ufactor=U7j, parallelize=PAR7j)
    for (j = 0; j <= A_nrows-1; j++) {
			t10[j] = t10[j] + e[i]*A[i][j]; 
			t6[j] = t6[j] + w[i]*A[i][j]; 
			t2[j] = t2[j] + A[i][j]*x[i]; 
		}

  transform Composite (
    regtile = (['i'], [U8ia]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U8i, parallelize=PAR8i)
	for (i = 0; i <= A_nrows-1; i++) 
		y[i] = t2[i]+y[i]; 

  transform Composite (
    regtile = (['i'], [U9ia]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U9i, parallelize=PAR9i)
	for (i = 0; i <= A_nrows-1; i++) 
		z[i] = t6[i]+z[i]; 


  transform Composite (
    regtile = (['i'], [U10ia]),
    vector = (VEC, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=U10i, parallelize=PAR10i)
	for (i = 0; i <= A_nrows-1; i++) 
		p[i] = t10[i]+p[i]; 

) @*/

/* Noop */

/*@ end @*/

/*@ end @*/
