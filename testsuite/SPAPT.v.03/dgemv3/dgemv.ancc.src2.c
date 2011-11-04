/* void DGEMV(int A_nrows, int A_ncols, double A[A_ncols][A_nrows], int e_nrows, double e[e_nrows], int w_nrows, double w[w_nrows], int x_nrows, double x[x_nrows], int p_nrows, double p[p_nrows], int y_nrows, double y[y_nrows], int z_nrows, double z[z_nrows]){
	int ii,i,j;
	double t2[A_nrows];
*/

/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'gcc -O3 -fopenmp -DDYNAMIC';
    arg libs = '-lm -lrt';
  }
   
  def performance_counter         
  {
    arg repetitions = 35;
  }
  
  def performance_params
  {
    # Cache tiling
    param T1_I[] = [1,16,32,64,128,256,512];
    param T1_J[] = [1,16,32,64,128,256,512];
    param T1_Ia[] = [1,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,64,128,256,512,1024,2048];
    
    param T3_I[] = [1,16,32,64,128,256,512];
    param T3_J[] = [1,16,32,64,128,256,512];
    param T3_Ia[] = [1,64,128,256,512,1024,2048];
    param T3_Ja[] = [1,64,128,256,512,1024,2048];

    
    param T5_I[] = [1,16,32,64,128,256,512];
    param T5_J[] = [1,16,32,64,128,256,512];
    param T5_Ia[] = [1,64,128,256,512,1024,2048];
    param T5_Ja[] = [1,64,128,256,512,1024,2048];

    
    param T7_I[] = [1,16,32,64,128,256,512];
    param T7_J[] = [1,16,32,64,128,256,512];
    param T7_Ia[] = [1,64,128,256,512,1024,2048];
    param T7_Ja[] = [1,64,128,256,512,1024,2048];


    # Unroll-jam 
    param U1_I[]  = [1,4,8,12,16]; 
    param U1_J[]  = [1,4,8,12,16];
    param U2_I[]  = [1,4,8,12,16];
    param U3_I[]  = [1,4,8,12,16];
    param U3_J[]  = [1,4,8,12,16];
    param U4_I[]  = [1,4,8,12,16];
    param U5_I[]  = [1,4,8,12,16];
    param U5_J[]  = [1,4,8,12,16];
    param U6_I[]  = [1,4,8,12,16];
    param U7_I[]  = [1,4,8,12,16];
    param U7_J[]  = [1,4,8,12,16];
    param U8_I[]  = [1,4,8,12,16];
    param U9_I[]  = [1,4,8,12,16];
    param U10_I[]  = [1,4,8,12,16];

    # Register tiling
    param RT1_I[] = [1,8,32];
    param RT1_J[] = [1,8,32];
    param RT3_I[] = [1,8,32];
    param RT3_J[] = [1,8,32];
    param RT5_I[] = [1,8,32];
    param RT5_J[] = [1,8,32];
    param RT7_I[] = [1,8,32];
    param RT7_J[] = [1,8,32];
    

    # Scalar replacement
    param SCR[]  = [False,True];

    # Vectorization
    param VEC1[] = [False,True];
    param VEC2[] = [False,True];
    param VEC3[] = [False,True];
    param VEC4[] = [False,True];
    param VEC5[] = [False,True];
    param VEC6[] = [False,True];
    param VEC7[] = [False,True];
    param VEC8[] = [False,True];
    param VEC9[] = [False,True];
    param VEC10[] = [False,True];

    # Parallelization
    # param OMP[] = [True,False];

    # Constraints
    constraint tileI1 = ((T1_Ia == 1) or (T1_Ia % T1_I == 0));
    constraint tileJ1 = ((T1_Ja == 1) or (T1_Ja % T1_J == 0));
    constraint tileI3 = ((T3_Ia == 1) or (T3_Ia % T3_I == 0));
    constraint tileJ3 = ((T3_Ja == 1) or (T3_Ja % T3_J == 0));
    constraint tileI5 = ((T5_Ia == 1) or (T5_Ia % T5_I == 0));
    constraint tileJ5 = ((T5_Ja == 1) or (T5_Ja % T5_J == 0));
    constraint tileI7 = ((T7_Ia == 1) or (T7_Ia % T7_I == 0));
    constraint tileJ7 = ((T7_Ja == 1) or (T7_Ja % T7_J == 0));

    
    constraint reg_capacity_1 = (RT1_I*RT1_J <= 150);
    constraint reg_capacity_3 = (RT3_I*RT3_J <= 150);
    constraint reg_capacity_5 = (RT5_I*RT5_J <= 150);
    constraint reg_capacity_7 = (RT7_I*RT7_J <= 150);
    
    constraint unroll_limit_1 = (U1_I == 1) or (U1_J == 1);
    constraint unroll_limit_3 = (U3_I == 1) or (U3_J == 1);
    constraint unroll_limit_5 = (U5_I == 1) or (U5_J == 1);
    constraint unroll_limit_7 = (U7_I == 1) or (U7_J == 1);

  }  
  
  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 10000;
  }
  
  def input_params
  {
    param M = 10000;
    param N = 10000;
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

int i,j, k;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


/*@ begin Loop (
  
  transform Composite(
      tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),
             (('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj')],
      unrolljam = (['i','j'],[U1_I,U1_J]),
      scalarreplace = (SCR, 'double'),
      regtile = (['i','j'],[RT1_I,RT1_J]),
      vector = (VEC1, ['ivdep','vector always'])
   )
    for (i = 0; i <= A_ncols-1; i++) 
        for (j = 0; j <= A_nrows-1; j++) 
            t2[j] = t2[j] + A[i][j]*x[i]; 


  transform Composite(
      unrolljam = (['i'],[U2_I]),
      vector = (VEC2, ['ivdep','vector always'])
   )
   for (i = 0; i <= A_nrows-1; i++) 
        y[i] = t2[i]+y[i]; 

   transform Composite(
      tile = [('i',T3_I,'ii'),('j',T3_J,'jj'),
             (('ii','i'),T3_Ia,'iii'),(('jj','j'),T3_Ja,'jjj')],
      unrolljam = (['i','j'],[U3_I,U3_J]),
      scalarreplace = (SCR, 'double'),
      regtile = (['i','j'],[RT3_I,RT3_J]),
      vector = (VEC3, ['ivdep','vector always'])
   )
    for (i = 0; i <= A_ncols-1; i++) 
       for (j = 0; j <= A_nrows-1; j++) 
            t6[j] = t6[j] + A[i][j]*w[i]; 

  transform Composite(
      unrolljam = (['i'],[U4_I]),
      vector = (VEC4, ['ivdep','vector always'])
   )
   for (i = 0; i <= A_nrows-1; i++) 
        z[i] = t6[i]+z[i]; 


    	
    transform Composite(
      tile = [('i',T5_I,'ii'),('j',T5_J,'jj'),
             (('ii','i'),T5_Ia,'iii'),(('jj','j'),T5_Ja,'jjj')],
      unrolljam = (['i','j'],[U5_I,U5_J]),
      scalarreplace = (SCR, 'double'),
      regtile = (['i','j'],[RT5_I,RT5_J]),
      vector = (VEC5, ['ivdep','vector always'])
   )
    for (i = 0; i <= A_ncols-1; i++) 
        for (j = 0; j <= A_nrows; j++) 
            t10[j] = t10[j] + A[i][j]*e[i]; 

  transform Composite(
      unrolljam = (['i'],[U6_I]),
      vector = (VEC6, ['ivdep','vector always'])
   )
    for (i = 0; i <= A_nrows-1; i++) 
        p[i] = t10[i]+p[i]; 


     transform Composite(
      tile = [('i',T7_I,'ii'),('j',T7_J,'jj'),
             (('ii','i'),T7_Ia,'iii'),(('jj','j'),T7_Ja,'jjj')],
      unrolljam = (['i','j'],[U7_I,U7_J]),
      scalarreplace = (SCR, 'double'),
      regtile = (['i','j'],[RT7_I,RT7_J]),
      vector = (VEC7, ['ivdep','vector always'])
   )
    for (i = 0; i <= A_ncols-1; i++) 
       for (j = 0; j <= A_nrows-1; j++) {
			t10[j] = t10[j] + e[i]*A[i][j]; 
			t6[j] = t6[j] + w[i]*A[i][j]; 
			t2[j] = t2[j] + A[i][j]*x[i]; 
		}


  transform Composite(
      unrolljam = (['i'],[U8_I]),
      vector = (VEC8, ['ivdep','vector always'])
   )
     for (i = 0; i <= A_nrows-1; i++) 
       y[i] = t2[i]+y[i]; 

  transform Composite(
      unrolljam = (['i'],[U9_I]),
      vector = (VEC9, ['ivdep','vector always'])
   )
     for (i = 0; i <= A_nrows-1; i++) 
       z[i] = t6[i]+z[i]; 


  transform Composite(
      unrolljam = (['i'],[U10_I]),
      vector = (VEC10, ['ivdep','vector always'])
   )
     for (i = 0; i <= A_nrows-1; i++) 
       p[i] = t10[i]+p[i]; 

) @*/

/* Noop */

/*@ end @*/

/*@ end @*/
