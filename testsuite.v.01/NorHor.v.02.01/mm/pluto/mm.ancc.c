
/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'icc -fast -openmp ';
    arg libs = '-lm -lrt';
  }
   
  def performance_counter         
  {
    arg repetitions = 35;
  }
  
  def performance_params
  {
    param T1_I[] = [1,16,32,64];
    param T1_J[] = [1,16,32,64];
    param T1_K[] = [1,16,32,64];
    param T2_I[] = [1,16,32,64];
    param T2_J[] = [1,16,32,64];
    param T2_K[] = [1,16,32,64];

    param ACOPY_A[] = [True,False];
    param ACOPY_B[] = [True,False];
    param ACOPY_C[] = [True,False];

    param U_I[] = range(1,31);
    param U_J[] = range(1,31);
    param U_K[] = range(1,31);

    param SCREP[] = [False,True];
    param VEC[] = [False,True];
    param OMP[] = [False,True];

    param PAR1i[] = [False,True];
    param PAR1j[] = [False,True];
    param PAR1k[] = [False,True];



    param PERMUTS[] = [
                       
		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'i','j','k'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'i','j','k'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'i','j','k'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'i','j','k'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'i','j','k'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'i','j','k'),
		       
		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'j','i','k'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'j','i','k'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'j','i','k'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'j','i','k'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'j','i','k'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'j','i','k'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'i','k','j'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'i','k','j'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'i','k','j'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'i','k','j'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'i','k','j'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'i','k','j'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'k','i','j'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'k','i','j'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'k','i','j'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'k','i','j'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'k','i','j'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'k','i','j'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'j','k','i'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'j','k','i'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'j','k','i'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'j','k','i'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'j','k','i'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'j','k','i'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'k','j','i'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'k','j','i'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'k','j','i'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'k','j','i'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'k','j','i'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'k','j','i'),

		       ];

    constraint tileI = ((T2_I == 1) or (T2_I % T1_I == 0));
    constraint tileJ = ((T2_J == 1) or (T2_J % T1_J == 0));
    constraint tileK = ((T2_K == 1) or (T2_K % T1_K == 0));

    constraint reg_capacity = (U_I*U_J + U_I*U_K + U_J*U_K <= 150);
    constraint unroll_limit = ((U_I == 1) or (U_J == 1) or (U_K == 1));

    constraint copy_limitA = ((not ACOPY_A) or (ACOPY_A and 
                              (T1_I if T1_I>1 else T2_I)*(T1_K if T1_K>1 else T2_K) <= 512*512));
    constraint copy_limitB = ((not ACOPY_B) or (ACOPY_B and 
                              (T1_K if T1_K>1 else T2_K)*(T1_J if T1_J>1 else T2_J) <= 512*512));
    constraint copy_limitC = ((not ACOPY_C) or (ACOPY_C and
                              (T1_I if T1_I>1 else T2_I)*(T1_J if T1_J>1 else T2_J) <= 512*512));
    constraint copy_limit = (not ACOPY_A or not ACOPY_B or not ACOPY_C);
  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 1000;

  }
  
  def input_params
  {
    param CONT = 10000;
    param NCONT = 100;
    param M = 100;
    param N = 100;
    param K = 10000;
  }
  def input_vars
  { 
    decl static double A[M][K] = random;
    decl static double B[K][N] = random;
    decl static double C[M][N] = 0;
  }            
) @*/

int i, j, k;
int ii, jj, kk;
int iii, jjj, kkk;

/*@ begin Loop(
  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T2_I,'iii'),(('jj','j'),T2_J,'jjj'),(('kk','k'),T2_K,'kkk')],
    permut = [PERMUTS],
    arrcopy = [(ACOPY_A,'A[i][k]',[(T1_I if T1_I>1 else T2_I),(T1_K if T1_K>1 else T2_K)],'_copy'),
               (ACOPY_B,'B[k][j]',[(T1_K if T1_K>1 else T2_K),(T1_J if T1_J>1 else T2_J)],'_copy'),
	       (ACOPY_C,'C[i][j]',[(T1_I if T1_I>1 else T2_I),(T1_J if T1_J>1 else T2_J)],'_copy')],
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k,A_copy,B_copy,C_copy)')
  )
  for(i=0; i<=M-1; i++) 
    for(j=0; j<=N-1; j++)   
      for(k=0; k<=K-1; k++) 
        C[i][j] = C[i][j] + A[i][k] * B[k][j]; 

) @*/
transform Unrolljam(ufactor=U_I,parallelize=PAR1i)
  for(i=0; i<=M-1; i++) 
    transform Unrolljam(ufactor=U_J,parallelize=PAR1j)
    for(j=0; j<=N-1; j++)   
      transform Unrolljam(ufactor=U_K,parallelize=PAR1k)
      for(k=0; k<=K-1; k++) 
        C[i][j] = C[i][j] + A[i][k] * B[k][j]; 

/*@ end @*/
/*@ end @*/
