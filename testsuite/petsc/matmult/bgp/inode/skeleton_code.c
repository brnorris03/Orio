#include <stdio.h>  
#include <stdlib.h> 
#include <sys/time.h>   

#include <papi.h>

/*@ global @*/  

#define max(x,y) ((x) > (y)? (x) : (y))
#define min(x,y) ((x) < (y)? (x) : (y))

int main()  
{
  /*@ prologue @*/  
  
  int trows = TROWS;
  int tcols = TCOLS;
  int brows = BROWS;
  int bcols = BCOLS;
  
  if (trows % brows != 0) {
    printf("error: the global number of rows must be divisible by the block number of rows.\n");
    exit(1);
  }

  double *y = (double *) malloc(trows*sizeof(double));
  double *x = (double *) malloc(tcols*sizeof(double));
  double *aa = (double *) malloc(trows*(bcols+100)*sizeof(double));
  int *ai = (int *) malloc(trows*sizeof(int));
  int *aj = (int *) malloc(trows*(bcols+100)*sizeof(int));
  int *node_sizes = (int *) malloc(trows*sizeof(int));
  int node_max = 0;
  {
    int i,j,k,ind;
    for (i=0; i<trows; i++)
      y[i]=0;
    for (i=0; i<tcols; i++)
      x[i]=(i%10)+1;
    ind=0;
    for (i=0; i<trows; i+=brows) {
      node_max++;
      node_sizes[i/brows]=brows;
      for (j=0; j<brows; j++) {
	ai[i+j]=ind;
	int start_pos=max((((int)(1.0*i/trows*tcols))-5*bcols),0);
	int mult=((5*(j+3))%3)+1;
	for (k=0; k<bcols; k++) {
	  aa[ind]=(((j+1)*k)%10)+j+2;
	  if (j==0)
	    aj[ind]=start_pos+k*mult;
	  else
	    aj[ind]=aj[ind-bcols];
	  ind++;
	}
      }
    }
  }
  double *y_o = y;
  double *x_o = x;
  double *aa_o = aa;
  int *ai_o = ai;
  int *aj_o = aj;
  int *node_sizes_o = node_sizes;

  long long orio_total_cycles = 0;
  long long orio_avg_cycles;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++) 
    {  
      y=y_o; x=x_o; aa=aa_o; ai=ai_o; aj=aj_o; node_sizes=node_sizes_o;
      
      int err, EventSet = PAPI_NULL;
      long long CtrValues[1];
      err = PAPI_library_init(PAPI_VER_CURRENT);
      if (err != PAPI_VER_CURRENT) {
	printf("PAPI library initialization error!\n");
	exit(1);
      }
      if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
	printf("Failed to create PAPI Event Set\n");
	exit(1);
      }
      if (PAPI_query_event(PAPI_TOT_CYC) == PAPI_OK)
	err = PAPI_add_event(EventSet, PAPI_TOT_CYC);
      if (err != PAPI_OK) {
	printf("Failed to add PAPI event\n");
	exit(1);
      }
      PAPI_start(EventSet);
      
      /*@ tested code @*/ 
      
      PAPI_stop(EventSet, &CtrValues[0]);
      orio_total_cycles += CtrValues[0];
    }  
  
  orio_avg_cycles = orio_total_cycles / REPS;
  printf("%ld\n", orio_avg_cycles);

  /*@ epilogue @*/  

  return y[0]; // to avoid the dead code elimination
}   

