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
  
  int stride = 4;
  int trows = TROWS+stride;
  int tcols = TCOLS;
  int brows = BROWS;
  int bcols = BCOLS;
  
  double *y = (double *) malloc(trows*sizeof(double));
  double *x = (double *) malloc(tcols*sizeof(double));
  double *aa = (double *) malloc(trows*(bcols+stride)*sizeof(double));
  int *ai = (int *) malloc(trows*sizeof(double));
  int *aj = (int *) malloc(trows*(bcols+stride)*sizeof(double));
  int *node_sizes = (int *) malloc(trows*sizeof*(double));
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

  long long orio_total_cycles = 0;
  long long orio_avg_cycles;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++) 
    {  
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

