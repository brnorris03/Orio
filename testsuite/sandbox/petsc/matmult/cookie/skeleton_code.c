#include <stdio.h>  
#include <stdlib.h> 
#include <sys/time.h>   

#include <omp.h>
#include <papi.h>

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
 
/*@ global @*/  
 
int main()  
{    
  /*@ prologue @*/  
 
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

  printf("{'/*@ coordinate @*/' : %ld}", orio_avg_cycles);

  /*@ epilogue @*/  

  return y[0]; // to avoid the dead code elimination
}   

